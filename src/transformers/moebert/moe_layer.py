import copy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers.moebert.utils import FeedForward
from copy import deepcopy

ONNX_EXPORT = os.environ.get('MOEBERT_EXPORT_ONNX', '0') == '1'
CHECK_LATENCY = os.environ.get('MOEBERT_CHECK_LATENCY', '0') == '1'
assert not (ONNX_EXPORT and CHECK_LATENCY)


class MoELayer(nn.Module):
    def __init__(self, hidden_size, num_experts, expert, route_method, vocab_size, hash_list):
        nn.Module.__init__(self)
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.vocab_size = vocab_size
        self.hash_list = hash_list
        self.experts = nn.ModuleList([copy.deepcopy(expert) for i in range(num_experts)])
        self.route_method = route_method
        if route_method in ["gate-token", "gate-sentence"]:
            self.gate = nn.Linear(hidden_size, num_experts, bias=False).float()
        elif route_method == "hash-random":
            self.hash_list = self._random_hash_list(vocab_size)
        elif route_method == "hash-balance":
            self.hash_list = self._balance_hash_list(hash_list)
        else:
            raise KeyError("Routing method not supported.")

    def _random_hash_list(self, vocab_size):
        hash_list = torch.randint(low=0, high=self.num_experts, size=(vocab_size,))
        self.register_buffer('hash_list_buffer', hash_list)
        return self.hash_list_buffer

    def _balance_hash_list(self, hash_list):
        with open(hash_list, "rb") as file:
            result = pickle.load(file)
        result = torch.tensor(result, dtype=torch.int64)
        return result

    def _forward_gate_token(self, x):
        bsz, seq_len, dim = x.size()

        x = x.view(-1, dim)
        logits_gate = self.gate(x)
        prob_gate = F.softmax(logits_gate, dim=-1) # [bxl, 4]
        gate = torch.argmax(prob_gate, dim=-1)

        order = gate.argsort(0)
        num_tokens = F.one_hot(gate, self.num_experts).gt(0).sum(0)
        gate_load = num_tokens.clone()
        x = x[order]  # reorder according to expert number
        x = x.split(num_tokens.tolist(), dim=0)  # a list of length self.num_experts

        # compute the load balancing loss
        P = prob_gate.mean(0)
        temp = num_tokens.float()
        f = temp / temp.sum(0, keepdim=True)
        balance_loss = self.num_experts * torch.sum(P * f)

        prob_gate = prob_gate.gather(dim=1, index=gate.unsqueeze(1))
        prob_gate = prob_gate[order]
        prob_gate = prob_gate.split(num_tokens.tolist(), dim=0)

        def forward_expert(input_x, prob_x, expert_idx): # nxd, nx1
            output = self.experts[expert_idx].forward(input_x)
            final_output = output * (torch.ones_like(prob_x) - prob_x.detach() + prob_x)
            for i in range(len(self.experts)):
                if i == expert_idx:
                    continue
                output = self.experts[i].forward(input_x)
                final_output = final_output + output * (torch.zeros_like(prob_x) - prob_x.detach() + prob_x)
            return final_output

        x = [forward_expert(x[i], prob_gate[i], i) for i in range(self.num_experts)]
        x = torch.vstack(x)
        x = x[order.argsort(0)]  # restore original order
        x = x.view(bsz, seq_len, dim)

        return x, balance_loss, gate_load

    def _forward_gate_token_onnx(self, x):
        bsz, seq_len, dim = x.size()
        x = x.view(-1, dim)
        logits_gate = self.gate(x) # [bxL, 4]
        gate = torch.argmax(logits_gate, dim=-1) #[bxl]

        if not hasattr(self, 'helper_orders') or self.helper_orders.numel() != seq_len * bsz:
            self.helper_orders = torch.arange(seq_len * bsz).to(x.device)
        orders = self.helper_orders
        gate_load = torch.tensor(0.0)
        balance_loss = torch.tensor(0.0)

        ids0 = (gate == 0)
        x_l0 = self.experts[0](x[ids0])
        order_l0 = orders[ids0]

        ids1 = (gate == 1)
        x_l1 = self.experts[1](x[ids1])
        order_l1 = orders[ids1]

        ids2 = (gate == 2)
        x_l2 = self.experts[2](x[ids2])
        order_l2 = orders[ids2]

        ids3 = (gate == 3)
        x_l3 = self.experts[3](x[ids3])
        order_l3 = orders[ids3]

        x = torch.cat([x_l0, x_l1, x_l2, x_l3], dim=0)
        order = torch.cat([order_l0, order_l1, order_l2, order_l3], dim=0)
        x = x[order.argsort(0)]  # restore original order
        x = x.view(bsz, seq_len, dim)
        return x, balance_loss, gate_load


    def _forward_gate_sentence(self, x, attention_mask):
        x_masked = x * attention_mask.unsqueeze(-1)
        x_average = x_masked.sum(1) / attention_mask.unsqueeze(-1).sum(1)
        logits_gate = self.gate(x_average)
        prob_gate = F.softmax(logits_gate, dim=-1)
        gate = torch.argmax(prob_gate, dim=-1)

        order = gate.argsort(0)
        num_sentences = F.one_hot(gate, self.num_experts).gt(0).sum(0)
        gate_load = num_sentences.clone()
        x = x[order]  # reorder according to expert number
        x = x.split(num_sentences.tolist(), dim=0)  # a list of length self.num_experts

        # compute the load balancing loss
        P = prob_gate.mean(0)
        temp = num_sentences.float()
        f = temp / temp.sum(0, keepdim=True)
        balance_loss = self.num_experts * torch.sum(P * f)

        prob_gate = prob_gate.gather(dim=1, index=gate.unsqueeze(1))
        prob_gate = prob_gate[order]
        prob_gate = prob_gate.split(num_sentences.tolist(), dim=0)

        def forward_expert(input_x, prob_x, expert_idx):
            input_x = self.experts[expert_idx].forward(input_x)
            input_x = input_x * prob_x.unsqueeze(-1)
            return input_x

        result = []
        for i in range(self.num_experts):
            if x[i].size(0) > 0:
                result.append(forward_expert(x[i], prob_gate[i], i))

        result = torch.vstack(result)
        result = result[order.argsort(0)]  # restore original order

        return result, balance_loss, gate_load

    def _forward_sentence_single_expert(self, x, attention_mask):
        x_masked = x * attention_mask.unsqueeze(-1)
        x_average = x_masked.sum(1) / attention_mask.unsqueeze(-1).sum(1)
        logits_gate = self.gate(x_average)
        prob_gate = F.softmax(logits_gate, dim=-1)
        gate = torch.argmax(prob_gate, dim=-1)

        gate_load = F.one_hot(gate, self.num_experts).gt(0).sum(0)
        x = self.experts[gate.cpu().item()].forward(x)
        return x, 0.0, gate_load

    def _forward_sentence_single_expert_onnx(self, x, attention_mask):
        x_masked = x * attention_mask.unsqueeze(-1)
        x_average = x_masked.sum(1) / attention_mask.unsqueeze(-1).sum(1)  # Bxdim
        logits_gate = self.gate(x_average)
        gate = torch.argmax(logits_gate, dim=-1).squeeze(0)
        if torch.eq(gate, 0):
            x = self.experts[0](x)
        elif torch.eq(gate, 1):
            x = self.experts[1](x)
        elif torch.eq(gate, 2):
            x = self.experts[2](x)
        else:  # torch.eq(gate, 3):
            x = self.experts[3](x)
        # x = self.experts[gate](x)
        return x, 0.0, 0.0

    def _forward_hash(self, x, input_ids):
        bsz, seq_len, dim = x.size()

        x = x.view(-1, dim)
        self.hash_list = self.hash_list.to(x.device)
        gate = self.hash_list[input_ids.view(-1)]

        order = gate.argsort(0)
        num_tokens = F.one_hot(gate, self.num_experts).gt(0).sum(0)
        gate_load = num_tokens.clone()
        x = x[order]  # reorder according to expert number
        x = x.split(num_tokens.tolist(), dim=0)  # a list of length self.num_experts

        x = [self.experts[i].forward(x[i]) for i in range(self.num_experts)]
        x = torch.vstack(x)
        x = x[order.argsort(0)]  # restore original order
        x = x.view(bsz, seq_len, dim)

        return x, 0.0, gate_load

    # def _forward_hash_onnx_old(self, x, input_ids):  # still has bug about argsort
    #     bsz, seq_len, dim = x.size()

    #     x = x.view(-1, dim)
    #     self.hash_list = self.hash_list.to(x.device)
    #     gate = self.hash_list[input_ids.view(-1)]
    #     # order = torch.argsort(gate, dim=0, stable=True) # values are unique. The order must be stable. !! # TODO(yujie)
    #     order = torch.argsort(gate, dim=0)  # values are unique. The order must be stable. !! # TODO(yujie)
    #     # num_tokens = F.one_hot(gate, self.num_experts).gt(0).sum(0)
    #     # gate_load = num_tokens.clone()
    #     gate_load = 0.0
    #     x = [self.experts[i](x[gate == i]) for i in range(self.num_experts)]
    #     x = torch.cat(x, dim=0)
    #     x = x[order.argsort(0)]  # restore original order
    #     x = x.view(bsz, seq_len, dim)
    #     return x, 0.0, gate_load

    def _forward_hash_onnx(self, x, input_ids):  # still has bug about argsort
        bsz, seq_len, dim = x.size()
        x = x.view(-1, dim)
        self.hash_list = self.hash_list.to(x.device)
        gate = self.hash_list[input_ids.view(-1)]
        # order = torch.argsort(gate, dim=0, stable=True) # values are unique. The order must be stable. !! # TODO(yujie)
        # order = torch.argsort(gate, dim=0)  # values are unique. The order must be stable. !! # TODO(yujie)
        if not hasattr(self, 'helper_orders') or self.helper_orders.numel() != seq_len * bsz:
            self.helper_orders = torch.arange(seq_len * bsz).to(x.device)
        # num_tokens = F.one_hot(gate, self.num_experts).gt(0).sum(0)
        # gate_load = num_tokens.clone()
        orders = self.helper_orders
        gate_load = 0.0

        ids0 = (gate == 0)
        x_l0 = self.experts[0](x[ids0])
        order_l0 = orders[ids0]

        ids1 = (gate == 1)
        x_l1 = self.experts[1](x[ids1])
        order_l1 = orders[ids1]
    
        ids2 = (gate == 2)
        x_l2 = self.experts[2](x[ids2])
        order_l2 = orders[ids2]

        ids3 = (gate == 3)
        x_l3 = self.experts[3](x[ids3])
        order_l3 = orders[ids3]

        x = torch.cat([x_l0, x_l1, x_l2, x_l3], dim=0)
        order = torch.cat([order_l0, order_l1, order_l2, order_l3], dim=0)
        x = x[order.argsort(0)]  # restore original order
        x = x.view(bsz, seq_len, dim)
        return x, 0.0, gate_load

    def forward(self, x, input_ids, attention_mask):
        if self.route_method == "gate-token":
            if ONNX_EXPORT:
                x, balance_loss, gate_load = self._forward_gate_token_onnx(x)
            else:
                x, balance_loss, gate_load = self._forward_gate_token(x)
        elif self.route_method == "gate-sentence":
            if x.size(0) == 1:
                if ONNX_EXPORT:
                    x, balance_loss, gate_load = self._forward_sentence_single_expert_onnx(x, attention_mask)
                else:
                    x, balance_loss, gate_load = self._forward_sentence_single_expert(x, attention_mask)
            else:
                x, balance_loss, gate_load = self._forward_gate_sentence(x, attention_mask)
        elif self.route_method in ["hash-random", "hash-balance"]:
            if ONNX_EXPORT:
                x, balance_loss, gate_load = self._forward_hash_onnx(x, input_ids)
            else:
                x, balance_loss, gate_load = self._forward_hash(x, input_ids)
        else:
            raise KeyError("Routing method not supported.")

        return x, balance_loss, gate_load


class ScriptedMoE_sentence(nn.Module):
    def __init__(self, moe_layer: MoELayer) -> None:
        super().__init__()
        self.num_experts = moe_layer.num_experts
        self.experts = deepcopy(moe_layer.experts)
        self.route_method = moe_layer.route_method
        route_method = moe_layer.route_method
        self.hidden_size = moe_layer.hidden_size
        hidden_size = moe_layer.hidden_size
        num_experts = moe_layer.num_experts
        vocab_size = moe_layer.vocab_size
        hash_list = deepcopy(moe_layer.hash_list)
        self.gate = deepcopy(moe_layer.gate)

    def forward(self, x, attention_mask):
        if attention_mask is not None:
            x_masked = x * attention_mask.unsqueeze(-1)
            x_average = x_masked.sum(1) / attention_mask.unsqueeze(-1).sum(1)  # Bxdim
        else:
            x_average = x.sum(1)
        logits_gate = self.gate(x_average)
        # prob_gate = F.softmax(logits_gate, dim=-1)
        gate = torch.argmax(logits_gate, dim=-1).squeeze(0)
        if torch.eq(gate, 0):
            x = self.experts[0](x)
        elif torch.eq(gate, 1):
            x = self.experts[1](x)
        elif torch.eq(gate, 2):
            x = self.experts[2](x)
        else:  # torch.eq(gate, 3):
            x = self.experts[3](x)
        return x, torch.tensor(0.0), torch.tensor(0.0)


class ScriptedMoE(nn.Module):
    def __init__(self, moe_layer) -> None:
        super().__init__()
        device = 'cpu'
        self.scripted = torch.jit.script(
            ScriptedMoE_sentence(moe_layer),
            example_inputs=[
                (torch.rand((1, 16, 768), device=device),
                 # torch.zeros((1, 16), device=device).long(),
                 torch.zeros((1, 16), device=device).long(),
                 )],

        )
        # print(self.scripted.code)

    def forward(self, x, input_ids, attention_mask):
        return self.scripted(x, attention_mask)


def create_scripted_moe(model):
    device = list(model.parameters())[0].device
    for bert_layer in model.bert.encoder.layer:
        if bert_layer.experts.route_method == "gate-sentence":
            moe = ScriptedMoE(bert_layer.experts)
            bert_layer.experts = moe
    return model
