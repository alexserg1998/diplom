from transformers import BertPreTrainedModel
from transformers import FNetModel

import torch
import torch.nn as nn
import torch.nn.functional as F


class FnetModule(BertPreTrainedModel):
    def __init__(self, config):
        super(FnetModule, self).__init__(config)
        self.bert = FNetModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, segment_ids=None, input_mask=None):
        outputs = self.bert(input_ids, segment_ids, input_mask)
        sequence_output, pooled_output = outputs[:2]

        input_mask_expanded = input_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        sum_embeddings = torch.sum(sequence_output * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, 1e-9)
        mean_pooling_out = sum_embeddings / sum_mask  # [batch_size, hidden_size]
        return mean_pooling_out


class SFnet(nn.Module):
    def __init__(self, model_path=None, config=None):
        super(SFnet, self).__init__()
        self.num_labels = 2
        self.fnet_module = FnetModule.from_pretrained(model_path)
        self.linear1 = nn.Linear(768 * 3, 768)
        self.linear2 = nn.Linear(768, self.num_labels)

    def forward(self, x_input_ids=None, x_segment_ids=None, x_input_mask=None,
                y_input_ids=None, y_segment_ids=None, y_input_mask=None, labels=None, train_sbert=True):
        if train_sbert:
            u = self.fnet_module(x_input_ids, x_segment_ids, x_input_mask)
            v = self.fnet_module(y_input_ids, y_segment_ids, y_input_mask)
            uv = torch.sub(u, v)
            uv_abs = torch.abs(uv)
            output = torch.cat([u, v, uv_abs], dim=-1)

            output = F.relu(self.linear1(output))
            logits = self.linear2(output)
            return logits

        else:
            return self.fnet_module(x_input_ids, x_segment_ids, x_input_mask)
