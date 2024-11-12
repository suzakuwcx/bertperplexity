import torch
from torch.nn import functional as F
from typing import Optional
from transformers import BertForPreTraining

class BertPerplexity(BertForPreTraining):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        mask_id: Optional[int] = 103, # BertTokenizer [MASK] id
    ):
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, device=super().device)
        
        if isinstance(attention_mask, list):
            attention_mask = torch.tensor(attention_mask, device=super().device)

        if isinstance(token_type_ids, list):
            token_type_ids = torch.tensor(token_type_ids, device=super().device)

        # remove [CLS] and [SEP], and first word and last word(the perplexity at first and last will higher than normal)
        sentence_length = len(input_ids) - 4

        # [ 1, len(input_ids) ]
        _input_ids = input_ids.unsqueeze(0)
        # [ sentence_length, len(input_ids) ]
        _input_ids = _input_ids.expand([sentence_length, len(input_ids)]).clone()

        for i in range(sentence_length):
            _input_ids[i][i + 2] = mask_id

        _attention_mask = attention_mask.unsqueeze(0).expand([sentence_length, len(input_ids)])
        _token_type_ids = token_type_ids.unsqueeze(0).expand([sentence_length, len(input_ids)])

        output = super().forward(input_ids=_input_ids, attention_mask=_attention_mask, token_type_ids=_token_type_ids)
        raw_pred = output.prediction_logits
        pred = F.softmax(raw_pred, dim=-1)

        prob = torch.zeros(sentence_length, device=super().device)

        for i in range(sentence_length):
            origin_token_ids = input_ids[i + 2]
            prob[i] = pred[i][i + 2][origin_token_ids]

        return (-prob.log2().mean()).exp2()

