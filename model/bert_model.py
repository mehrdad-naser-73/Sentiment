import torch
from transformers import BertTokenizer, BertModel, AutoConfig, AutoTokenizer, AutoModel, AdamW
import torch.nn as nn


class BertClassifier(nn.Module):

    def __init__(self, n_class, model_name):

        super(BertClassifier, self).__init__()
        if model_name == 'parsbert':
            pretrain_path = "HooshvareLab/bert-base-parsbert-uncased"
        elif model_name == 'multi':
            pretrain_path = 'bert-base-multilingual-cased'
        D_in, D_out, H = 768, n_class, 256
        if model_name == 'multi':
            self.bert = BertModel.from_pretrained(pretrain_path)
        else:
            self.bert = AutoModel.from_pretrained(pretrain_path, config=AutoConfig.from_pretrained(pretrain_path))

        self.classifier = nn.Sequential(

            #             nn.Linear(D_in, H),
            #             nn.ReLU(),
            #             nn.Dropout(0.5),
            nn.Linear(D_in, D_out)
        )

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        last_hidden_state_cls = outputs[1]
        logits = self.classifier(last_hidden_state_cls)

        return logits