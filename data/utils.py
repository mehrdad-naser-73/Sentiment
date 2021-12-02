import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


class Data:

    def __init__(self, texts, labels, max_len, tokenizer, batch_size):

        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def encode_texts(self):

        input_ids = []
        att_masks = []

        for text in self.texts:

            tokens = ['[CLS]'] + self.tokenizer.tokenize(text) + ['[SEP]']
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            seq_len = len(token_ids)
            while len(token_ids) < self.max_len:
                token_ids.append(0)
            token_ids = token_ids[:self.max_len]

            token_ids = torch.tensor(token_ids).long()
            att_mask = torch.zeros(token_ids.size()).long()
            att_mask[:seq_len] = 1

            input_ids.append(token_ids)
            att_masks.append(att_mask)

        return input_ids, att_masks

    def get_data_loader(self, mode="train"):

        input_ids, att_masks = self.encode_texts()
        labels = [torch.LongTensor([int(label)]) for label in self.labels]
        data = []
        for ids, masks, label in zip(input_ids, att_masks, labels):
            data.append([ids, masks, label])
        sampler = RandomSampler(data) if mode == "train" else SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)

        return dataloader



def load_file(path):
    texts, labels = [], []
    insts = open(path, 'r', encoding='utf-8').read().split('\n')
    for inst in insts[1:]:
        if len(inst) == 0:
            continue
        sep = inst.split('\t')
        texts.append(sep[1])
        labels.append(sep[-1])
    return texts, labels