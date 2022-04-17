import torch
from torch.utils.data import DataLoader, Dataset, sampler
from torch.nn.utils.rnn import pad_sequence


import json

class CustomNerDataset(Dataset):
    def __init__(self, file_path='data/cluener_public/train.json', vocab=None, tags=None):
        self.data = []

        if vocab:
            self.char2id = vocab
        else:
            self.char2id = {'PAD': 0, 'UNK': 1}

        if tags:
            self.label2id = tags
        else:
            self.label2id = {'O': 0}

        for line in open(file_path):
            line = json.loads(line)
            one_example = []

            chars = []

            for c in line['text']:
                chars.append(c)
                if not vocab:
                    if c not in self.char2id:
                        self.char2id[c] = len(self.char2id)
            one_example.append(chars)

            labels = ['O'] * len(chars)
            for k, v in line.get('label', {}).items():
                if not tags:
                    if 'B-' + k not in self.label2id:
                        self.label2id['B-' + k] = len(self.label2id)
                        self.label2id['I-' + k] = len(self.label2id)
                        self.label2id['E-' + k] = len(self.label2id)
                        self.label2id['S-' + k] = len(self.label2id)

                for spans in v.values():
                    for start, end in spans:
                        assert start <= end
                        if start == end:
                            labels[start] = 'S-' + k
                        elif start + 1 == end:
                            labels[start:end + 1] = ['B-' + k, 'E-' + k]
                        else:
                            labels[start] = 'B-' + k
                            for i in range(start + 1, end):
                                labels[i] = 'I-' + k
                            labels[end] = 'E-' + k

            one_example.append(labels)
            self.data.append(one_example)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chars, labels = self.data[idx]
        chars = [self.char2id[c] if c in self.char2id else self.char2id['UNK'] for c in chars]
        labels = [self.label2id[lbl] for lbl in labels]

        return {
            'chars': torch.LongTensor(chars),
            'labels': torch.LongTensor(labels),
            'len_chars': len(chars)
        }


class TrainDevData:
    def __init__(self, train_path="data/cluener_public/train.json",
                 dev_path="data/cluener_public/dev.json", vocab=None):
        self.train_data = CustomNerDataset(train_path, vocab=vocab)
        self.eval_data = CustomNerDataset(dev_path,
                                          vocab=self.train_data.char2id,
                                          tags=self.train_data.label2id)
        self.id2char = {v: k for k, v in self.train_data.char2id.items()}
        self.id2tag = {v: k for k, v in self.train_data.label2id.items()}
        self.vocab_size = len(self.train_data.char2id)
        self.num_tags = len(self.train_data.label2id)

        self.train_dataloader = DataLoader(self.train_data, batch_size=25,
                                           shuffle=True, collate_fn=self.len_collate_fn)
        self.eval_dataloader = DataLoader(self.eval_data, batch_size=100, collate_fn=self.len_collate_fn)


    def len_collate_fn(self, batch_data):
        chars, labels, seq_lens = [], [], []
        for d in batch_data:
            chars.append(d['chars'])
            labels.append(d['labels'])
            seq_lens.append(d['len_chars'])

        chars = pad_sequence(chars, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=0)
        return chars, labels, torch.LongTensor(seq_lens)






if __name__ == '__main__':
    dataset = CustomNerDataset()
    print(len(dataset.char2id))
    print(len(dataset.label2id))
    print(dataset.data[-1])
