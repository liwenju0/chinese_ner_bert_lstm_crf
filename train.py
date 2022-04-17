import torch

from data import CustomNerDataset
from model import NerModel
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import collections

train_data = CustomNerDataset("data/cluener_public/train.json")
eval_data = CustomNerDataset("data/cluener_public/dev.json", vocab=train_data.char2id, tags=train_data.label2id)

id2char = {v: k for k, v in train_data.char2id.items()}
id2tag = {v: k for k, v in train_data.label2id.items()}


def len_collate_fn(batch_data):
    chars, labels, seq_lens = [], [], []
    for d in batch_data:
        chars.append(d['chars'])
        labels.append(d['labels'])
        seq_lens.append(d['len_chars'])

    chars = pad_sequence(chars, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    return chars, labels, torch.LongTensor(seq_lens)


train_dataloader = DataLoader(train_data, batch_size=25, shuffle=True, collate_fn=len_collate_fn)
eval_dataloader = DataLoader(eval_data, batch_size=100, collate_fn=len_collate_fn)

vocab_size = len(train_data.char2id)
num_tags = len(train_data.label2id)
lr = 0.001
embedding_size = 256
hidden_size = 128

model = NerModel(embedding_size=embedding_size, hidden_size=hidden_size, vocab_size=vocab_size, num_tags=num_tags)

params = [{"params": [], 'lr': lr}, {'params': [], 'lr': 100 * lr}]
for p in model.named_parameters():
    if "crf" in p[0]:
        params[1]['params'].append(p[1])
    else:
        params[0]['params'].append(p[1])

optimizer = torch.optim.Adam(params)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)


def decode_prediction(chars, tags):
    assert len(chars) == len(tags), "{}{}".format(chars, tags)
    result = collections.defaultdict(set)
    entity = ''
    type1 = ''
    for char, tag in zip(chars, tags):
        if "S" in tag:
            if entity:
                result[type1].add(entity)
            result[tag.split("-")[1]].add(char)
            type1 = ''
            entity = ''
        elif 'B' in tag:
            if entity:
                result[type1].add(entity)
            entity = char
            type1 = tag.split('-')[1]
        elif 'I' in tag:
            type2 = tag.split('-')[1]
            if type1 == type2:
                entity += char
            else:
                entity += '[ERROR]'
        elif 'E' in tag:
            type2 = tag.split('-')[1]
            if entity:
                if type1 == type2:
                    entity += char
                else:
                    entity += '[ERROR]'
                result[type1].add(entity)
                entity = ''
                type1 = ''

        else:
            if entity:
                result[type1].add(entity)
            entity = ''
    if entity:
        result[type1].add(entity)
    return result


def eval(model=model, eval_dataloader=eval_dataloader):
    model.eval()
    result = {}
    for index, (input_tensor, true_tags, seq_lens) in enumerate(eval_dataloader):
        predict_tags = model.decode(input_tensor, seq_lens)
        true_tags = list(true_tags.numpy())
        input_tensor = list(input_tensor.numpy())
        for pre, true, input in zip(predict_tags, true_tags, input_tensor):
            pre = [id2tag[t] for t in pre]
            true = [id2tag[t] for t in true]
            chars = [id2char[c] for c in input if c != 0]
            true = true[:len(chars)]
            pre_result = decode_prediction(chars, pre)
            true_result = decode_prediction(chars, true)
            for type, cnt in pre_result.items():
                if type not in result:
                    result[type] = [0, 0, 0]
                result[type][1] += len(cnt)
                if type in true_result:
                    result[type][0] += len(pre_result[type] & true_result[type])
            for type, cnt in true_result.items():
                if type not in result:
                    result[type] = [0, 0, 0]
                result[type][2] += len(cnt)

    for type, (x, y, z) in result.items():
        X, Y, Z = 1e-10, 1e-10, 1e-10
        X += x
        Y += y
        Z += z

        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        result[type].append(round(precision, 3))
        result[type].append(round(recall, 3))
        result[type].append(round(f1, 3))
    result = [(k, v) for k, v in result.items()]
    macro_f1 = sum([v[1][-1] for v in result])/len(result)
    print("macrof1 {}".format(macro_f1))
    result.sort()
    model.train()
    return result


def train(model=model, train_loader=train_dataloader, optimizer=optimizer, scheduler=scheduler, epoch=10):
    model.train()
    for i in range(epoch):
        epoch_loss = 0
        epoch_count = 0
        before = -1
        for index, (input_tensor, tags, seq_lens) in enumerate(train_loader):
            loss = model.compute_loss(input_tensor, tags, seq_lens)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_count += input_tensor.shape[0]

            if index % 100 == 0:
                print(round(epoch_loss / epoch_count, 3))
                cur = epoch_loss / epoch_count
                if cur < 0.2 and (before - cur) / before > 0.01:
                    result = eval(model, eval_dataloader)
                    print(i, index, result)
                if cur < before:
                    before = cur

        scheduler.step()

if __name__ == '__main__':

    train()
