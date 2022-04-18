''''
使用了bert+lstm+crf模型。
简化起见，仅使用了bert的vocab，tokenize还是一个个字符。但是效果macro-f1 只能达到0.64
'''

from data import *
from model import BertNerModel
import collections
from transformers import AutoTokenizer

from train_lstm_crf import  decode_prediction

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
vocab = tokenizer.get_vocab()

train_dev_data = TrainDevData(vocab=vocab)
id2tag = train_dev_data.id2tag
id2char = train_dev_data.id2char
train_dataloader = train_dev_data.train_dataloader
eval_dataloader = train_dev_data.eval_dataloader

vocab_size = train_dev_data.vocab_size
num_tags = train_dev_data.num_tags
lr = 2e-5
hidden_size = 128

model = BertNerModel(num_tags=num_tags)

params = [{"params": [], 'lr': lr}, {'params': [], 'lr': 100 * lr}]
for p in model.named_parameters():
    if "crf" in p[0]:
        params[1]['params'].append(p[1])
    else:
        params[0]['params'].append(p[1])

optimizer = torch.optim.Adam(params)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)



def eval(model=model, eval_dataloader=eval_dataloader):
    model.eval()
    result = {}
    for index, (input_tensor, true_tags, seq_lens) in enumerate(eval_dataloader):
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        predict_tags = model.decode(input_tensor)
        true_tags = list(true_tags.numpy())
        input_tensor = list(input_tensor.cpu().numpy())
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
    macro_f1 = sum([v[1][-1] for v in result]) / len(result)
    print("macrof1 {}".format(macro_f1))
    result.sort()
    model.train()
    return result


def train(model=model, train_loader=train_dataloader, optimizer=optimizer, scheduler=scheduler, epoch=10):
    model.train()
    if torch.cuda.is_available():
        model = model.cuda()
    for i in range(epoch):
        epoch_loss = 0
        epoch_count = 0
        before = -1
        for index, (input_tensor, tags, seq_lens) in enumerate(train_loader):
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
                tags = tags.cuda()
            loss = model.compute_loss(input_tensor, tags)
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
