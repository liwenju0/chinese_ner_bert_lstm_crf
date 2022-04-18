import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import nn, optim
from torchcrf import CRF

from transformers import RobertaModel



class LstmNerModel(nn.Module):
    def __init__(self, embedding_size=256, num_tags=41,
                 vocab_size=3675, hidden_size=128,
                 batch_first=True, dropout=0.1):
        super(LstmNerModel, self).__init__()
        self.batch_first = batch_first
        self.embedding = nn.Embedding(vocab_size, embedding_size, dtype=torch.float32)

        self.lstm = nn.LSTM(embedding_size, hidden_size // 2,
                            num_layers=2, batch_first=True,
                            bidirectional=True, dropout=dropout)
        for name, param in self.lstm.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)

        self.fc = nn.Linear(hidden_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, input_tensor, seq_lens):
        input_tensor = self.embedding(input_tensor)
        total_length = input_tensor.size(1) if self.batch_first else input_tensor.size(0)
        input_packed = pack_padded_sequence(input_tensor, seq_lens, batch_first=self.batch_first, enforce_sorted=False)
        output_lstm, hidden = self.lstm(input_packed)
        output_lstm, length = pad_packed_sequence(output_lstm, batch_first=self.batch_first, total_length=total_length)
        output_fc = self.fc(output_lstm)
        return output_fc

    def compute_loss(self, input_tensor, tags, seq_lens):
        mask = torch.zeros(input_tensor.shape[:2])
        mask = torch.greater(input_tensor, mask).type(torch.ByteTensor)
        output_fc = self.forward(input_tensor, seq_lens)

        loss = -self.crf(output_fc, tags, mask, reduction='mean')
        return loss

    def decode(self, input_tensor, seq_lens):
        out = self.forward(input_tensor, seq_lens)
        mask = torch.zeros(input_tensor.shape[:2])
        mask = torch.greater(input_tensor, mask).type(torch.ByteTensor)
        predicted_index = self.crf.decode(out, mask)
        return predicted_index



class BertNerModel(nn.Module):
    def __init__(self, embedding_size=768,
                 num_tags=41,
                 hidden_size=128,
                 batch_first=True,
                 dropout=0.1):
        super(BertNerModel, self).__init__()
        self.batch_first = batch_first
        self.model = RobertaModel.from_pretrained("hfl/chinese-roberta-wwm-ext")

        self.lstm = nn.LSTM(embedding_size, hidden_size // 2,
                            num_layers=2, batch_first=True,
                            bidirectional=True, dropout=dropout)
        for name, param in self.lstm.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)

        self.fc = nn.Linear(hidden_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, input_tensor, seq_lens):
        input_tensor = self.model(input_tensor)
        input_tensor = input_tensor.last_hidden_state
        input_packed = pack_padded_sequence(input_tensor, seq_lens, batch_first=self.batch_first, enforce_sorted=False)
        output_lstm, hidden = self.lstm(input_packed)
        output_lstm, length = pad_packed_sequence(output_lstm, batch_first=self.batch_first)
        output_fc = self.fc(output_lstm)
        return output_fc

    def compute_loss(self, input_tensor, tags, seq_lens):
        mask = torch.zeros(input_tensor.shape[:2])
        mask = torch.greater(input_tensor, mask).type(torch.ByteTensor)
        output_fc = self.forward(input_tensor, seq_lens)

        loss = -self.crf(output_fc, tags, mask, reduction='mean')
        return loss

    def decode(self, input_tensor, seq_lens):
        out = self.forward(input_tensor, seq_lens)
        mask = torch.zeros(input_tensor.shape[:2])
        mask = torch.greater(input_tensor, mask).type(torch.ByteTensor)
        predicted_index = self.crf.decode(out, mask)
        return predicted_index
