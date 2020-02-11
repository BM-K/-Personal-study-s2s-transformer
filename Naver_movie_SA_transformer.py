import time
import re
import pandas as pd
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torchtext import data
from konlpy.tag import Mecab
from torchtext.data import TabularDataset
from torchtext.data import Iterator, BucketIterator
from torchtext.vocab import Vectors
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

    # input_dim = len(TEXT.vocab)
    # embedding_dim = 160 # kr-data 벡터 길이
    # hidden_dim = 256
    # output_dim = 1 # sentiment analysis

def tokenizer1(text):
    result_text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》;]', '', text)
    a = Mecab().morphs(result_text)
    return( [a[i] for i in range(len(a))] )

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = "Transformer"
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp*fix_length, 1)
        self.init_weights()
        self.dropout = nn.Dropout(dropout)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        hidden = output[0, :, :]
        for i in range(1, output.size(0)):
            hidden = torch.cat((hidden, output[i, :, :]), dim=1)
        hidden = self.dropout(hidden)

        output = self.decoder(hidden)

        return torch.sigmoid(output)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    #round predictions to the closest integer
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        #hidden = model.init_hidden()

        predictions = model(batch.text).view(-1,1).squeeze(1)

        batch.label = batch.label.float()
        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            #hidden = model.init_hidden()

            predictions = model(batch.text).view(-1,1).squeeze(1)

            batch.label = batch.label.float()
            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


ID = data.Field(sequential=False,
                use_vocab=False)
fix_length=20
TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=tokenizer1,
                  lower=True,
                  batch_first=False,
                  fix_length=fix_length,
                  )

LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   is_target=True,
                   )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data, test_data, val_data = TabularDataset.splits(
    path='.', train='naver_movie_train.txt', test='naver_movie_test.txt',
    validation='naver_movie_eval.txt', format='tsv',
    fields=[('id', ID), ('text', TEXT), ('label', LABEL)], skip_header=True
) # train과 text파일이 현재 디렉토리에 있어햐함. 여기서 tsv로 구분된 것 사용. 첫 번째 header는 무시

batch_size = 500
train_loader = BucketIterator(dataset=train_data, batch_size=batch_size, device=device, shuffle=True)
test_loader = BucketIterator(dataset=test_data, batch_size=batch_size, device=device, shuffle=True)
val_loader = BucketIterator(dataset=val_data, batch_size=batch_size, device=device, shuffle=True)

vectors = Vectors(name="kr-projected.txt")

TEXT.build_vocab(train_data, vectors=vectors, min_freq=5, max_size=15000)

if __name__ == '__main__':
    input_dim = len(TEXT.vocab)
    embedding_dim = 160 # kr-data 벡터 길이
    hidden_dim = 256
    output_dim = 1
    dropout = 0.5
    nlayers = 1
    nhead = 1

    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    #model = RNN(input_dim, embedding_dim, hidden_dim, output_dim, batch_size, dropout, PAD_IDX)
    model = TransformerModel(input_dim, embedding_dim, nhead, hidden_dim, nlayers, dropout)

    model.embedding.weight.data.copy_(TEXT.vocab.vectors)
    #UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    #model.embedding.weight.data[UNK_IDX] = torch.zeros(embedding_dim)
    #model.embedding.weight.data[PAD_IDX] = torch.zeros(embedding_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.cuda()
    criterion = nn.BCELoss()

    N_EPOCHS = 20

    best_valid_loss = float('inf')
    x_epoch = []
    drow_loss = []
    drow_acc = []
    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, val_loader, criterion)

        x_epoch.append(epoch)
        drow_loss.append(valid_loss)
        drow_acc.append(valid_acc)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'one-model.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    model.load_state_dict(torch.load('one-model.pt'))

    test_loss, test_acc = evaluate(model, test_loader, criterion)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
