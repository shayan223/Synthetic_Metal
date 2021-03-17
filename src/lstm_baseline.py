import torch
from torch import nn
import pandas as pd
from collections import Counter
import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import filter
import random
data = filter.load_data()
data = data[:2000]
print(len(data))

#use gpu for NN training
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
#print device name to ensure gpu is being used
print(torch.cuda.get_device_name(0))

'''Based on this tutorial https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html'''

class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 3

        n_vocab = len(dataset.uniq_words)
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))

class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            sequence_len
    ):
        self.sequence_length = sequence_len
        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()

        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def load_words(self):

        train_df = pd.DataFrame(data,columns=['song'])
        text = train_df['song'].str.cat(sep=' ')
        return text.split(' ')

    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        return len(self.words_indexes) - self.sequence_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index + self.sequence_length]),
            torch.tensor(self.words_indexes[index + 1:index + self.sequence_length + 1]),
        )


def train(dataset, model, epochs, batch_size, sequence_length):
    model.train()

    dataloader = DataLoader(dataset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training for :", epochs," Epochs")
    for epoch in range(epochs):
        state_h, state_c = model.init_state(sequence_length)

        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

            print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })



def predict(dataset, model, text, next_words=100):
    model.eval()

    words = text.split(' ')
    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])

    return words

def training_tests():
    epochs = 5
    batch_size = 512
    sequence_len = 4


    dataset = Dataset(sequence_len)
    model = Model(dataset)

    train(dataset, model, epochs, batch_size, sequence_len)
    test_output = predict(dataset, model, text='All my life')
    test_output = ' '.join(test_output)
    print(test_output)

    # save model for further evaluation
    torch.save(model.state_dict(), '../data/trained.model')



def model_evaluation():
    epochs = 1
    batch_size = 512
    sequence_len = 4

    dataset = Dataset(sequence_len)
    model = Model(dataset)
    model.load_state_dict(torch.load('../data/trained.model'))

    test_output = predict(dataset, model, text='Have a great day')
    test_output = ' '.join(test_output)
    print(test_output)

def gen_sample_output():
    sequence_len = 4

    dataset = Dataset(sequence_len)
    model = Model(dataset)
    model.load_state_dict(torch.load('../data/trained.model'))

    seed_sent = 'Have a great day'
    samples = []

    for i in range(300):
        try:
            test_output = predict(dataset, model, text=seed_sent)
            test_output = ' '.join(test_output)
            samples.append(test_output)

            #get the last 4 words as seed for the next output
            word_list = test_output.split()
            next_seed = word_list[-4:]
            #convert list of words back to string
            seed_sent = " ".join(next_seed)
        except:
            #if that doesn't work just pick a single random word from the corpus
            seed_sent = random.choice(dataset.uniq_words)
        print(seed_sent)

    corp = pd.DataFrame(samples)
    corp.to_csv('baseline_sample_output')

#training_tests()
#model_evaluation()
gen_sample_output()







