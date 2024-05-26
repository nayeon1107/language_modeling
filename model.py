import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class CharRNN(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, output_size, num_layers=1):

        # write your codes here
        super(CharRNN, self).__init__()

        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.input_size, self.embedding_dim)
        self.rnn = nn.RNN(self.embedding_dim, self.hidden_size, self.num_layers, batch_first=True)     
        
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size // 2, self.output_size)
        

    def forward(self, input, hidden):

        # write your codes here
        embeds = self.embedding(input)
        rnn_output, hidden = self.rnn(embeds, hidden)
        output = rnn_output.contiguous().view(-1, self.hidden_size)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        return output, hidden
    

    def init_hidden(self, batch_size):

        # write your codes here
        initial_hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE)

        return initial_hidden


class CharLSTM(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, output_size, num_layers=1, dropout=0.1):

        # write your codes here
        super(CharLSTM, self).__init__()

        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.input_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.hidden_size // 2, self.output_size)


    def forward(self, input, hidden):

        # write your codes here
        embeds = self.embedding(input)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_size)
        output = self.fc1(lstm_out)
        output = self.relu(output)
        output = self.fc2(output)
        
        return output, hidden
    

    def init_hidden(self, batch_size):

        # write your codes here
        initial_hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE),
                          torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE))
        return initial_hidden
