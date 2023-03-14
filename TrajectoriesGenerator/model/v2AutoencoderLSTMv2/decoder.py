

import torch
import torch.nn as nn

class Decoder(nn.Module):

    def __init__(self, droput_prob, input_dim,hid_dim,n_layers,output_dim):
        super().__init__()
        # loss function
        #self.criterion = torch.nn.CrossEntropyLoss()        
        self.dropout = torch.nn.Dropout(droput_prob)
        self.device =torch.device('cuda')
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers     
        self.output_dim = output_dim 
        self.fc = torch.nn.Linear(self.hid_dim, self.output_dim)             
        self.rnn = nn.LSTM(input_size=self.input_dim,hidden_size=self.hid_dim,  num_layers=self.n_layers, batch_first=True)     

    def forward(self, batch, hidden, cell):
        #cell = torch.zeros(self.n_layers, self.hid_dim).to(self.device)
        output, (hidden, cell) = self.rnn(batch, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell