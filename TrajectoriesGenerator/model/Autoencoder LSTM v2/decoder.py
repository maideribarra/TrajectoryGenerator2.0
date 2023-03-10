

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
        self.rnn = nn.LSTM(input_size=self.input_dim,hidden_size=self.hid_dim,  num_layers=self.n_layers, dropout=droput_prob,batch_first=True)     

    def forward(self, batch, hidden, cell):
        # input = [batch_size]
        # hidden = [n_layers * n_dir, batch_size, hid_dim]
        # cell = [n_layers * n_dir, batch_size, hid_dim]
        
        #input = batch.unsqueeze(0)
        # input : [1, ,batch_size]
        
        #input = self.dropout(batch)
        # embedded = [1, batch_size, emb_dim]
       # print('decoder batch dim',batch.shape)
        #print('decoder hidden dim',hidden.size())
        #print('decoder cell dim',cell.size())
        cell = torch.zeros(self.n_layers, self.hid_dim).to(self.device)
        output, (hidden, cell) = self.rnn(batch, (hidden, cell))
        # output = [seq_len, batch_size, hid_dim * n_dir]
        # hidden = [n_layers * n_dir, batch_size, hid_dim]
        # cell = [n_layers * n_dir, batch_size, hid_dim]
        
        # seq_len and n_dir will always be 1 in the decoder
        prediction = self.fc(output)
        # prediction = [batch_size, output_dim]
       # print('output shape',output.shape)
        return prediction, hidden, cell