
import torch
import torch.nn as nn
import numpy as np
class VAEencoder(nn.Module):

    def __init__(self, dropout_prob, input_dim,hid_dim,n_layers,num_seq,device):
        super().__init__()
        # loss function
        ##self.criterion = torch.nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(dropout_prob)
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers       
        self.rnn = nn.LSTM(input_size=self.input_dim,hidden_size=self.hid_dim, num_layers=self.n_layers, batch_first=True )
        self.fc21 = nn.Linear(self.hid_dim, self.hid_dim)
        self.fc22 = nn.Linear(self.hid_dim, self.hid_dim)
        self.trg_len =  num_seq
        self.device=torch.device('cuda')
        self.kl=0
        self.N = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))

    def forward(self, batch):  
        encoder_output1, (encoder_hidden1, encoder_cell1) = self.rnn(batch[0])
        mu_hidden = self.fc21(encoder_hidden1)
        #print('mu_hidden.shape',mu_hidden.shape)
        sigma =self.fc22(encoder_hidden1)
        sigma_hidden = torch.exp(sigma).to(self.device)
        #print('sigma_hidden',sigma_hidden.shape)
        normalD=self.N.sample([self.hid_dim,self.hid_dim]).to(self.device)
        #print('normalD shape', normalD.shape)
        multi=torch.matmul(sigma_hidden,normalD).to(self.device)
        #print('multi shape', multi.shape)
        zhidden = mu_hidden + multi
        #arrzhiden = torch.tensor(np.array([zhidden.cpu()])).to(torch.device)
        #print('zhidden shape', zhidden.shape)
        self.kl = (sigma_hidden**2 + mu_hidden**2 - torch.log(sigma_hidden) - 1/2).sum()
        return zhidden, encoder_cell1