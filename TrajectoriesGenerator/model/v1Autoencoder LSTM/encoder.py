
import torch
import torch.nn as nn
class Encoder(nn.Module):

    def __init__(self, dropout_prob, input_dim,hid_dim,n_layers,num_seq,device):
        super().__init__()
        # loss function
        ##self.criterion = torch.nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(dropout_prob)
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers       
        self.rnn = nn.LSTM(self.input_dim,self.hid_dim, num_layers=self.n_layers,dropout=0.3, batch_first=True )        
        self.trg_len =  num_seq
        self.device=torch.device('cuda')

    def forward(self, batch):
        print('encoder batch',batch.size())    
        encoder_outputs = torch.zeros(self.trg_len, self.hid_dim, device=self.device)
        encoder_hidden = torch.zeros(2, self.hid_dim, device=self.device)
        encoder_cell = torch.zeros(2, self.hid_dim, device=self.device)
        for ei in range(1,self.trg_len):
            encoder_output, (encoder_hidden, encoder_cell) = self.rnn(batch[:,ei,:], (encoder_hidden, encoder_cell))
            encoder_outputs[ei] = encoder_output[0, 0]   
        print('encoder hidden',encoder_hidden.size())
        print('encoder cell',encoder_cell.size())
        print('salgo for encoder')
        return encoder_hidden, encoder_cell