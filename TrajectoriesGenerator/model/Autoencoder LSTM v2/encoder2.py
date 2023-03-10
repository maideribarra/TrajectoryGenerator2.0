
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
        self.rnn = nn.LSTM(input_size=self.input_dim,hidden_size=self.hid_dim, num_layers=self.n_layers,dropout=0.3, batch_first=True )        
        self.trg_len =  num_seq
        self.device=torch.device('cuda')

    def forward(self, batch):
        print('encoder batch',batch.size())  
        print(len(batch))  
        batch_size=len(batch)
        encoder_outputs = torch.zeros(batch_size,self.trg_len, self.hid_dim, device=self.device)
        encoder_hidden = torch.zeros(self.n_layers, batch_size, self.hid_dim, device=self.device)
        print('encoder hidden batch',encoder_hidden.size()) 
        
        if batch_size>1:
            encoder_cell = torch.zeros(self.n_layers,batch_size, self.hid_dim, device=self.device)
            encoder_hidden = torch.zeros(self.n_layers, batch_size, self.hid_dim, device=self.device)
        else:
            encoder_cell = torch.zeros(self.n_layers, self.hid_dim, device=self.device)
            encoder_hidden = torch.zeros(self.n_layers, self.hid_dim, device=self.device)

        #encoder_output, (encoder_hidden, encoder_cell) = self.rnn(batch, (encoder_hidden, encoder_cell))
        for ei in range(1,self.trg_len):
            encoder_output, (encoder_hidden, encoder_cell) = self.rnn(batch[:,ei,:], (encoder_hidden, encoder_cell))
            #encoder_outputs[ei] = encoder_output[0, 0]   
        #print('encoder hidden',encoder_hidden.size())
        #print('encoder cell',encoder_cell.size())
        #print('salgo for encoder')
        return encoder_hidden, encoder_cell