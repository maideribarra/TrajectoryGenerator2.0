
import torch
import torch.nn as nn
#class Encoder(nn.Module):
class Encoder(nn.Module):

    def __init__(self, dropout_prob, input_dim,hid_dim,n_layers,num_seq,device):
        super().__init__()
        # loss function
        ##self.criterion = torch.nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(dropout_prob)
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers       
        self.rnn = nn.LSTM(input_size=self.input_dim,hidden_size=self.hid_dim, num_layers=self.n_layers, batch_first=True )        
        self.trg_len =  num_seq
        self.device=torch.device('cuda')

    def forward(self, batch):  
        batch_size=len(batch)
        encoder_outputs = torch.zeros(batch_size,self.trg_len, self.hid_dim, device=self.device)
        encoder_output, (encoder_hidden, encoder_cell) = self.rnn(batch[0])
        return encoder_hidden, encoder_cell
    
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        resultados = []
        output = self.forward(batch)
        print(batch[0][:,8])
        resultados.append([batch[0][:,8],output])
        return resultados

