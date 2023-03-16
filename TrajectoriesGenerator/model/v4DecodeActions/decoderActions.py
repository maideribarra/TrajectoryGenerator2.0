

import torch
import torch.nn as nn

class DecoderActions(nn.Module):

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
    
    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = torch.sqrt(self.criterion(output, batch[0]))   
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = torch.sqrt(self.criterion(output, batch[0]))
        return loss

    def test_step(self, batch, batch_idx):
        input = batch
        output = self.forward(batch)
        loss = torch.sqrt(self.criterion(output, input))
        self.log("test_loss", loss)
        return self.arrHiddenVec
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        resultados = []
        input = batch.movedim(0,1)
        output = self.forward(batch)
        resultados.append([input,output])
        return resultados

