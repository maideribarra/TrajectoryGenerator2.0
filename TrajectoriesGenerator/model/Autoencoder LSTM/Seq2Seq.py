import time, random, math, string

import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from decoder import Decoder
from encoder import Encoder


class SeqtoSeq(pl.LightningModule):

    def __init__(self, dropout_prob, input_dim,hid_dim,output_dim,n_layers,num_seq,device, learning_rate):
        super(SeqtoSeq,self).__init__()
        self.learning_rate= learning_rate
        self.encoder = Encoder(dropout_prob, input_dim,hid_dim,n_layers,num_seq,device)
        self.decoder = Decoder(dropout_prob, input_dim,hid_dim,n_layers,output_dim)
        self.trg_len =  num_seq
        self.n_features= output_dim
        # loss function
        self.criterion = torch.nn.MSELoss()
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                          lr=self.learning_rate)
        return optimizer

    def forward(self, batch):
       # src = [sen_len, batch_size]
        # trg = [sen_len, batch_size]
        # teacher_forcing_ratio : the probability to use the teacher forcing.
        batch_size = len(batch)
        print('forward seq2seq',type(batch))
        # tensor to store decoder outputs
        outputs = torch.zeros(self.trg_len,batch_size,  self.n_features).to(self.device)
        
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(batch)
       
        input = batch[:,0]
        print('iteración ',self.trg_len)
        for t in range(1, self.trg_len):
            # insert input, previous hidden and previous cell states 
            # receive output tensor (predictions) and new hidden and cell states.            
            output, hidden, cell = self.decoder(input, hidden, cell)            
            # replace predictions in a tensor holding predictions for each token            
            outputs[t] = output           
            # decide if we are going to use teacher forcing or not.
            #teacher_force = random.random() < teacher_forcing_ratio            
            # get the highest predicted token from our predictions.
            top1 = output.argmax(1)
            # update input : use ground_truth when teacher_force 
            #input = trg[t] if teacher_force else top1
            #input = top1
            input = batch[:,t]
        print('sale forward')
        print('size input',input.size())
        print('size outputs',outputs.size())
        print('size output',output.size())
        return outputs


    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        input = batch.movedim(0,1)
        print('calculo loss')
        print('batch size',batch.size())
        print('output size',output.size())
        loss = self.criterion(output, input)        
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        print(batch)
        print(batch_idx)
        input = batch.movedim(0,1)
        output = self.forward(batch)
        print('calculo loss')
        print('batch size',batch.size())
        print('output size',output.size())
        loss = self.criterion(output, input)
        return loss

    def test_step(self, batch, batch_idx):
        print(batch)
        print(batch_idx)
        input = batch.movedim(0,1)
        output = self.forward(batch)
        print('calculo loss')
        print('batch size',batch.size())
        print('output size',output.size())
        loss = self.criterion(output, input)
        return loss
