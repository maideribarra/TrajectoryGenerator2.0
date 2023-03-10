import time, random, math, string

import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from decoder import Decoder
from encoder import Encoder
import yaml



class SeqtoSeq(pl.LightningModule):

    def __init__(self,NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,DROPOUT_PROB,LEARNING_RATE):
        super(SeqtoSeq,self).__init__()
        device =torch.device('cuda')
        self.learning_rate= LEARNING_RATE
        self.encoder = Encoder(DROPOUT_PROB, INPUT_DIM,HID_DIM,N_LAYERS,NUM_SEQ,device)
        self.decoder = Decoder(DROPOUT_PROB, INPUT_DIM,HID_DIM,N_LAYERS,OUTPUT_DIM)
        self.trg_len =  NUM_SEQ
        self.n_features= OUTPUT_DIM
        # loss function
        self.criterion = torch.nn.MSELoss()
        self.arrHiddenVec = []
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                          lr=self.learning_rate)
        return optimizer

    def forward(self, batch):
       # src = [sen_len, batch_size]
        # trg = [sen_len, batch_size]
        # teacher_forcing_ratio : the probability to use the teacher forcing.
        batch_size = len(batch)
        #print('forward seq2seq',type(batch))
        # tensor to store decoder outputs
        outputs = torch.zeros(self.trg_len,batch_size,  self.n_features).to(self.device)
        
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(batch)
        res=[hidden, cell]
        self.arrHiddenVec.append(res)
        #print('forward hidden shape',hidden.shape)
        input = torch.zeros( self.trg_len, self.n_features).to(self._device)
        output, hidden, cell = self.decoder(input, hidden, cell) 
        #print('iteración ',self.trg_len)
        #for t in range(1, self.trg_len):
            # insert input, previous hidden and previous cell states 
            # receive output tensor (predictions) and new hidden and cell states.            
           # output, hidden, cell = self.decoder(input, hidden, cell)            
            # replace predictions in a tensor holding predictions for each token  
            #print('output shape',output.shape)          
            #outputs[t] = output           
            # decide if we are going to use teacher forcing or not.
            #teacher_force = random.random() < teacher_forcing_ratio            
            # get the highest predicted token from our predictions.
            #print(output)
            #top1 = output.argmax(1)
            # update input : use ground_truth when teacher_force 
            #input = trg[t] if teacher_force else top1
            #input = top1
            #input = output
        #print('sale forward')
        #print('size input',input.size())
        #print('size outputs',outputs.size())
       # print('size output decoder',output.size())
        return output


    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        #input = batch.movedim(0,1)
        #print('input shape',input.shape)
        #print('calculo loss')
        #print('batch size',batch[0].size())
        #print('output size',output.size())
        loss = self.criterion(output, batch[0])        
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        #print(batch)
        #print(batch_idx)
        output = self.forward(batch)
        #print('calculo loss')
        #print('batch size',batch.size())
        #print('output size',output.size())
        loss = self.criterion(output, batch[0])
        return loss

    def test_step(self, batch, batch_idx):
        #self.arrHiddenVec=[]
        #print('batch size', len(batch))
        #print('shape inicial arr hidden',len(self.arrHiddenVec))
        #input = batch.movedim(0,1)
        input = batch
        #print('input shape',input.shape)
        output = self.forward(batch)
        #print('shape final arr hidden test',len(self.arrHiddenVec))
        loss = self.criterion(output, input)
        self.log("test_loss", loss)
        return self.arrHiddenVec
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        resultados = []
        input = batch.movedim(0,1)
        output = self.forward(batch)
        resultados.append([input,output])
        return resultados


    def estados_ocultos(self):
        #print('shape final arr hidden',len(self.arrHiddenVec[0]))
        return self.arrHiddenVec
