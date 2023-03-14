import time, random, math, string

import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from model.v3VAELSTM.VAEdecoder import VAEdecoder
from model.v3VAELSTM.VAEencoder import VAEencoder
import yaml



class VAESeqtoSeq(pl.LightningModule):

    def __init__(self,NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,DROPOUT_PROB,LEARNING_RATE):
        super(VAESeqtoSeq,self).__init__()
        device =torch.device('cuda')
        self.learning_rate= LEARNING_RATE
        self.encoder = VAEencoder(DROPOUT_PROB, INPUT_DIM,HID_DIM,N_LAYERS,NUM_SEQ,device)
        self.decoder = VAEdecoder(DROPOUT_PROB, INPUT_DIM,HID_DIM,N_LAYERS,OUTPUT_DIM)
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
        batch_size = len(batch)
        outputs = torch.zeros(self.trg_len,batch_size,  self.n_features).to(self.device)        
        hidden, cell = self.encoder(batch)
        print(hidden.shape)
        res=[hidden, cell]
        self.arrHiddenVec.append(res)
        input = torch.zeros( self.trg_len, self.n_features).to(self._device)
        output, hidden, cell = self.decoder(input, hidden, cell) 
        return output


    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = torch.sqrt(self.criterion(output, batch[0]))  
        lossT = loss + self.encoder.kl 
        self.log("train_loss", lossT)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = torch.sqrt(self.criterion(output, batch[0]))
        lossT = loss + self.encoder.kl 
        return lossT

    def test_step(self, batch, batch_idx):
        input = batch
        output = self.forward(batch)
        loss = torch.sqrt(self.criterion(output, input))
        lossT = loss + self.encoder.kl 
        self.log("test_loss", lossT)
        return self.arrHiddenVec
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        resultados = []
        input = batch.movedim(0,1)
        output = self.forward(batch)
        resultados.append([input,output])
        return resultados


    def estados_ocultos(self):
        return self.arrHiddenVec
