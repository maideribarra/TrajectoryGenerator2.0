import time, random, math, string

import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from model.v2AutoencoderLSTMv2.decoder import Decoder
from model.v2AutoencoderLSTMv2.encoder import Encoder
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
        self.arrCellVec = []
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                          lr=self.learning_rate)
        return optimizer

    def forward(self, batch):
        batch_size = len(batch)
        outputs = torch.zeros(self.trg_len,batch_size,  self.n_features).to(self.device)        
        hidden, cell = self.encoder(batch)
        print(batch[0][:,8])
        res=[hidden, cell]
        self.arrHiddenVec.append(res)
         
        return res


    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = torch.sqrt(self.criterion(output, batch[0]))   
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return NotImplemented

    def test_step(self, batch, batch_idx):
        input = batch
        output = self.forward(batch)
        loss = torch.sqrt(self.criterion(output, input))
        self.log("test_loss", loss)
        return self.arrHiddenVec
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        resultados = []
        output = self.forward(batch)
        resultados.append([batch[0][:,8],output])
        return resultados


    def estados_ocultos(self):
        return self.arrHiddenVec
    

    def transform_Traj(self):
        return NotImplemented
