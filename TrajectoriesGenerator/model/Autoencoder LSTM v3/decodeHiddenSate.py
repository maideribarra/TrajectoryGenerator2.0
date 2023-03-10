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



class decodeHiddenState(pl.LightningModule):

    def __init__(self,NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,DROPOUT_PROB,LEARNING_RATE, batch_size):
        super(decodeHiddenState,self).__init__()
        device =torch.device('cuda')
        self.learning_rate= LEARNING_RATE
        self.encoder = Encoder(DROPOUT_PROB, INPUT_DIM,HID_DIM,N_LAYERS,NUM_SEQ,device)
        self.decoder = Decoder(DROPOUT_PROB, INPUT_DIM,HID_DIM,N_LAYERS,OUTPUT_DIM)
        self.trg_len =  NUM_SEQ
        self.n_features= OUTPUT_DIM
        self.batch_size = batch_size
        # loss function
        self.criterion = torch.nn.MSELoss()
        self.arrHiddenVec = []
        self.arrCellVec = []
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                          lr=self.learning_rate)
        return optimizer

    def forward(self, hidden, cell):       
        input = torch.zeros(self.batch_size, self.trg_len, self.n_features).to(self._device)
        output, hidden, cell = self.decoder(input, hidden, cell)         
        return output


    def training_step(self, hidden, cell):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pass
    
    def transform_Traj(self):
        resultados = []
        for i in range(len(self.arrHiddenVec)):
            output = self.forward(self.arrHiddenVec[i],self.arrCellVec[i])
            resultados.append(output)
        return resultados

    def set_estados_ocultos(self,hid_vect):
        self.arrHiddenVec = hid_vect

    def set_celdas_ocultas(self,cell_vect):
        self.arrCellVec = cell_vect
