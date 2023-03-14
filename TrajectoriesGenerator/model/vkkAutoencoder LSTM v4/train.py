import time, random, math, string
import datetime
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from pytorch_lightning.loggers import TensorBoardLogger
from Seq2Seq import SeqtoSeq
import sys
sys.path.insert(0,"..")
from dataModuleNormalize import DataModuleNormalize
from dataModule import DataModule
import pytorch_lightning as pl
import os 
import yaml

def train(NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,ENC_DROPOUT,DEC_DROPOUT,DROPOUT_PROB,LEARNING_RATE,BATCH_SIZE,NUM_EPOCHS,TRAIN_DATASET,VAL_DATASET,TEST_DATASET,LOG_DIR):
    device =torch.device('cuda')
    model = SeqtoSeq(NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,DROPOUT_PROB,LEARNING_RATE)  
    workdir = cwd+'/../../data/ficheros/'
    data = DataModule(workdir + TRAIN_DATASET,
                    workdir + VAL_DATASET,
                    workdir + TEST_DATASET,
                    BATCH_SIZE,
                    NUM_SEQ)
    logdir = cwd +'/../'+ LOG_DIR
    print(LOG_DIR)
    logger = TensorBoardLogger(LOG_DIR, name="LSTM")
    num_gpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(max_epochs = NUM_EPOCHS, logger=logger, gpus=num_gpus)
    trainer.fit(model, datamodule=data)
    

def trainWithNormalization(NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,ENC_DROPOUT,DEC_DROPOUT,DROPOUT_PROB,LEARNING_RATE,BATCH_SIZE,NUM_EPOCHS,TRAIN_DATASET,VAL_DATASET,TEST_DATASET,LOG_DIR,LOSS_FUNCTION):
    device =torch.device('cuda')
    model = SeqtoSeq(NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,DROPOUT_PROB,LEARNING_RATE,LOSS_FUNCTION)  
    workdir = cwd+'/../../data/ficheros/'
    data = DataModuleNormalize(workdir + TRAIN_DATASET,
                    workdir + VAL_DATASET,
                    workdir + TEST_DATASET,
                    BATCH_SIZE,
                    NUM_SEQ)
    logdir = cwd +'/../'+ LOG_DIR
    print(LOG_DIR)
    logger = TensorBoardLogger(LOG_DIR, name="LSTM")
    num_gpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(max_epochs = NUM_EPOCHS, logger=logger, gpus=num_gpus)
    trainer.fit(model, datamodule=data)
    

if __name__ == "__main__":
    cwd = os.getcwd()
    experimento=sys.argv[1]
    
    dir="/home/ubuntu/ws_acroba/src/shared/egia/TrajectoriesGenerator/model/Autoencoder LSTM v2/experimentos/"+experimento+'/'
    with open(dir+experimento+".yaml", "rb") as f:
        datos = yaml.load(f, yaml.Loader)
        NUM_SEQ = datos['NUM_SEQ']
        INPUT_DIM = datos['INPUT_DIM']
        OUTPUT_DIM = datos['OUTPUT_DIM']
        HID_DIM = datos['HID_DIM']
        N_LAYERS = datos['N_LAYERS']
        ENC_DROPOUT = datos['ENC_DROPOUT']
        DEC_DROPOUT = datos['DEC_DROPOUT']
        DROPOUT_PROB = datos['DROPOUT_PROB']
        LEARNING_RATE = datos['LEARNING_RATE']
        BATCH_SIZE = datos['BATCH_SIZE']
        NUM_EPOCHS = datos['NUM_EPOCHS']
        TRAIN_DATASET = datos['TRAIN_DATASET']
        VAL_DATASET = datos['VAL_DATASET']
        TEST_DATASET = datos['TEST_DATASET']
        LOG_DIR = datos['LOG_DIR']
        LOSS_FUNCTION =  datos['LOSS_FUNCTION']
    if len(sys.argv)==3:
        arg2=sys.argv[2]
        if arg2=='n':
            print('Dataset normalizado')
            trainWithNormalization(NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,ENC_DROPOUT,DEC_DROPOUT,DROPOUT_PROB,LEARNING_RATE,BATCH_SIZE,NUM_EPOCHS,TRAIN_DATASET,VAL_DATASET,TEST_DATASET,LOG_DIR,LOSS_FUNCTION)
    else:
        train(NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,ENC_DROPOUT,DEC_DROPOUT,DROPOUT_PROB,LEARNING_RATE,BATCH_SIZE,NUM_EPOCHS,TRAIN_DATASET,VAL_DATASET,TEST_DATASET,LOG_DIR)



    
    