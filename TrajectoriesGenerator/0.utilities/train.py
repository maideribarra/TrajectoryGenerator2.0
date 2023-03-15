import time, random, math, string
import datetime
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from pytorch_lightning.loggers import TensorBoardLogger
import sys
sys.path.insert(0,"..")
from model.v2AutoencoderLSTMv2.Seq2Seq import SeqtoSeq
from model.v3VAELSTM.VAESeq2Seq import VAESeqtoSeq
from data.dataModuleNormalize import DataModuleNormalize
from data.dataModule import DataModule
import pytorch_lightning as pl
import os 
import yaml

def train(NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,ENC_DROPOUT,DEC_DROPOUT,DROPOUT_PROB,LEARNING_RATE,BATCH_SIZE,NUM_EPOCHS,TRAIN_DATASET,VAL_DATASET,TEST_DATASET,LOG_DIR,LOSS_FUNCTION):
    device =torch.device('cuda')
    model = SeqtoSeq(NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,DROPOUT_PROB,LEARNING_RATE)  
    workdir = cwd+'/../data/ficheros/'
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
    model = SeqtoSeq(NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,DROPOUT_PROB,LEARNING_RATE)  
    workdir = cwd+'/../data/ficheros/'
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

def trainVAE(NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,ENC_DROPOUT,DEC_DROPOUT,DROPOUT_PROB,LEARNING_RATE,BATCH_SIZE,NUM_EPOCHS,TRAIN_DATASET,VAL_DATASET,TEST_DATASET,LOG_DIR,LOSS_FUNCTION):
    device =torch.device('cuda')
    model = VAESeqtoSeq(NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,DROPOUT_PROB,LEARNING_RATE)  
    workdir = cwd+'/../data/ficheros/'
    data = DataModule(workdir + TRAIN_DATASET,
                    workdir + VAL_DATASET,
                    workdir + TEST_DATASET,
                    BATCH_SIZE,
                    NUM_SEQ)
    print(LOG_DIR)
    logger = TensorBoardLogger(LOG_DIR, name="LSTM")
    num_gpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(max_epochs = NUM_EPOCHS, logger=logger, gpus=num_gpus)
    trainer.fit(model, datamodule=data)

def trainfromPretrainnedVAE(NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,ENC_DROPOUT,DEC_DROPOUT,DROPOUT_PROB,LEARNING_RATE,BATCH_SIZE,NUM_EPOCHS,TRAIN_DATASET,VAL_DATASET,TEST_DATASET,LOG_DIR,LOSS_FUNCTION,CHK_PATH):
    device =torch.device('cuda')
    model = SeqtoSeq(NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,DROPOUT_PROB,LEARNING_RATE)  
    workdir = cwd+'/../data/ficheros/'
    data = DataModule(workdir + TRAIN_DATASET,
                    workdir + VAL_DATASET,
                    workdir + TEST_DATASET,
                    BATCH_SIZE,
                    NUM_SEQ)
    print(LOG_DIR)
    model2 = model.load_from_checkpoint(CHK_PATH,NUM_SEQ=NUM_SEQ,INPUT_DIM=INPUT_DIM,OUTPUT_DIM=OUTPUT_DIM,HID_DIM=HID_DIM,N_LAYERS=N_LAYERS,DROPOUT_PROB=DROPOUT_PROB,LEARNING_RATE=LEARNING_RATE)
    pretrained_dict = model2.encoder.rnn.state_dict()
    model3 = VAESeqtoSeq(NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,DROPOUT_PROB,LEARNING_RATE)  
    model3.encoder.rnn.load_state_dict(pretrained_dict)
    logger = TensorBoardLogger(LOG_DIR, name="LSTM")
    num_gpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(max_epochs = NUM_EPOCHS, logger=logger, gpus=num_gpus)
    trainer.fit(model3, datamodule=data)

def trainfromPretrainnedVAE2(NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,ENC_DROPOUT,DEC_DROPOUT,DROPOUT_PROB,LEARNING_RATE,BATCH_SIZE,NUM_EPOCHS,TRAIN_DATASET,VAL_DATASET,TEST_DATASET,LOG_DIR,LOSS_FUNCTION,CHK_PATH):
    device =torch.device('cuda')
    model = VAESeqtoSeq(NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,DROPOUT_PROB,LEARNING_RATE)   
    workdir = cwd+'/../data/ficheros/'
    data = DataModule(workdir + TRAIN_DATASET,
                    workdir + VAL_DATASET,
                    workdir + TEST_DATASET,
                    BATCH_SIZE,
                    NUM_SEQ)
    print(LOG_DIR)
    model2 = model.load_from_checkpoint(CHK_PATH,NUM_SEQ=NUM_SEQ,INPUT_DIM=INPUT_DIM,OUTPUT_DIM=OUTPUT_DIM,HID_DIM=HID_DIM,N_LAYERS=N_LAYERS,DROPOUT_PROB=DROPOUT_PROB,LEARNING_RATE=LEARNING_RATE)
    workdir = cwd+'/../data/ficheros/'
    data = DataModule(workdir + TRAIN_DATASET,
                    workdir + VAL_DATASET,
                    workdir + TEST_DATASET,
                    BATCH_SIZE,
                    NUM_SEQ)
    print(LOG_DIR)
    logger = TensorBoardLogger(LOG_DIR, name="LSTM")
    num_gpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(max_epochs = NUM_EPOCHS, logger=logger, gpus=num_gpus)
    trainer.fit(model2, datamodule=data)

if __name__ == "__main__":
    cwd = os.getcwd()
    experimento=sys.argv[1]
    print(sys.argv)
    dir="/home/ubuntu/ws_acroba/src/shared/egia/TrajectoriesGenerator/experiments/"+experimento+'/'
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
        MODEL = datos['MODEL']
        CHK_PATH = datos['CHK_PATH']
    print(MODEL)
    if len(sys.argv)==3:
        arg2=sys.argv[2]
        if arg2=='n':
            print('Dataset normalizado')
            trainWithNormalization(NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,ENC_DROPOUT,DEC_DROPOUT,DROPOUT_PROB,LEARNING_RATE,BATCH_SIZE,NUM_EPOCHS,TRAIN_DATASET,VAL_DATASET,TEST_DATASET,LOG_DIR,LOSS_FUNCTION)
        if arg2=='p':
            print('train from pretrainned')
            trainfromPretrainnedVAE2(NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,ENC_DROPOUT,DEC_DROPOUT,DROPOUT_PROB,LEARNING_RATE,BATCH_SIZE,NUM_EPOCHS,TRAIN_DATASET,VAL_DATASET,TEST_DATASET,LOG_DIR,LOSS_FUNCTION,CHK_PATH)

    if MODEL=='VAE':
        print('trainVAE')
        trainVAE(NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,ENC_DROPOUT,DEC_DROPOUT,DROPOUT_PROB,LEARNING_RATE,BATCH_SIZE,NUM_EPOCHS,TRAIN_DATASET,VAL_DATASET,TEST_DATASET,LOG_DIR,LOSS_FUNCTION)

    else:
        train(NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,ENC_DROPOUT,DEC_DROPOUT,DROPOUT_PROB,LEARNING_RATE,BATCH_SIZE,NUM_EPOCHS,TRAIN_DATASET,VAL_DATASET,TEST_DATASET,LOG_DIR,LOSS_FUNCTION)



    
    