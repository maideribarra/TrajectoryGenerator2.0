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
from model.v2AutoencoderLSTMv2.encoder import Encoder
from model.v2AutoencoderLSTMv2.Seq2Seq import SeqtoSeq
from data.dataModule import DataModule
import pytorch_lightning as pl
import os 
import yaml
from matplotlib import pyplot as plt 
import pickle
import io

if __name__ == "__main__":
    # First initialize our model.
    #Experimento para probar dataset de 40000, hidden layer=1000
    cwd = os.getcwd()
    dir="/home/ubuntu/ws_acroba/src/shared/egia/TrajectoriesGenerator/experiments/exp23/"
    with open(dir+"exp23.yaml", "rb") as f:
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
        CHK_PATH =datos['CHK_PATH']
        LOSS_FUNCTION =datos['LOSS_FUNCTION']
    print(LEARNING_RATE)
    device =torch.device('cuda')      
    workdir = cwd+'/../../data/ficheros/'
    data = DataModule(workdir + TRAIN_DATASET,
                    workdir + VAL_DATASET,
                    workdir + TEST_DATASET,
                    BATCH_SIZE,
                    NUM_SEQ)
    logdir = cwd +'/../'+ LOG_DIR
    logger = TensorBoardLogger(logdir, name="LSTM")
    num_gpus = 1 if torch.cuda.is_available() else 0
    chk_path = CHK_PATH
    model = SeqtoSeq(NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,DROPOUT_PROB,LEARNING_RATE)  
    model2 = model.load_from_checkpoint(chk_path,NUM_SEQ=NUM_SEQ,INPUT_DIM=INPUT_DIM,OUTPUT_DIM=OUTPUT_DIM,HID_DIM=HID_DIM,N_LAYERS=N_LAYERS,DROPOUT_PROB=DROPOUT_PROB,LEARNING_RATE=LEARNING_RATE, lossFunction=LOSS_FUNCTION)
    pretrained_dict = model2.encoder.state_dict()
    device=torch.device('cuda')
    model3 =Encoder(ENC_DROPOUT, INPUT_DIM,HID_DIM,N_LAYERS,NUM_SEQ,device) 
    model3.load_state_dict(pretrained_dict)    
    predict_loader = data.predict_dataloader
    resultado=model3.eval()
    print(len(resultado))
    #np.savetxt('hiddenVecExp9.txt', nphiddVect, delimiter=',') 
    #np.savetxt('cellVecExp9.txt', nphiddCell, delimiter=',') 
    