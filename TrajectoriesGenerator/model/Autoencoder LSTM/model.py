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
from dataModule import DataModule
import pytorch_lightning as pl

if __name__ == "__main__":
    # First initialize our model.
    NUM_SEQ = 137
    INPUT_DIM = 9
    OUTPUT_DIM = 9
    HID_DIM = 55    
    N_LAYERS = 2
    ENC_DROPOUT = 0.3
    DEC_DROPOUT = 0.3
    DROPOUT_PROB = 0.3
    LEARNING_RATE = 2e-5
    BATCH_SIZE = 4
    NUM_EPOCHS = 6
    device =torch.device('cuda')
    model = SeqtoSeq(DROPOUT_PROB, INPUT_DIM,HID_DIM,OUTPUT_DIM,N_LAYERS,NUM_SEQ,device, LEARNING_RATE)
    N_EPOCHS = 10
    CLIP = 1
    
   
    workdir = '../data/'

    data = DataModule(workdir + 'train.dat',
                    workdir + 'val.dat',
                    workdir + 'test.dat',
                    BATCH_SIZE)
    logdir = "bert_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logger = TensorBoardLogger(logdir, name="LSTM")
    num_gpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(max_epochs = NUM_EPOCHS, logger=logger, gpus=num_gpus)
    trainer.fit(model, datamodule=data)
    test_out = trainer.test(model, datamodule=data)
    print(test_out)