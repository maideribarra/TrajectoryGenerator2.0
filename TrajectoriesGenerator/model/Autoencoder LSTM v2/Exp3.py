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
from dataModule import DataModule
import pytorch_lightning as pl
import os 

if __name__ == "__main__":
    # First initialize our model.
    #Experimento para probar dataset de 40000, hidden layer=1000
    cwd = os.getcwd()
    NUM_SEQ = 260
    INPUT_DIM = 9
    OUTPUT_DIM = 9
    HID_DIM = 1000   
    N_LAYERS = 2
    ENC_DROPOUT = 0.3
    DEC_DROPOUT = 0.3
    DROPOUT_PROB = 0.3
    LEARNING_RATE = 2e-5
    BATCH_SIZE = 16
    NUM_EPOCHS = 6
    device =torch.device('cuda')
    model = SeqtoSeq(DROPOUT_PROB, INPUT_DIM,HID_DIM,OUTPUT_DIM,N_LAYERS,NUM_SEQ,device, LEARNING_RATE)
    N_EPOCHS = 5
    CLIP = 1
    
       
    workdir = cwd+'/../../data/ficheros/'

    data = DataModule(workdir + 'train40000.dat',
                    workdir + 'val40000.dat',
                    workdir + 'test40000.dat',
                    BATCH_SIZE,260)
    logdir = cwd +'/../'+"TrajectoryGenerator_logs_exp3/" 
    logger = TensorBoardLogger(logdir, name="LSTM")
    num_gpus = 1 if torch.cuda.is_available() else 0
    #trainer = pl.Trainer(max_epochs = NUM_EPOCHS, logger=logger, gpus=num_gpus)
    #trainer.fit(model, datamodule=data)
    #test_out = trainer.test(model, datamodule=data)
    #print(test_out)
    chk_path = '/home/ubuntu/ws_acroba/src/epoch=2-step=5892.ckpt'
    print(chk_path)
    model2 = model.load_from_checkpoint(chk_path)
    trainer = pl.Trainer(max_epochs = NUM_EPOCHS, logger=logger, gpus=num_gpus)
    #trainer.fit(model, datamodule=data)
    #test_out = trainer.test(model2, datamodule=data)
    predictions = trainer.predict(model2, datamodule=data)

    from matplotlib import pyplot as plt 
    #plt.plot(predictions, predictions)
    npres=np.asarray(predictions)
    print('array',npres.shape)
    contador=1
    numEjemplo=55
    tsinput=npres[numEjemplo][0][0]
    tsoutput=npres[numEjemplo][0][1]
    print('tsinput',tsinput.shape)
    bachSize=tsinput.shape[0]
    numTraj=tsinput.shape[1]
    plt.plot(tsinput[1][:][:,0],tsinput[1][:][:,1])
    plt.plot(tsoutput[1][:][:,0],tsoutput[1][:][:,1])
    print('input',tsinput[0][:][:,:])
    print('output',tsoutput[0][:][:,:])
    plt.tight_layout(pad=1.0)
    