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
sys.path.insert(0,"/home/ubuntu/ws_acroba/src/shared/egia/TrajectoriesGenerator/model")
from dataModule import DataModule
import pytorch_lightning as pl
import os 
import yaml
from matplotlib import pyplot as plt 

def showOutput(predictions):
    out = predictions[0][0][1]
    input = predictions[0][0][0].movedim(0,1)
    for i in range(1,10):
        plt.plot(out[i][:][:,0],out[i][:][:,1])
        plt.plot(input[i][:][:,0],input[i][:][:,1])
        plt.show()


def showAllOutputs(predictions):
    for index, batch in enumerate(predictions):
        output=batch[0][1]
        #print('output shape',output.shape)
        for inner, traj in enumerate(output):
            #print('traj shape',traj.shape)
            plt.plot(traj[:][:,0],traj[:][:,1])
            #print(traj[:])
            #outputs.append(traj[:][:].numpy())
            #print(f"total: {index}/{len(predictions)-1} local: {inner}/{len(output)-1}", end="\r")
    plt.show()  

def showAllInputs(predictions):
    for batch in predictions:
        output=batch[0][0].movedim(0,1)
        for traj in output:
            plt.plot(traj[:][:,0],traj[:][:,1])
            #print(traj[:])
            #outputs.append(traj[:][:].numpy())
    plt.show()    

def showAllOutputsBatchOne(predictions):
    for index, batch in enumerate(predictions):
        output=batch[0][1]
        plt.plot(output[:][:,0],output[:][:,1])
    plt.show()  

def showAllInputsBatchOne(predictions):
    for batch in predictions:
        output=batch[0][0].movedim(0,1)
        #print('output shape',output.shape)
        plt.plot(output[0][:][:,0],output[0][:][:,1])
    plt.show()    

def showOutputBatchOne(predictions):
    for i in range(1,10):
        out = predictions[i][0][1]
        input = predictions[i][0][0].movedim(0,1)
        plt.plot(out[:][:,0],out[:][:,1])
        plt.plot(input[0][:][:,0],input[0][:][:,1])
        plt.show()

def showStatisticsOfDimensionBatchOne(predictions):
    res = []
    contador = 0
    for batch in predictions:
        output=batch[0][1]
        for traj in output:
            trajEp = np.append(traj.numpy(),contador)
            res.append(trajEp)
        contador = contador+1
    print('Size dataframe 0',len(res))
    print('Size dataframe 1',len(res[0]))
    print(contador)
    npres = np.asarray(res)
    dfres =pd.DataFrame(npres)
    dfres.columns = ['x','y','vx','vy','angle','vangle','l0','l1','action','n_episodio']
    print(dfres.describe(include='all'))
    print(dfres['y'].where(dfres['l0']>0.6).dropna())
    print(dfres['y'].where(dfres['l0']>0.6).dropna().describe(include='all'))
    print(dfres['l0'].where(dfres['y']<0.01).dropna())
    print(dfres['l0'].where(dfres['y']<0.01).dropna().describe(include='all'))

 
    

if __name__ == "__main__":
    # First initialize our model.
    #Experimento para probar dataset de 40000, hidden layer=1000
    cwd = os.getcwd()
    dir="/home/ubuntu/ws_acroba/src/shared/egia/TrajectoriesGenerator/model/Autoencoder LSTM v2/experimentos/exp9/"
    with open(dir+"exp9.yaml", "rb") as f:
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
    print(LEARNING_RATE)
    device =torch.device('cuda')
    model = SeqtoSeq(NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,DROPOUT_PROB,LEARNING_RATE)       
    workdir = cwd+'/../../data/ficheros/'
    test = "/home/ubuntu/ws_acroba/src/shared/egia/TrajectoriesGenerator/data/ficheros/test40000.dat"
    val = "/home/ubuntu/ws_acroba/src/shared/egia/TrajectoriesGenerator/data/ficheros/val40000.dat"
    train = "/home/ubuntu/ws_acroba/src/shared/egia/TrajectoriesGenerator/data/ficheros/train40000.dat"
    data = DataModule(workdir + TRAIN_DATASET,
                     workdir + VAL_DATASET,
                     workdir + TEST_DATASET,
                     BATCH_SIZE,
                     NUM_SEQ)
   
    logdir = cwd +'/../'+ LOG_DIR
    logger = TensorBoardLogger(logdir, name="LSTM")
    num_gpus = 1 if torch.cuda.is_available() else 0
    chk_path = CHK_PATH
    model2 = model.load_from_checkpoint(chk_path,NUM_SEQ=NUM_SEQ,INPUT_DIM=INPUT_DIM,OUTPUT_DIM=OUTPUT_DIM,HID_DIM=HID_DIM,N_LAYERS=N_LAYERS,DROPOUT_PROB=DROPOUT_PROB,LEARNING_RATE=LEARNING_RATE)
    trainer = pl.Trainer(max_epochs = NUM_EPOCHS, logger=logger, gpus=num_gpus)
    predictions = trainer.predict(model2, datamodule=data)
    #print(predictions)
    #npres=np.asarray(predictions)
    npres=predictions[0][0]
    #print(np.shape(npres[0]))
    print(len(predictions))
    print(len(predictions[0]))
    print(len(predictions[0][0]))
    print(len(predictions[0][0][0]))
    print(len(predictions[0][0][0][0]))
    showOutputBatchOne(predictions)
    showAllInputsBatchOne(predictions)
    showAllOutputsBatchOne(predictions)
    showStatisticsOfDimensionBatchOne(predictions)
    