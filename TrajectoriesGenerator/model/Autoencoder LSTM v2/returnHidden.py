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
import yaml
from matplotlib import pyplot as plt 
import pickle
import io

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
    checkpoint = torch.load(chk_path, map_location=lambda storage, loc: storage)
    print(checkpoint)
    trainer = pl.Trainer(max_epochs = NUM_EPOCHS, logger=logger, gpus=num_gpus)
    trainer.test(model2, datamodule=data)
    HiddenCellVect = model2.estados_ocultos()
    print('HiddenCellVect size 0',len(HiddenCellVect))
    print('HiddenCellVect size 1',len(HiddenCellVect[0]))
    print('HiddenCellVect size 2',len(HiddenCellVect[0][0]))
    print('HiddenCellVect size 3',len(HiddenCellVect[0][0][0]))
    print(type(HiddenCellVect[0]))
    print(type(HiddenCellVect[0][0]))
    hiddVect2 = [torch.movedim(x[0],0,1).to(torch.device('cpu')).numpy() for x in HiddenCellVect]
    hiddCell2 = [torch.movedim(x[1],0,1).to(torch.device('cpu')).numpy() for x in HiddenCellVect]
    print('HiddenVec size 0',len(hiddVect2))
    print('HiddenVec size 1',len(hiddVect2[0]))
    print('HiddenVec size 2',len(hiddVect2[0][0]))
    print('HiddenVec size 3',len(hiddVect2[0][0][0]))
    print(type(hiddVect2[0]))
    hiddVect = [np.mean(x, axis=1, dtype=torch.dtype) for x in hiddVect2]
    #print(hiddVect.shape)    
    print('hiddVect size 0',len(hiddVect))
    print('hiddVect size 1',len(hiddVect[0]))
    print('hiddVect size 2',len(hiddVect[0][0]))
    #nphiddVect =np.asarray(hiddVect)
    #print('nphiddVect.shape',nphiddVect.shape) #(num_batches,hid_dim)
    #print(nphiddVect)
    for batch in hiddVect:
        for traj in batch:
            #print(traj.shape)
            #vecDib=traj[::10]
            #plt.plot(vecDib)
            plt.plot(traj)
    plt.show()        
    file = io.BytesIO()
    fileName = dir+'\resultado'
    serialized = pickle.dump(HiddenVect, file)
    with open(fileName, "wb") as f:
        f.write(file.getbuffer())
       
    