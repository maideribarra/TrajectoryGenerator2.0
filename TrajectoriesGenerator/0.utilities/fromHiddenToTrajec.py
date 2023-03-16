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
import json
from v2AutoencoderLSTMv2.decodeHiddenSate import decodeHiddenState

x0 = -1.5
x1 = 1.5
y0 = -1
y1 = 2

arrVariables = ['x','y','vx','vy','angle','vangle','l0','l1','action1','action2','action3','action4']

def plotAllOutputsWithTime(predictions, ruta, y):
    for index, batch in enumerate(predictions):
        axisX = list(range(0,len(batch)))
        plt.plot(axisX,batch.detach().numpy()[:,y])
    plt.ylim(y0, y1)
    plt.xlabel('Time')
    plt.ylabel(arrVariables[y])
    plt.title('Outputs Time '+arrVariables[y])
    plt.savefig(ruta+'Time'+arrVariables[y]+'allOutputBatchOneTimefromhidden.png')
    plt.show()  

def plotAllOutputs(predictions, ruta, x, y):
    for index, batch in enumerate(predictions):        
        plt.plot(batch.detach().numpy()[:,x],batch.detach().numpy()[:,y])
    plt.xlim(x0, x1)
    plt.ylim(y0, y1)
    plt.xlabel(arrVariables[x])
    plt.ylabel(arrVariables[y])
    plt.title('Outputs '+arrVariables[x]+arrVariables[y])
    plt.savefig(ruta+arrVariables[x]+arrVariables[y]+'allOutputfromhidden.png')
    plt.show()  

def plotOutput(predictions, ruta, y):
    for i in range(1,10):
        out = predictions[i]
        axisX = list(range(0,len(out)))
        plt.plot(axisX,out.detach().numpy()[:,y], label='output')
        plt.legend(loc="upper left")
        plt.ylim(y0, y1)
        plt.xlabel('Time')
        plt.ylabel(arrVariables[y])
        plt.title('Time '+arrVariables[y])
        plt.savefig(ruta+'Time'+arrVariables[y]+str(i)+'outBatchOnefromhidden.png')
        plt.show()




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
    model = decodeHiddenState(NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,DROPOUT_PROB,LEARNING_RATE,1)       
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
    model2 = model.load_from_checkpoint(chk_path,NUM_SEQ=NUM_SEQ,INPUT_DIM=INPUT_DIM,OUTPUT_DIM=OUTPUT_DIM,HID_DIM=HID_DIM,N_LAYERS=N_LAYERS,DROPOUT_PROB=DROPOUT_PROB,LEARNING_RATE=LEARNING_RATE, batch_size=1)
    #leer fichero hidden
    hidden = np.loadtxt('hiddenVecExp9.txt', delimiter=',')
    hidd2 = [[[x]] for x in hidden] 
    torchHidden=torch.Tensor( hidd2)   
    cell = np.loadtxt('cellVecExp9.txt', delimiter=',')
    hiddCell2 = [[[x]] for x in cell]
    torchCell =torch.Tensor(hiddCell2)   
    print(torchCell.shape)
    #leer fichero cell
    model2.set_estados_ocultos(torchHidden)
    model2.set_celdas_ocultas(torchCell)
    predictions = model2.transform_Traj()
    print('len predictions 0',len(predictions))
    print('len predictions 1',len(predictions[0]))
    print('len predictions 2',len(predictions[0][0]))
    ruta =''
    plotAllOutputsWithTime(predictions, ruta, 0)
    plotAllOutputs(predictions, ruta, 0, 1)
    plotOutput(predictions, ruta, 0)

    plotAllOutputsWithTime(predictions, ruta, 1)
    plotOutput(predictions, ruta, 1)

    plotAllOutputsWithTime(predictions, ruta, 2)
    plotOutput(predictions, ruta, 2)

    plotAllOutputsWithTime(predictions, ruta, 3)
    plotOutput(predictions, ruta, 3)

    plotAllOutputsWithTime(predictions, ruta, 4)
    plotOutput(predictions, ruta, 4)

    plotAllOutputsWithTime(predictions, ruta, 5)
    plotOutput(predictions, ruta, 5)

    plotAllOutputsWithTime(predictions, ruta, 6)
    plotOutput(predictions, ruta, 6)

    plotAllOutputsWithTime(predictions, ruta, 7)
    plotOutput(predictions, ruta, 7)

    plotAllOutputsWithTime(predictions, ruta, 8)
    plotOutput(predictions, ruta, 8)
       
    