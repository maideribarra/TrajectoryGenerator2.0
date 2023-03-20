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
from data.dataModule import DataModule
import pytorch_lightning as pl
import os 
import yaml
from matplotlib import pyplot as plt 
import pickle
import io
import json
from model.v2AutoencoderLSTMv2.decodeHiddenSate import decodeHiddenState
from sklearn.preprocessing import MinMaxScaler

x0 = -1.5
x1 = 1.5
y0 = -1
y1 = 2

arrVariables = ['x','y','vx','vy','angle','vangle','l0','l1','action1','action2','action3','action4']

def plotAllOutputsWithTime(predictions, ruta, y,random,itBucle):
    for index, batch in enumerate(predictions):
        axisX = list(range(0,len(batch)))
        plt.plot(axisX,batch.detach().numpy()[:,y])
    plt.ylim(y0, y1)
    plt.xlabel('Time')
    plt.ylabel(arrVariables[y])
    plt.title('Outputs Time '+arrVariables[y])
    plt.savefig(ruta+random+'Time'+str(itBucle)+arrVariables[y]+'allOutputBatchOneTimefromhidden.png')
    plt.show()  

def plotAllOutputs(predictions, ruta, x, y,random,itBucle):
    print(ruta)
    for index, batch in enumerate(predictions):        
        plt.plot(batch.detach().numpy()[:,x],batch.detach().numpy()[:,y])
    plt.xlim(x0, x1)
    plt.ylim(y0, y1)
    plt.xlabel(arrVariables[x])
    plt.ylabel(arrVariables[y])
    plt.title('Outputs '+arrVariables[x]+arrVariables[y])
    plt.savefig(ruta+random+str(itBucle)+arrVariables[x]+arrVariables[y]+'allOutputfromhidden.png')
    plt.show()  

def plotOutput(predictions, rutaHidden, y, random,itBucle):
    for i in range(1,10):
        out = predictions[i]
        axisX = list(range(0,len(out)))
        plt.plot(axisX,out.detach().numpy()[:,y], label='output')
        plt.legend(loc="upper left")
        plt.ylim(y0, y1)
        plt.xlabel('Time')
        plt.ylabel(arrVariables[y])
        plt.title('Time '+arrVariables[y])
        plt.savefig( rutaHidden+random+str(itBucle)+'Time'+arrVariables[y]+str(i)+'outBatchOnefromhidden.png')
        plt.show()

def plots(predictions, ruta, random,itBucle):
    plotAllOutputsWithTime(predictions, ruta, 0, random,itBucle)
    plotAllOutputs(predictions, ruta, 0, 1, random,itBucle)
    #plotOutput(predictions, ruta, 0, random,itBucle)
    plotAllOutputsWithTime(predictions, ruta, 1, random,itBucle)
    #plotOutput(predictions, ruta, 1, random,itBucle)
    plotAllOutputsWithTime(predictions, ruta, 2, random,itBucle)
    #plotOutput(predictions, ruta, 2, random,itBucle)
    plotAllOutputsWithTime(predictions, ruta, 3, random,itBucle)
    #plotOutput(predictions, ruta, 3, random,itBucle)
    plotAllOutputsWithTime(predictions, ruta, 4, random,itBucle)
    #plotOutput(predictions, ruta, 4, random,itBucle)
    plotAllOutputsWithTime(predictions, ruta, 5, random,itBucle)
    #plotOutput(predictions, ruta, 5, random,itBucle)
    plotAllOutputsWithTime(predictions, ruta, 6, random,itBucle)
    #plotOutput(predictions, ruta, 6, random,itBucle)
    plotAllOutputsWithTime(predictions, ruta, 7, random,itBucle)
    #plotOutput(predictions, ruta, 7, random,itBucle)
    plotAllOutputsWithTime(predictions, ruta, 8, random,itBucle)
    #plotOutput(predictions, ruta, 8, random,itBucle)

def readHiddenFromFile3d(rutaHidden,rutaCell):
    nphidden, npcell = readHiddenFromFile2dnp(rutaHidden,rutaCell)
    hidd2 = [[x] for x in nphidden] 
    torchHidden=torch.Tensor( hidd2)   
    hiddCell2 = [[x] for x in npcell]
    torchCell =torch.Tensor(hiddCell2)   
    print(torchCell.shape)
    return torchHidden,torchCell

def readHiddenFromFile2dnp(rutaHidden,rutaCell):
    hidden = np.loadtxt(rutaHidden, delimiter=',')  
    cell = np.loadtxt(rutaCell, delimiter=',')  
    return hidden,cell
    

def fromHiddenToTraject(chk_path,NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,DROPOUT_PROB,LEARNING_RATE, rutaHidden, rutaCell):
    model = SeqtoSeq(NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,DROPOUT_PROB,LEARNING_RATE)  
    model2 = model.load_from_checkpoint(chk_path,NUM_SEQ=NUM_SEQ,INPUT_DIM=INPUT_DIM,OUTPUT_DIM=OUTPUT_DIM,HID_DIM=HID_DIM,N_LAYERS=N_LAYERS,DROPOUT_PROB=DROPOUT_PROB,LEARNING_RATE=LEARNING_RATE)
    torchHidden,torchCell = readHiddenFromFile3d(rutaHidden,rutaCell)    
    #leer fichero cell
    model2.set_estados_ocultos(torchHidden)
    model2.set_celdas_ocultas(torchCell)
    predictions = model2.transform_Traj()
    print('len predictions 0',len(predictions))
    print('len predictions 1',len(predictions[0]))
    print('len predictions 2',len(predictions[0][0]))
    return predictions

def fromRandomToTrajectNormalize(dfarr):
    scaler = MinMaxScaler()
    scaler.fit(dfarr)
    dfarrN=scaler.transform(dfarr)

def createRandomArr(arr):
    print('arr',arr.shape)
    print('arr shape 0',arr.shape[1])
    dfarr = pd.DataFrame(arr)
    scaler = MinMaxScaler()
    scaler.fit(dfarr)
    dfarrN=scaler.transform(dfarr)
    randomN = np.random.uniform(low=0, high=1, size=(1,arr.shape[1]))
    random = scaler.inverse_transform(randomN)
    random3d = [[x] for x in random] 
    torcharr=torch.Tensor( random3d) 
    return torcharr

def fromHiddenToTrajectRandom(chk_path,NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,DROPOUT_PROB,LEARNING_RATE, rutaHidden, rutaCell):
    model = SeqtoSeq(NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,DROPOUT_PROB,LEARNING_RATE)  
    npHidden,npCell = readHiddenFromFile2dnp(rutaHidden, rutaCell)    
    torchRHidden = createRandomArr(npHidden)
    torchRCell = createRandomArr(npCell)
    model2 = model.load_from_checkpoint(chk_path,NUM_SEQ=NUM_SEQ,INPUT_DIM=INPUT_DIM,OUTPUT_DIM=OUTPUT_DIM,HID_DIM=HID_DIM,N_LAYERS=N_LAYERS,DROPOUT_PROB=DROPOUT_PROB,LEARNING_RATE=LEARNING_RATE)
    model2.set_estados_ocultos(torchRHidden)
    model2.set_celdas_ocultas(torchRCell)
    predictions = model2.transform_Traj()
    print('len predictions 0',len(predictions))
    print('len predictions 1',len(predictions[0]))
    print('len predictions 2',len(predictions[0][0]))
    return predictions

def bucleRandomTraj(numTraj,chk_path,NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,DROPOUT_PROB,LEARNING_RATE, rutaHidden, rutaCell, dir, fileSer):
    predictions=fromHiddenToTrajectRandom(chk_path,NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,DROPOUT_PROB,LEARNING_RATE,rutaHidden,rutaCell)
    for i in range(0,numTraj):
        plots(predictions, dir,'randomHiddenVecResImages/',i)
    saveTraj2SerializableFile(predictions, dir, fileSer)

def bucleTraj(numTraj,chk_path,NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,DROPOUT_PROB,LEARNING_RATE, rutaHidden, rutaCell, dir, fileSer):
    predictions=fromHiddenToTraject(chk_path,NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,DROPOUT_PROB,LEARNING_RATE,rutaHidden,rutaCell)
    for i in range(0,numTraj):
        plots(predictions, dir,'hiddenVecFiles/Images/',i)
    saveTraj2SerializableFile(predictions, dir, fileSer)

def saveTraj2SerializableFile(arrPredictions, dir, file):
    file = io.BytesIO()
    fileName = dir+file
    serialized = pickle.dump(arrPredictions, file)
    with open(fileName, "wb") as f:
        f.write(file.getbuffer())



if __name__ == "__main__":
    # First initialize our model.
    #Experimento para probar dataset de 40000, hidden layer=1000
    cwd = os.getcwd()
    dir="/home/ubuntu/ws_acroba/src/shared/egia/TrajectoriesGenerator/experiments/exp19/"
    with open(dir+"exp19.yaml", "rb") as f:
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
    bucleRandomTraj(10,chk_path,NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,DROPOUT_PROB,LEARNING_RATE, '/home/ubuntu/ws_acroba/src/shared/egia/TrajectoriesGenerator/0.utilities/hiddenVecExp19.txt','/home/ubuntu/ws_acroba/src/shared/egia/TrajectoriesGenerator/0.utilities/cellVecExp19.txt')
    #predictions=fromHiddenToTraject(chk_path,NUM_SEQ,INPUT_DIM,OUTPUT_DIM,HID_DIM,N_LAYERS,DROPOUT_PROB,LEARNING_RATE, '/home/ubuntu/ws_acroba/src/shared/egia/TrajectoriesGenerator/0.utilities/hiddenVecExp19.txt','/home/ubuntu/ws_acroba/src/shared/egia/TrajectoriesGenerator/0.utilities/cellVecExp19.txt')
    #plots(predictions, dir)
    
    