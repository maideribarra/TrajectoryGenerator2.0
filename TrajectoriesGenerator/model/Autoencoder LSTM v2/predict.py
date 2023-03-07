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

x0 = -1.5
x1 = 1.5
y0 = -1
y1 = 2

arrVariables = ['x','y','vx','vy','angle','vangle','l0','l1','action','n_episodio']

def showOutput(predictions, ruta, x, y):
    out = predictions[0][0][1]
    input = predictions[0][0][0].movedim(0,1)
    for i in range(1,10):
        plt.plot(out[i][:][:,x],out[i][:][:,y])
        plt.plot(input[i][:][:,x],input[i][:][:,y])
        plt.title(arrVariables[x]+arrVariables[y])
        plt.xlim(x0, x1)
        plt.ylim(y0, y1)
        plt.xlabel(arrVariables[x])
        plt.ylabel(arrVariables[y])
        plt.savefig(ruta+str(i)+arrVariables[x]+arrVariables[y]+'outBatchOne.png')
        plt.show()
        


def showAllOutputs(predictions, ruta, x, y):
    for index, batch in enumerate(predictions):
        output=batch[0][1]
        #print('output shape',output.shape)
        for inner, traj in enumerate(output):
            #print('traj shape',traj.shape)
            plt.plot(traj[:][:,x],traj[:][:,y])
            #print(traj[:])
            #outputs.append(traj[:][:].numpy())
            #print(f"total: {index}/{len(predictions)-1} local: {inner}/{len(output)-1}", end="\r")
    plt.xlim(x0, x1)
    plt.ylim(y0, y1)
    plt.xlabel(arrVariables[x])
    plt.ylabel(arrVariables[y])
    plt.title(arrVariables[x]+arrVariables[y])
    plt.savefig(ruta+arrVariables[x]+arrVariables[y]+'allOutput.png')
    plt.show()  
    

def showAllInputs(predictions, ruta, x, y):
    for batch in predictions:
        output=batch[0][0].movedim(0,1)
        for traj in output:
            plt.plot(traj[:][:,x],traj[:][:,y])
            #print(traj[:])
            #outputs.append(traj[:][:].numpy())
    plt.xlim(x0, x1)
    plt.ylim(y0, y1)
    plt.xlabel(arrVariables[x])
    plt.ylabel(arrVariables[y])
    plt.title(arrVariables[x]+arrVariables[y])
    plt.savefig(ruta+arrVariables[x]+arrVariables[y]+'allInput.png')
    plt.show()    
    

def showAllOutputsBatchOne(predictions, ruta, x, y):
    for index, batch in enumerate(predictions):
        output=batch[0][1]
        plt.plot(output[:][:,x],output[:][:,y])
    plt.xlim(x0, x1)
    plt.ylim(y0, y1)
    plt.xlabel(arrVariables[x])
    plt.ylabel(arrVariables[y])
    plt.title(arrVariables[x]+arrVariables[y])
    plt.savefig(ruta+arrVariables[x]+arrVariables[y]+'allOutputBatchOne.png')
    plt.show()  
    

def showAllInputsBatchOne(predictions, ruta, x,y):
    for batch in predictions:
        output=batch[0][0].movedim(0,1)
        #print('output shape',output.shape)
        plt.plot(output[0][:][:,x],output[0][:][:,y])
    plt.xlim(x0, x1)
    plt.ylim(y0, y1)
    plt.xlabel(arrVariables[x])
    plt.ylabel(arrVariables[y])
    plt.title(arrVariables[x]+arrVariables[y])
    plt.savefig(ruta+arrVariables[x]+arrVariables[y]+'allInputBatchOne.png')
    plt.show()    
    

def showOutputBatchOne(predictions, ruta, x,y):
    for i in range(1,10):
        out = predictions[i][0][1]
        input = predictions[i][0][0].movedim(0,1)
        plt.plot(out[:][:,x],out[:][:,y])
        plt.plot(input[0][:][:,x],input[0][:][:,y])
        plt.xlim(x0, x1)
        plt.ylim(y0, y1)
        plt.xlabel(arrVariables[x])
        plt.ylabel(arrVariables[y])
        plt.title(arrVariables[x]+arrVariables[y])
        plt.savefig(ruta+str(i)+arrVariables[x]+arrVariables[y]+'outBatchOne.png')
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
    experimento=sys.argv[1]
    dir='/home/ubuntu/ws_acroba/src/shared/egia/TrajectoriesGenerator/model/Autoencoder LSTM v2/experimentos/'+experimento+'/'+experimento+'.yaml'
    with open(dir, "rb") as f:
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
    logdir = LOG_DIR
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
    #Trayectorias
    showOutputBatchOne(predictions, dir,0,1)
    showAllInputsBatchOne(predictions, dir,0,1)
    showAllOutputsBatchOne(predictions, dir,0,1)

    showOutputBatchOne(predictions, dir,0,2)
    showAllInputsBatchOne(predictions, dir,0,2)
    showAllOutputsBatchOne(predictions, dir,0,2)

    showOutputBatchOne(predictions, dir,1,3)
    showAllInputsBatchOne(predictions, dir,1,3)
    showAllOutputsBatchOne(predictions, dir,1,3)

    showOutputBatchOne(predictions, dir,4,5)
    showAllInputsBatchOne(predictions, dir,4,5)
    showAllOutputsBatchOne(predictions, dir,4,5)

    showOutputBatchOne(predictions, dir,1,6)
    showAllInputsBatchOne(predictions, dir,1,6)
    showAllOutputsBatchOne(predictions, dir,1,6)

    showOutputBatchOne(predictions, dir,1,7)
    showAllInputsBatchOne(predictions, dir,1,7)
    showAllOutputsBatchOne(predictions, dir,1,7)
    showStatisticsOfDimensionBatchOne(predictions)
    