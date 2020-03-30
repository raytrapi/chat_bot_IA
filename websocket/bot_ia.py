from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import matplotlib.pyplot as plt
import numpy as np

import os
import shutil
#import pandas
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import mean_squared_error



import tensorflow as tf

from tensorflow.keras import layers
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

caracteres=[' ','\n']
caracteres=[' ', '\n', 'L', 'a', 'R', 'e', 'g', 'n', 't', 'p', 'o', 'r', 'l', 'd', 'A', 's', '«', 'C', 'í', '»', 'i', 'b', 'F', 'é', ',', 'M', '1', '9', '0', '.', 'P', 'ó', 'q', 'u', 'f', 'W', 'j', '_', 'm', 'h', 'v', 'á', 'c', 'y', ';', 'T', 'E', 'z', 'x', 'S', 'ñ', ':', 'ú', '(', ')', 'H', 'N', 'D', 'Y', 'k', 'Q', 'J', 'V', 'I', '¡', '!', 'O', 'G', 'B', '-', 'Á', '¿', "'", '?', 'U', 'É', 'ü', 'w', 'º', '2', '3', '4', 'Í', 'Z', 'ï', 'Ú', 'K', 'X', 'Ó', 'è', '8', '6', '7', 'ö', '+', '5', 'ç', '*', 'à', '/', '"', '%', '@', '$']
np.random.seed(7)
VOCABULARIO=len(caracteres)
print(VOCABULARIO)
LONGITUD=128
FICHERO_PESOS="pruebas/modelos/pesosRNNTexto.hdf5"

def construirModelo(vocabulario=VOCABULARIO, longitud=LONGITUD):
    model = tf.keras.Sequential()
    model.add(layers.LSTM(512, input_shape = (longitud, 1),  return_sequences = True ))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(512))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(vocabulario, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    
    return model
print("Construyendo el modelo")
model = construirModelo(VOCABULARIO)
print("Cargando pesos")
model.load_weights(FICHERO_PESOS)
print("compilando el modelo")
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

#LONGITUD_SALIDA=100
import ipywidgets as widgets
#from IPython.display import clear_output
textoOriginal = []
prediccion = []
def procesar(texto,longitudSalida=50, aleatorio=True, limpiar=True):
   global textoOriginal
   global prediccion
    
   listaTextoCompleto = list(texto)
   if(len(listaTextoCompleto)<LONGITUD):
      listaCaracteresSinProcesar = [' ' for i in range(LONGITUD-len(listaTextoCompleto))]+listaTextoCompleto[:len(listaTextoCompleto)]
   else:
      indice=len(listaTextoCompleto)-LONGITUD
      listaCaracteresSinProcesar = listaTextoCompleto[indice:indice+LONGITUD]
    
   textoProcesado = listaTextoCompleto[:len(textoOriginal)]    
   textoCompleto = texto   
   #palabraSinProcesar.append(' ')
    
   textoAnterior = [caracteres.index(c) for c in textoProcesado]
   ultimoTexto = [caracteres.index(c) for c in listaCaracteresSinProcesar]
    
   #Alimentamos la red con toda la entrada anterior
   #TODO
    
    
   numero=[]
   salida=[]
   probabilidades=[]
   caracteresPosibles=[]
   rng = np.random.default_rng()
   iCaracter=0
   terminar=False
   while(iCaracter<longitudSalida and not terminar):
      iCaracter+=1
      #X=np.array([ultimoTexto])
      X = np.reshape(ultimoTexto, (1, LONGITUD,1))
      caracteresPosibles = model.predict((X/float(VOCABULARIO)))

      prediccionesOrdenadas=(caracteresPosibles.argsort()[0])[::-1]
      candidatos=[]
      for i,pred in enumerate(prediccionesOrdenadas):
         if((caracteresPosibles[0][prediccionesOrdenadas[0]]-caracteresPosibles[0][pred])<0.07):
            candidatos.append(pred)
      candidato=rng.integers(len(candidatos))
      #print(len(candidatos))
      siguienteCaracter=candidatos[(candidato if aleatorio else 0)]
      #probabilidades.append({"letra":caracteres[siguienteCaracter],"probabilidad":caracteresPosibles[0][siguienteCaracter]})
      probabilidades.append(caracteresPosibles[0][siguienteCaracter])
      ultimoTexto.append(siguienteCaracter)
      salida.append(siguienteCaracter)
      ultimoTexto = ultimoTexto[1:]
      numero.append(str(siguienteCaracter))
      if(siguienteCaracter==1):
         terminar=True
   resultado=''.join(caracteres[c] for c in salida)
   return resultado


def responder(mensaje):
   if(mensaje[len(mensaje)-1]!='\n'):
      mensaje+="\n"
   return procesar(mensaje.lower(),longitudSalida=200)