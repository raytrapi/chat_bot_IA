from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import numpy as np

import os
import shutil
import time
import re
import math

import tensorflow as tf

import s2s

from tensorflow.keras import layers
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


vocabulario=[]
with open("../pruebas/modelos/vocabulario_s2s.txt",mode="r",encoding="utf8") as fichero:
   for palabra in fichero:
      vocabulario.append(palabra[:-1])
print("Longitud vocabulario", len(vocabulario))

numeroCapas = 4 #6
d = 128 #512
dff = 512 #2048
numeroCabeceras = 8
tamañoVocabulario = len(vocabulario) 
ratioDespreciar = 0.1

ratioAprendizaje=s2s.ProgramarOptimizacion(d)
optimizador=tf.keras.optimizers.Adam(ratioAprendizaje, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

ratioAprendizajeProgramadoTemporal=s2s.ProgramarOptimizacion(d)

transformador = s2s.Transformer(numeroCapas, d, numeroCabeceras, dff,
                          tamañoVocabulario, tamañoVocabulario, 
                          posicionesEntrada=tamañoVocabulario, posicionesSalida=tamañoVocabulario,
                          ratio=ratioDespreciar)


ficheroPuntoControl="../pruebas/modelos/seq2seq/"
puntoControl=tf.train.Checkpoint(transformer=transformador,  optimizer=optimizador)
manejarPuntoControl=tf.train.CheckpointManager(puntoControl, ficheroPuntoControl, max_to_keep=5)
if manejarPuntoControl.latest_checkpoint:
   puntoControl.restore(manejarPuntoControl.latest_checkpoint)
   print ('Restauramos el último punto de control')


def tokenizar(frase):
   palabras=frase.split(" ")
   resultado=[]
   for palabra in palabras:
      palabra=palabra.lower()
      #if palabra not in vocabulario:
         #print("no esta "+palabra+" en el vocabulario")
      #   vocabulario.append(palabra)
      #resultado.append(vocabulario.index(palabra))
      if palabra in vocabulario:
         resultado.append(vocabulario.index(palabra))
   #resultado.append(0)
   return resultado
def desTokenizar(entrada):
   frase=""
   for e in entrada:
      #print(e)
      frase+=(" " if len(frase)>0 else "")+vocabulario[e]
   return frase

def respuesta(entrada):
   # inp sentence is portuguese, hence adding the start and end token
   entrada = tokenizar(entrada)
   entrada = tf.expand_dims([1]+entrada+[2], 0)

   print(entrada)
   pesosAtencion=None
   salida = [1]
   print(salida)
   salida = tf.expand_dims(salida, 0)
   print(salida)
   for i in range(50):
      mascaraCodificadaEmpaquetada, mascaraCombinada, mascaraDecodificadaEmpaquetada = s2s.crearMascara(entrada,salida)
      prediccion, pesosAtencion = transformador(entrada, salida, False, mascaraCodificadaEmpaquetada, mascaraCombinada, mascaraDecodificadaEmpaquetada)
      prediccion = prediccion[: ,-1:, :]
      prediccionID=tf.cast(tf.argmax(prediccion, axis=-1), tf.int32)
      #print("Pred",prediccionID)
      if tf.equal(prediccionID,2):
         break
      #print(prediccionID)
      salida=tf.concat([salida,prediccionID], axis=-1)


      #if predicted_id == tokenizer_en.vocab_size+1:
      #    return tf.squeeze(output, axis=0), attention_weights

   return tf.squeeze(salida, axis=0), pesosAtencion



def responder(mensaje):
   print(mensaje)
   resultado, pesos = respuesta(mensaje)
   return desTokenizar(resultado[1:]) 

#responder("hola")

#print("Si " if "hola" in vocabulario else "NO")
#print(vocabulario)
