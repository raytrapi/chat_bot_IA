from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf

def cogerAngulo(pos, i, d):
  ratiosAngulo = 1 / np.power(10000, (2 * (i//2)) / np.float32(d))
  return pos * ratiosAngulo

def codiciacionPosicional(posicion, d):
  angulos = cogerAngulo(np.arange(posicion)[:, np.newaxis], np.arange(d)[np.newaxis, :], d)
  angulos[:, 0::2] = np.sin(angulos[:, 0::2]) #Lo hacemos para los pares
  angulos[:, 1::2] = np.cos(angulos[:, 1::2]) #Lo hacemos para los impares
  posCodificados = angulos[np.newaxis, ...]
  return tf.cast(posCodificados, dtype=tf.float32)



def crearMascaraEmpaquetada(secuencia):
   secuencia=tf.cast(tf.math.equal(secuencia,0), tf.float32)
   return secuencia[:, tf.newaxis, tf.newaxis, :]

def crearMascaraAlFrente(n): #Retorna una máscara diagonal de 0's en el triangulo inferior y de tamaño nxn
   mascara=1-tf.linalg.band_part(tf.ones((n,n)),-1,0)
   return mascara

def productoEscalarConAtencion(q,k,v, mascaraAtencion):
   matrizQK=tf.matmul(q,k,transpose_b=True)
   #Calculamos la profuncidad d
   dk=tf.cast(tf.shape(k)[-1],tf.float32)
   #Valor escalar de atención
   atencionEscalar=matrizQK/tf.math.sqrt(dk)

   #Sumamos la máscara de atención
   if(mascaraAtencion is not None):
      atencionEscalar+=(mascaraAtencion*-1e9) #el -1e9 es para generar un valor infinito.
   #Calculamos la función softmax y normalizamos al últmo eje
   pesosAtencion=tf.nn.softmax(atencionEscalar, axis=-1)
   salida=tf.matmul(pesosAtencion, v)
   return salida, pesosAtencion

class AtencionMultiple(tf.keras.layers.Layer):
   def __init__(self, d, numeroCabezas):
      super(AtencionMultiple, self).__init__()
      self.numeroCabezas = numeroCabezas
      self.d = d
      
      assert d % self.numeroCabezas == 0
      
      self.profundida = d // self.numeroCabezas #División entera
      
      self.wq = tf.keras.layers.Dense(d)
      self.wk = tf.keras.layers.Dense(d)
      self.wv = tf.keras.layers.Dense(d)
      
      self.capaDensa = tf.keras.layers.Dense(d)

   #  Divide la última dimensio dentro (numeroCabezas, profundidad) transponiendo el resultado
   # que tiene dimensioines (tamañoPaque, numeroCabezas, longitudSecuencia, profundidad)
   def dividirCabezas(self, x, tamañoPaquete):
      x = tf.reshape(x, (tamañoPaquete, -1, self.numeroCabezas, self.profundida))
      return tf.transpose(x, perm=[0, 2, 1, 3])
      
   def call(self, v, k, q, mascara):
      tamañoPaquete = tf.shape(q)[0]
      
      q = self.wq(q)
      k = self.wk(k) 
      v = self.wv(v)
      
      q = self.dividirCabezas(q, tamañoPaquete)
      k = self.dividirCabezas(k, tamañoPaquete)
      v = self.dividirCabezas(v, tamañoPaquete)
      
      resultadoAtencion, pesosAtencion = productoEscalarConAtencion(q, k, v, mascara)
      resultadoAtencion = tf.transpose(resultadoAtencion, perm=[0, 2, 1, 3])
      concatenarAtenciones = tf.reshape(resultadoAtencion, (tamañoPaquete, -1, self.d))
      resultado = self.capaDensa(concatenarAtenciones)  
      return resultado, pesosAtencion

def redNeuronalHaciaDelante(d, dff):
   return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d)
   ])

class CapaCodificadora(tf.keras.layers.Layer):
   def __init__(self, d, numeroCabezas, dff, ratio=0.1):
      super(CapaCodificadora, self).__init__()

      self.mCA = AtencionMultiple(d, numeroCabezas)
      self.redFF = redNeuronalHaciaDelante(d, dff)

      self.capaNormal1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
      self.capaNormal2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
      
      self.despreciar1 = tf.keras.layers.Dropout(ratio) 
      self.despreciar2 = tf.keras.layers.Dropout(ratio)
    
   def call(self, x, entrenamiento, mascara):

      salidaAtencion, _ = self.mCA(x, x, x, mascara)
      salidaAtencion = self.despreciar1(salidaAtencion, training=entrenamiento)
      salida1 = self.capaNormal1(x + salidaAtencion)  
      
      salidaFFN = self.redFF(salida1)
      salidaFFN = self.despreciar2(salidaFFN, training=entrenamiento)
      salida2 = self.capaNormal2(salida1 + salidaFFN)  
      
      return salida2

class CapaDecodificadora(tf.keras.layers.Layer):
   def __init__(self, d, numeroCabeceras, dff, ratio=0.1):
      super(CapaDecodificadora, self).__init__()

      self.mCA1=AtencionMultiple(d, numeroCabeceras)
      self.mCA2=AtencionMultiple(d, numeroCabeceras)

      self.redFF=redNeuronalHaciaDelante(d,dff)

      self.capaNormal1=tf.keras.layers.LayerNormalization(epsilon=1e-6)
      self.capaNormal2=tf.keras.layers.LayerNormalization(epsilon=1e-6)
      self.capaNormal3=tf.keras.layers.LayerNormalization(epsilon=1e-6)

      self.despreciar1=tf.keras.layers.Dropout(ratio)
      self.despreciar2=tf.keras.layers.Dropout(ratio)
      self.despreciar3=tf.keras.layers.Dropout(ratio)

   def call(self,x,salidaCodificada, entrenamiento, mascaraAlFrente, mascaraEmpaquetada):
      atencion1, bloquePesosAtencion1 = self.mCA1(x,x,x,mascaraAlFrente)
      atencion1=self.despreciar1(atencion1,training=entrenamiento)
      salida1=self.capaNormal1(atencion1+x)
      #print(salida1)
      atencion2, bloquePesosAtencion2 = self.mCA2(salidaCodificada,salidaCodificada,salida1,mascaraEmpaquetada)
      atencion2=self.despreciar2(atencion2,training=entrenamiento)
      salida2=self.capaNormal2(atencion2+salida1)

      salidaRedFF=self.redFF(salida2)
      salidaRedFF=self.despreciar3(salidaRedFF, training=entrenamiento)
      salida3=self.capaNormal3(salidaRedFF+salida2)

      return salida3, bloquePesosAtencion1, bloquePesosAtencion2

class Codificador(tf.keras.layers.Layer):
   def __init__(self, numeroCapas, d, numeroCabeceras, dff, tamañoVocabulario, poisicionMaximaCodificacion, ratio=0.1):
      super(Codificador, self).__init__()

      self.d=d
      self.numeroCapas=numeroCapas

      self.embedding=tf.keras.layers.Embedding(tamañoVocabulario, d)
      self.posicionesCodificadas=codiciacionPosicional(poisicionMaximaCodificacion, self.d)

      self.capasCodificacion=[CapaCodificadora(d, numeroCabeceras,dff, ratio) for _ in range(self.numeroCapas)]

      self.despreciar=tf.keras.layers.Dropout(ratio)

   def call(self, x, entrenamiento, mascara):

      longitudSecuencia=tf.shape(x)[1]

      x=self.embedding(x)
      x*=tf.math.sqrt(tf.cast(self.d,tf.float32))
      x+=self.posicionesCodificadas[:, :longitudSecuencia, :]

      x=self.despreciar(x, training=entrenamiento)

      for i in range(self.numeroCapas):
         x=self.capasCodificacion[i](x,entrenamiento,mascara)

      return x

class Decodificador(tf.keras.layers.Layer):
   def __init__(self, numeroCapas, d, numeroCabeceras, dff, tamañoVocabulario, posicionMaximaCodificacion, ratio=0.1):
      super(Decodificador, self).__init__()

      self.d=d
      self.numeroCapas=numeroCapas

      self.embedding = tf.keras.layers.Embedding(tamañoVocabulario, d)
      self.posicionesCodificadas=codiciacionPosicional(posicionMaximaCodificacion, d)

      self.capasDecodificacion=[CapaDecodificadora(d, numeroCabeceras,dff, ratio) for _ in range(self.numeroCapas)]

      self.despreciar=tf.keras.layers.Dropout(ratio)

   def call(self, x, salidaCodificada, entrenamiento, mascaraAlFrente, mascaraEmpaquetada):

      longitudSecuencia=tf.shape(x)[1]
      pesosAtencion={}

      x=self.embedding(x)
      x*=tf.math.sqrt(tf.cast(self.d,tf.float32))
      x+=self.posicionesCodificadas[:, :longitudSecuencia, :]

      x=self.despreciar(x, training=entrenamiento)

      for i in range(self.numeroCapas):
         x, bloque1, bloque2 =self.capasDecodificacion[i](x,salidaCodificada,entrenamiento,mascaraAlFrente, mascaraEmpaquetada)

         pesosAtencion['capaDecodificadora{}_bloque1'.format(i+1)]=bloque1
         pesosAtencion['capaDecodificadora{}_bloque2'.format(i+1)]=bloque2

      return x, pesosAtencion

class Transformer(tf.keras.Model):
   def __init__(self, numeroCapas, d, numeroCabeceras, dff, tamañoVocabularioEntrada, 
               tamañoVocabularioSalida, posicionesEntrada, posicionesSalida, ratio=0.1):
      super(Transformer, self).__init__()

      self.codificador = Codificador(numeroCapas, d, numeroCabeceras, dff, 
                              tamañoVocabularioEntrada, posicionesEntrada, ratio)

      self.decodificador = Decodificador(numeroCapas, d, numeroCabeceras, dff, 
                              tamañoVocabularioSalida, posicionesSalida, ratio)

      self.capaFinal = tf.keras.layers.Dense(tamañoVocabularioSalida)
    
   def call(self, entrada, objetivo, entrenamiento, mascaraCodificadaEmpaquetada, 
           mascaraAlFrente, mascaraDecodificadaEmpaquetada):

      salidaCodificada = self.codificador(entrada, entrenamiento, mascaraCodificadaEmpaquetada)  
      
   
      salidaDecodificada, pesosAtencion = self.decodificador(
         objetivo, salidaCodificada, entrenamiento, mascaraAlFrente, mascaraDecodificadaEmpaquetada)
      
      salida = self.capaFinal(salidaDecodificada) 
      
      return salida, pesosAtencion

def crearMascara(entrada, salida):
  mascaraCodificadaEmpaquetada = crearMascaraEmpaquetada(entrada)
  mascaraDecodificadaEmpaquetada = crearMascaraEmpaquetada(entrada)
  mascaraAlFrente = crearMascaraAlFrente(tf.shape(salida)[1])
  mascaraDecodificadaEmpaqutadaObjetivo = crearMascaraEmpaquetada(salida)
  
  mascarasCombinadas = tf.maximum(mascaraDecodificadaEmpaqutadaObjetivo, mascaraAlFrente)
  return mascaraCodificadaEmpaquetada, mascarasCombinadas, mascaraDecodificadaEmpaquetada

class ProgramarOptimizacion(tf.keras.optimizers.schedules.LearningRateSchedule):
   def __init__(self, d, pasosWarmup=4000):
      super(ProgramarOptimizacion, self).__init__()
      
      self.d = d
      self.d = tf.cast(self.d, tf.float32)

      self.pasosWarmup = pasosWarmup
    
   def __call__(self, paso):
      argumento1 = tf.math.rsqrt(paso)
      argumento2 = paso * (self.pasosWarmup ** -1.5)

      return tf.math.rsqrt(self.d) * tf.math.minimum(argumento1, argumento2)
