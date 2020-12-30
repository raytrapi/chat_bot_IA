import silabar
def insertarPalabra(palabra, palabras, posiciones, minuscula, palabrasInvalidas=["\n"]):
    if len(palabra)==0 or palabra in palabrasInvalidas:
        return
    if minuscula:
        palabra=palabra.lower()
    if(palabra not in palabras):
        palabras.append(palabra)
    indice=palabras.index(palabra)
    posiciones.append(indice)
#Vocabulario obtiene tanto la lista de palabras como las posiciones por párrafo de cada palabra en el texto dado
#juntarParrafos tiene en cuenta que un \n que no está precedido por un "." o un ". " no se considera como parrafo nuevo
def vocabulario1(linea, caracterSeparacion=' ', caracteresPuntuacion=['!','"','#','$','%','&','(',')','*','+',',','-','.','/',':',';','<','=','>','?','@','[','\\','\'',']','^','_','`','{','|','}','~','\t','\n','\r','¿','¡'], caracteresDescartados=['\r', '\t'], minuscula=True, palabras=['\n'], conParrafos=False, juntarParrafos=False, caracteresFinalesParrafo=['.','_']):
    posiciones=[]
    parrafo=[]
    esPuntuacion=False
    palabra=""
    ultimoCaracter=""
    palabrasInvalidas=caracteresDescartados
    for c in linea:
        if(c==caracterSeparacion):
            #if(esPuntuacion):
            #    palabra+=c
            insertarPalabra(palabra, palabras, parrafo, minuscula,palabrasInvalidas)
            palabra=""
            esPuntuacion=False
        else:
            if(c not in caracteresPuntuacion):
                if(esPuntuacion):
                    insertarPalabra(palabra, palabras, parrafo,minuscula,palabrasInvalidas)
                    esPuntuacion=False
                    palabra=c
                else:
                    palabra+=c
            else:
                insertarPalabra(palabra, palabras, parrafo,minuscula,palabrasInvalidas)
                palabra=""
                if c not in caracteresDescartados:
                    palabra=c
                    esPuntuacion=True
        if c!=caracterSeparacion and c not in caracteresDescartados and c!='\n':
            ultimoCaracter=c
    
    if conParrafos and len(parrafo)>0:
        if(not juntarParrafos or ultimoCaracter in caracteresFinalesParrafo):
            posiciones.append(parrafo)
            parrafo=[]
    #TODO Implementar un mecanismo para evitar palabras cortadas al final de línea con el -
    if palabra!="":
        insertarPalabra(palabra, palabras, parrafo,minuscula,palabrasInvalidas)
        palabra=""

    if(len(parrafo)>0):
        posiciones.append(parrafo)
        parrafo=[]
    return palabras,posiciones

def vocabulario(frase, voc=[None,"_^_","_$_"],sinFinLinea=False):
   palabras=frase.split()
   posiciones=[]
   for palabra in palabras:
      vocTemp=silabar.obtenerSilabas(palabra,sinFinLinea)
      for v in vocTemp:
         if v not in voc:
            voc.append(v)
         posiciones.append(voc.index(v))
      
   return voc,[posiciones]