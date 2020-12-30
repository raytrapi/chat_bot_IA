tiposCaracter=[
    ['0','1','2','3','4','5','6','7','8','9'],
    ['i','u'],
    ['a','e','o'],
    ['á','é','í','ó','ú'],
    ['b','c','d','f','g','h','j','k','l','m','n','ñ','p','q','r','s','t','v','w','x','y','z']
]
def obtenerTipoCaracter(c):
    for idx,tipo in enumerate(tiposCaracter):
        if c in tipo:
            return idx
    return len(tiposCaracter)
consonantesNoSeparables={
    "b":"r", 
    "c":"r", 
    "d":"r", 
    "g":"r", 
    "f":"r", 
    "k":"r", 
    "t":"r", 
    "b":"l", 
    "c":"l", 
    "g":"l", 
    "f":"l", 
    "k":"l", 
    "p":"l",

    "l":"l"
    
}
longitudTiposCaracter=len(tiposCaracter)
prefijos=["intro", "intra","micro", "multi", "sobre", "super", "abs","anti", "auto", "post", "hemi", "hipo", "meta", "mega", "omni", "tele", "pre", "pos", "sin","sub", "an", "bi", "co", "in", "im", "re"]
sufijos=["arquía", "filia", "isimo", "cidio", "logía", "ultra", "audi", "ismo", "cida", "des", "dis", "ito", "ita", "azo","de", "di"]
def obtenerSilabas(palabra,sinFinLinea=False):
   silabas=[]
   sinVocal=True
   silaba=""
   tipoCaracter=-1
   #tipoUltimoCaracter=-1
   #caracterAnterior=""
   #caracteresGuardados=""
   palabraMinuscula=palabra.lower()
   separar=False
   desde=0
   hasta=len(palabra)
   su_=[]
   #quitamos los signos de puntuación
   for i in range(len(palabraMinuscula)):
      if obtenerTipoCaracter(palabraMinuscula[i])<longitudTiposCaracter:
         break
      silabas.append(palabra[i])
      desde+=1
   for i in range(len(palabraMinuscula)):
      if obtenerTipoCaracter(palabraMinuscula[len(palabraMinuscula)-1-i])<longitudTiposCaracter:
         break
      su_.insert(0,palabra[len(palabraMinuscula)-1-i])
      hasta-=1
   palabraMinuscula=palabraMinuscula[desde:hasta]
   palabra=palabra[desde:hasta]
   desde=0
   hasta=len(palabra)
   #Primero buscamos los prefijos y los sufijos
   
   iAnterior=-1
   iSiguiente=1
   for p in prefijos:
      if p==palabraMinuscula[0:len(p)]:
         silabas.append(palabra[0:len(p)])
         desde=len(p)
         break
   
   for s in sufijos:
      if s==palabraMinuscula[len(palabra)-len(s):]:
         su_.insert(0,palabra[len(palabra)-len(s):])
         hasta=len(palabra)-len(s)
         break
   #Casos especiales
   #if desde==0 and len(palabraMinuscula)>3 and "amb" ==palabraMinuscula[0:3]:
   #   desde=2
   #   silabas.append(palabra[0:2])
   palabraMinuscula=palabraMinuscula[desde:hasta]
   palabra=palabra[desde:hasta]
   ultimoIdxPalabra=len(palabra)-1 #Pongo 2 para evitar sumar más abajo en la 
   for i in range(len(palabraMinuscula)):
      c=palabra[i]
      cMinuscula=palabraMinuscula[i]
      
      cAnterior=palabraMinuscula[iAnterior] if iAnterior>=0 else ""
      cSiguiente=palabraMinuscula[iSiguiente] if iSiguiente<=ultimoIdxPalabra else ""
      tipoCaracter=obtenerTipoCaracter(cMinuscula)
      tipoCaracterIz=obtenerTipoCaracter(cAnterior)
      tipoCaracterDe=obtenerTipoCaracter(cSiguiente)
      
      noSeparable=(cAnterior in consonantesNoSeparables and consonantesNoSeparables[cAnterior]==cMinuscula)
      separar=(cMinuscula in consonantesNoSeparables and consonantesNoSeparables[cMinuscula]==cSiguiente) or tipoCaracter>4 or tipoCaracterIz>4
      separar=separar or (cMinuscula == 't' and cAnterior=='s')
      separar=separar or (cMinuscula == 'b' and cAnterior=='m')
      separar=separar or (tipoCaracter==0 or tipoCaracter==len(tiposCaracter))
      junto=tipoCaracter==4 and (tipoCaracterIz in [1,2,3] and tipoCaracterDe not in [1,2,3]) and len(silaba)>1
      junto=junto or (tipoCaracterIz in [1,2] and tipoCaracter in [1,2,3])or (tipoCaracterIz in [1,2,3] and tipoCaracter in [1,2])
      #junto=junto or (tipoCaracter == "m" and cSiguiente=="b")
      if silaba=="" or ((sinVocal or noSeparable or junto) and not separar):
         if tipoCaracter in [1,2,3]:
               sinVocal=False
         silaba+=c
      else:
         if len(silaba)==1 and obtenerTipoCaracter(silaba)==4 and len(silabas)>0 and obtenerTipoCaracter(silabas[-1]) not in [0, len(tiposCaracter)]:
            silabas[len(silabas)-1]+=silaba
         else:
            silabas.append(silaba)
         if (tipoCaracter==0 or tipoCaracter==len(tiposCaracter)):
            silabas.append(c)
            c=""
         
         silaba=c
         #caracteresGuardados=""
         sinVocal=tipoCaracter not in [1,2,3]
      if not noSeparable:
         iAnterior=i
      #iAnterior=i
      iSiguiente=i+2
      #tipoUltimoCaracter=tipoCaracter
      #caracterAnterior=c
   if tipoCaracter==4 and len(silabas)>0 and len(silaba)==1 and obtenerTipoCaracter(silabas[-1]) not in [0, len(tiposCaracter)]:
      silabas[len(silabas)-1]+=silaba
   else:
      silabas.append(silaba)
   silabas=silabas+su_
   if sinFinLinea:
      silabas.append(" ")
   else:
      silabas[len(silabas)-1]+="_"
   return silabas

'''
palabrasPrueba=[
    "aprender", "Abscisas", "usurpar","mesa", "silla", "comer","azul", "camisa","época",
    "cofre", "ladrón", "plato", "francés","subrayar", "sublunar", "súbliminar","postromántico",
    "cuota","ambiguo","cuidar","ruido","bueno","cuerpo","guardar","paraguas","ciudades",
    "triunfo","violencia","violín","tiembla","siembra","caiga","vaina","gourmet","Lourdes",
    "oigo","heroico","Eustaquio","reunión","reinar","aceite","autopista","caudal",
    "baile","aire","Qué","ahuecar"
]

print(obtenerSilabas("¿aprender?"))
print(obtenerSilabas("estás?"))
for palabra in palabrasPrueba:
    print(palabra)
    print(obtenerSilabas(palabra))

'''
