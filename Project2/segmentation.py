# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 15:40:10 2021

@author: Sílvia Mourão
FC57541
"""
#%%
# =============================================================================
# # PDI - Projeto 2
# =============================================================================

#Importacao de modulos
from skimage.morphology import rectangle, erosion, dilation, \
    binary_erosion, binary_dilation, binary_opening, binary_closing, \
        skeletonize
from skimage.segmentation import watershed
from imageio import imread
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


#Definicao de funcoes que irao ser utilizadas
def reconstrucao_dual(mask,marker):
    a=1
    ee=rectangle(3,3)
    while a!=0:
        E=erosion(marker,ee)
        R=np.maximum(mask.astype(float),E.astype(float))
        a=np.count_nonzero(marker!=R)
        marker=deepcopy(R)
    return R

def reconstrucao_bin (mask,marker):
    se= rectangle(3,3)
    ant=np.zeros(mask.shape)
    R=np.copy(marker)
    while np.array_equal(R,ant) == False:
        ant = R
        R=np.logical_and(binary_dilation(R,se),mask)
    return R


plt.close('all')

#%%
# =============================================================================
# # Exercício 1.1
# =============================================================================

LSat0 = imread('lsat01.tif')

# Separação da imagem Lsat01 nas três bandas R G B
LSat0_R=LSat0[:,:,0]
LSat0_G=LSat0[:,:,1]
LSat0_B=LSat0[:,:,2]

# Imagem de cinzentos a partir da média das três bandas
LSat=(LSat0_R/3 + LSat0_G + LSat0_B/3).astype(int)


# Ver o histograma da imagem resultante
g1,r1 = np.histogram(LSat, bins=256, range=(0,256))


# Por análise do histograma vemos que a zona de maior intensidade se encontra 
# antes do valor de cinzento 29
lim=29

#Threshold da imagem
LSat_bin=np.zeros(LSat.shape)
LSat_bin[LSat>lim] = 0
LSat_bin[LSat<lim] = 1

# A imagem resultante tem ainda alguns pixels que não pertencem ao rio. Para 
# eliminar estes pixels fazemos uma operação de abertura e depois de fecho

# Com um retângulo (4,4) conseguimos o melhor resultado nos dois pontos mais 
# difíceis (ponte no canto sup direito da imagem e curva no canto inf direito)

ee=rectangle(4,4)
LSat_f1=binary_opening(LSat_bin,ee)
LSat_f=binary_closing(LSat_f1,ee)


# =============================================================================
# # Exercício 1.2
# =============================================================================


# Medição da distância aos pixels do rio
d= ndimage.distance_transform_edt(LSat_f==1)

# Reconstrução dual da distância
d1=d+1
rd_d= reconstrucao_dual(d,d1)

# Mínimo regional
min_reg = rd_d - d

# Watershed para cálculo da linha média do rio
marker,n= ndimage.label(min_reg)
W = watershed(LSat_f, marker, mask = np.ones(LSat_f.shape))
Lint = W- erosion(W, ee)

# Imagem binária da linha do rio
Lint_bin=Lint.astype(bool)

# Skeletonize da linha do rio para menor espessura
sk=skeletonize(Lint_bin)

# A imagem sk é binaria com valor de 1 e 0
# Multiplicar a imagem sk para esta aparecer a branco sobre a imagem final
sk=sk*255

# Imagem final com a linha média sobreposta
LSat_med =LSat + sk

#%%
# =============================================================================
# # Exercício 2
# =============================================================================

# Solução com base nas bandas da imagem
# Separação da imagem Ik02 nas três bandas R G B
Ik0=imread("ik02.tif")
img2_r=Ik0[:,:,0]
img2_g=Ik0[:,:,1]
img2_b=Ik0[:,:,2]

# Média das 3 bandas
Ik=np.uint(img2_r/3 + img2_g/3 + img2_b/3).astype(int)

# Aplicação de um filtro gaussiano para suavizar a imagem
Ik=ndimage.filters.gaussian_filter(Ik, 1)

# Ver o histograma da imagem resultante
g2,r2 = np.histogram(Ik, bins=256, range=(0,256))

# Threshold por histerese
# Escolha dos limites da imagem da média para isolar os pixels fortes dos que
# não pertencem às estradas
t2 = 230
t1 = 180
X = Ik <= t1
Y = Ik >= t2

# Procura de pixéis fracos que pertencem ainda às estradas
M = np.logical_not(np.logical_or(X,Y))
Z = reconstrucao_bin(M, np.logical_and(binary_dilation(Y, rectangle(3, 3)), M))
TH3 = np.logical_or(Y, Z)

# Realização de uma operação de fecho com um elemento estruturante largo para
# fechar algumas estradas que tem partes mais escuras na imagem original
f=binary_closing(TH3,rectangle(15,15))

# Por multiplicação da imagem pelo resultado do fecho nas três bandas podemos 
# obter uma primeira imagem
estradas_final0 = np.copy(Ik0)
for i in range(Ik0.shape[2]):
    estradas_final0[:,:,i] = estradas_final0[:,:,i] * f

# Esta imagem contem ainda muitas zonas que não pertencem às estradas.
# Por um processo semelhante ao do exercício 1 podemos encontrar as bacias de
# watershed da imagem e usar as fronteiras das bacias como marcador para
# reconstruir a imagem que tinha resultado do primeiro passo de threshold

d2= ndimage.distance_transform_edt(f==1)
d1_2=d2+1
rd_d2= reconstrucao_dual(d2,d1_2)
min_reg2 = rd_d2 - d2
ee2=rectangle(3,3)
marker2,n2= ndimage.label(min_reg2)
W2 = watershed(f, marker2, mask = np.ones(f.shape))
Iint = W2- erosion(W2, ee2)
Iint_bin=Iint.astype(bool) #Imagem binária das fronteiras
a=reconstrucao_bin(TH3,Iint_bin)

# Multiplicação da imagem do passo anterior pela imagem original para obtenção 
# de um resultado final 24 bits
estradas_final = np.copy(Ik0)
for i in range(Ik0.shape[2]):
    estradas_final[:,:,i] = estradas_final[:,:,i] * a
  
# =============================================================================
# # Solução alternativa com base na diferença entre bandas da imagem
# =============================================================================

# Seguindo um método semelhante ao do exercício 3
dif_ik=img2_r.astype(float)-(img2_b.astype(float))

# Definimos os valores negativos da imagem como tendo um valor de 255
dif_ik[dif_ik<0]=255
Ik1=dif_ik
Ik1=ndimage.filters.gaussian_filter(Ik1, 1)

# Threshold por histerese
# Neste caso os pixels que pertencem às estradas têm valores próximos de 0
# pois têm um valor de cinzento semelhante nas duas bandas
# Processo semelhante ao da primeira solução
t2_2 = 50
t2_1 = 70
X2 = Ik1 >= t2_1
Y2 = Ik1 <= t2_2

M2 = np.logical_not(np.logical_or(X2,Y2))
Z2 = reconstrucao_bin(M2, np.logical_and(binary_dilation(Y2, rectangle(3, 3)), M2))
TH2 = np.logical_or(Y2, Z2)
fecho2 = binary_closing(TH2,rectangle(15,15))

d3= ndimage.distance_transform_edt(fecho2==1)
d1_3=d3+1
rd_d3= reconstrucao_dual(d3,d1_3)
min_reg3 = rd_d3 - d3
marker2,n= ndimage.label(min_reg3)
W3 = watershed(fecho2, marker2, mask = np.ones(f.shape))
Iint2 = W3- erosion(W3, ee2)
Iint_bin2=Iint2.astype(bool)

a2=reconstrucao_bin(TH2,Iint_bin2)

# Multiplicar a imagem resultante pela imagem na banda do vermelho onde existe
# maior contraste entre as zonas que ainda são visíveis na imagem, mas que não
# fazem parte das estradas
Ik2=np.copy(Ik0[:,:,0])*a2

# Realização de um novo threshold por histerese
# Processo semelhante ao anterior
t2a = 230
t1a = 200
Xa = Ik2 <= t1a
Ya = Ik2 >= t2a

Ma = np.logical_not(np.logical_or(Xa,Ya))
Za = reconstrucao_bin(Ma, np.logical_and(binary_dilation(Ya, rectangle(3, 3)), Ma))
TH3a = np.logical_or(Ya, Za)
fechoa = binary_closing(TH3a,rectangle(15,15))

da= ndimage.distance_transform_edt(fechoa==1)
d1a=da+1
rd_da= reconstrucao_dual(da,d1a)
min_reg = rd_da - da
markera,n= ndimage.label(min_reg)
Wa = watershed(fechoa, markera, mask = np.ones(f.shape))
Iinta = Wa- erosion(Wa, ee2)
Iint_bina=Iinta.astype(bool)

a1=reconstrucao_bin(TH3a,Iint_bina)

# Multiplicação da imagem do passo anterior pela imagem original para obtenção 
# de um resultado final 24 bits
estradas_finala = np.copy(Ik0)
for i in range(Ik0.shape[2]):
    estradas_finala[:,:,i] = estradas_finala[:,:,i] * a1

#%%
# =============================================================================
# # Exercício 3
# =============================================================================

# Diferença de valores entre as bandas R e B de forma a fazer sobressair telhados
dif=img2_r.astype(float)-(img2_b.astype(float))

# Histograma
uu, r = np.histogram(dif, bins = 256, range=(0,256))

# Cálculo das retas para medição da distância ao histograma
b1=uu[0]
x1=np.where(uu==np.max(uu))[0][0]
y1=uu[x1]

x2=255
y2=uu[x2]

m1=((y2-y1)/(x2-x1))
m2=-(1/m1)

# Determinamos a distância através da intersecção entre as duas retas
# y1=m1x+b1
# y2=m2x+b2
# Igualando as duas temos que
# y=m2*((b2-b1)/(m1-m2))+b2

b2=[]
x4=[]
y4=[]
dist=[]

for t in range (255):
    b2.append(uu[t]-m2*t)
    x4.append((b2[t]-b1)/(m1-m2))
    y4.append(m2*((b2[t]-b1)/(m1-m2))+b2[t])
    dist.append(np.sqrt((x4[t]-t)**2+(y4[t])-uu[t]))
    
th=np.where(dist==np.max(dist))[0][0]

# Estabelecer o valor de th como threshold para a imagem final
roof = dif > th

# Resultado não suavizado
telhado1 = np.copy(Ik0)
for i in range(Ik0.shape[2]):
    telhado1[:,:,i] = telhado1[:,:,i] * roof

# Resultado suavizado
telhado2= np.copy(Ik0)
for i in range(Ik0.shape[2]):
    telhado2[:,:,i] = telhado2[:,:,i] * roof
    telhado2[:,:,i] = erosion(telhado2[:,:,i], rectangle(3,3))
    telhado2[:,:,i] = dilation(telhado2[:,:,i], rectangle(3,3))

#%%

# =============================================================================
# #Plots
# =============================================================================

fig1=plt.figure()
fig1.suptitle('River')
plt.subplot(141); plt.imshow(LSat, 'gray'); plt.title('Original',fontsize=8); plt.axis('off')
plt.subplot(142); plt.imshow(LSat_bin, 'gray'); plt.title('Binary Image',fontsize=8); plt.axis('off')
plt.subplot(143); plt.imshow(LSat_f, 'gray'); plt.title('Opening+Closing',fontsize=8); plt.axis('off')
plt.subplot(144); plt.imshow(LSat_med, 'gray'); plt.title('Average River Line',fontsize=8); plt.axis('off')

fig2=plt.figure()
fig2.suptitle('Roads')
plt.subplot(221); plt.imshow(Ik, 'gray'); plt.title('Original'); plt.axis('off')
plt.subplot(222); plt.imshow(estradas_final0, 'gray'); plt.title('Initial Result'); plt.axis('off')
plt.subplot(223); plt.imshow(estradas_final,'gray'); plt.title('Final Result'); plt.axis('off')
plt.subplot(224); plt.imshow(estradas_finala, 'gray'); plt.title('Alternative Solution'); plt.axis('off')

fig3=plt.figure()
fig3.suptitle('Rooftops')
plt.subplot(221);plt.imshow(Ik0,'gray');plt.title('Original'); plt.axis('off')
plt.subplot(222);plt.imshow(telhado1,'gray');plt.title('Rooftops'); plt.axis('off')
plt.subplot(223);plt.imshow(telhado2,'gray');plt.title('Rooftops Smooth'); plt.axis('off')
