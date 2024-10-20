# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 16:23:36 2021

@author: Sílvia Mourão
FC57541
"""
from imageio import imread, imwrite
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import ndimage
plt.close('all');

#Leitura da Imagem
Img=imread('noisyland.tif')
dim= Img.shape
lin=dim[0]
col=dim[1]

#Filtragem Espacial

#Correção das Linhas
Filtro_1= np.ones((7,30)) / (7*30) #filtro
PB1=ndimage.median_filter(Img, footprint=Filtro_1, mode='constant', cval=0.0)
PA1= Img.astype(float) - PB1.astype(float)

Filtro_2=np.ones((1,30)) / (1*30) #filtro 2
PB2=ndimage.median_filter(Img, footprint=Filtro_2, mode='constant', cval=0.0)
PA2= Img.astype(float) - PB2.astype(float)
Imagem_corrigida_linhas= PB1.astype(float) + PA2

#Colunas das colunas (tendo já sido corrigidas as linhas)
Filtro_3= np.ones((30,6)) / (30*6) #filtro
PB3=ndimage.median_filter(Imagem_corrigida_linhas, footprint=Filtro_3, mode='constant', cval=0.0)
PA3= Imagem_corrigida_linhas.astype(float) - PB3.astype(float)

Filtro_4=np.ones((30,1)) / (1*30) #filtro 2
PB4=ndimage.median_filter(Imagem_corrigida_linhas, footprint=Filtro_4, mode='constant', cval=0.0)
PA4= Imagem_corrigida_linhas.astype(float) - PB4.astype(float)

Imagem_filtrada= PB3.astype(float) + PA4

# #Output
plt.figure(figsize=(15,10));
plt.suptitle('Spatial Filter',fontsize=20)
plt.subplot(332);plt.imshow(Img,'gray',vmin=0,vmax=255); plt.title('Original'); plt.axis('off')
plt.subplot(334);plt.imshow(PB1,'gray',vmin=0,vmax=255); plt.title('LP1'); plt.axis('off')
plt.subplot(335);plt.imshow(PA2,'gray', vmin=0,vmax=255); plt.title('HP1'); plt.axis('off')
plt.subplot(336);plt.imshow(Imagem_corrigida_linhas,'gray', vmin=0,vmax=255) ;plt.title('Horizontal Noise Correction'); plt.axis('off')
plt.subplot(337);plt.imshow(PB3,'gray', vmin=0,vmax=255); plt.title('LP3'); plt.axis('off')
plt.subplot(338);plt.imshow(PA4,'gray', vmin=0,vmax=255); plt.title('HP4'); plt.axis('off')
plt.subplot(339);plt.imshow(Imagem_filtrada,'gray', vmin=0,vmax=255); plt.title('Filtered Image'); plt.axis('off')




#Fourier

ddft=np.fft.fft2(Img)
espectro_c_log=np.log10(np.abs(np.fft.fftshift(ddft)))
mask1=espectro_c_log>=3
mask2=np.zeros((lin,col))
q=5
p=10
mask2[89-q:89+q,311-q:311+q]=1
mask2[189-q:189+q,211-q:211+q]=1
mask2[289-q:289+q,311-q:311+q]=1
mask2[189-q:189+q,411-q:411+q]=1

mask2[88-p:88+p,410-p:410+p]=1
mask2[88-p:88+p,210-p:210+p]=1
mask2[288-p:288+p,210-p:210+p]=1
mask2[288-p:288+p,410-p:410+p]=1
mask=np.logical_not(np.logical_and(mask1,mask2))


Final= np.real(np.fft.ifft2(np.fft.fftshift(mask)*ddft))


# #Output
fig2=plt.figure(figsize=(15,10))
plt.suptitle('Fourier Filter',fontsize=20)
plt.subplot(221);plt.imshow(Img, 'gray'); plt.title('Original'); plt.axis('off')
plt.subplot(222); plt.imshow(espectro_c_log, 'gray'); plt.title('Centered Spectrum')
plt.subplot(223); plt.imshow(mask, 'gray'); plt.title('Mask')
plt.subplot(224); plt.imshow(Final, 'gray', vmin=0,vmax=255); plt.title('Filtered Image')



