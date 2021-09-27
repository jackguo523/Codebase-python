# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 09:29:43 2021

@author: Jack
"""

# this program reads measured samples from excel file and fits an edge response

import pandas
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt


sheet=pandas.read_excel('ESF-vertical.xlsx',encoding='utf8')
X=sheet['X'].to_numpy()
Y=sheet['mean'].to_numpy()

normY=(Y-Y.min())/(Y.max()-Y.min())

def f(x,a,b,c,d): return a/(np.exp(((x-b)/c))+1)+d

curve=scipy.optimize.curve_fit(f,X,Y)

def fermi(para,x):
    y=para[0]/(np.exp((x-para[1])/para[2])+1)+para[3]
    
    return y

def get_fermi(para,x):
    y=para[0]/(np.exp((x-para[1])/para[2])+1)+para[3]
    
    return y

def inv_fermi(para,y):
    x=np.log(para[0]/(y-para[3])-1)*para[2]+para[1]
    
    return x

nY=fermi(curve[0],X)

fit=(nY-nY.min())/(nY.max()-nY.min()) # normalize based on the fitted curve
data=(Y-nY.min())/(nY.max()-nY.min())

top_cutoff=inv_fermi(curve[0],(nY.max()-nY.min())*0.9+nY.min())
bot_cutoff=inv_fermi(curve[0],(nY.max()-nY.min())*0.1+nY.min())

fd=-np.diff(fit)

result=bot_cutoff-top_cutoff

plt.figure(1)
plt.plot(X,fit,X,data)
plt.title('Edge Response Fitting')
plt.show()

