# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 11:43:00 2021

@author: Jack
"""


import h5py
import pandas as pd

f=h5py.File('test.h5') # read h5 file

weights={} # initial weights

for layer,group in f.items(): # load layer bias and/or parameters
    weights[layer]=[]

    for p_name in group.keys():
        param=group[p_name]
        
        if len(param)==0:
            weights[layer].extend(None)
        else:
            for k_name in param.keys():
                weights[layer].extend(param[k_name].value[:].flatten().tolist())
                
f.close()

weights_list=[[key]+value for key,value in weights.items()]
weights_df=pd.DataFrame(weights_list)
weights_df.to_csv('vessel.csv',index=False,header=False)