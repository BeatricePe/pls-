# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 17:13:58 2018

@author: Bea
"""

''' COMPARE SF AND AGGRESSION '''


import pandas as pd
from numpy import array  
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn import cross_decomposition
import matplotlib.pyplot as plt


''' Import file of day1 and day2; drop Frames and Beh columns '''
def Reading (file_in, sepfile):
    DataFrame = pd.read_csv (file_in, sep = sepfile)
    DataFrame = DataFrame.drop ('Frames',1)
    beh = DataFrame.loc [:,'Beh']
    DataFrame = DataFrame.drop ('Beh',1)
    neuronsList = []
    for el in DataFrame:
        neuronsList.append(el)
    DataFrame = StandardScaler().fit_transform(DataFrame)
    DataFrame = pd.DataFrame(DataFrame,columns=neuronsList)
    DataFrame ['Beh'] = beh
    return DataFrame

''' Creation of DataFrame for pls '''
def CreationDataFrame (FileList, n, strList):
    df = pd.DataFrame ()
    df2 = pd.DataFrame()
    listColumn = ['state']
    for i in range (n):
        df = df.append(FileList[i]) 
        y = pd.DataFrame(strList[i], index=np.arange(len(FileList[i])), columns=listColumn)
        df2 = df2.append(y)
    return df, df2

''' PLS+Graphing '''
def Pls (df, df2, string):
    pls2 = PLSRegression(n_components=2)
    (xs,ys) = pls2.fit_transform(df,df2)
    t = df2.values
    principalDf = pd.DataFrame(data = xs
             , columns = ['pls 1', 'pls 2'])
    pls = cross_decomposition.PLSRegression(n_components = 10)
    pls.fit(df, df2) 
    variance = np.var(pls.x_scores_, axis = 0) 
    principalDf [string] = t
    return principalDf, variance 

''' Graph '''
def Graphing (pDf, Title, StrList, StrList2, colorList, string): 
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 10)
    ax.set_ylabel('Principal Component 2', fontsize = 10)
    ax.set_title(Title, fontsize = 15)
    targets = StrList
    colors = colorList
    for target, color in zip(targets,colors):
        indicesToKeep = pDf[string] == target
        ax.scatter(pDf.loc[indicesToKeep, 'pls 1']
               , pDf.loc[indicesToKeep, 'pls 2']
               , c = color
               , s = 50
               ,alpha = 0.8)
    ax.legend(StrList2)
    ax.grid()
    return fig

        
Data1 = Reading (r"C:\Users\Bea\Desktop\second day\Vmh5SF21beh.csv",';')      
Data2 = Reading (r"C:\Users\Bea\Desktop\aggression\Vmh5a21beh.csv",';')
Data3 = Reading (r"C:\Users\Bea\Desktop\first day\Vmh5SF10beh_bea.csv",';') #Baseline 

Data2= Data2.drop(Data2.index[2184:])
Data1= Data1.drop(Data1.index[2370:])
#pre-elaboration d1
listIndexD = []
for i in range (len (Data1)):
    if Data1['Beh'][i] != 'face the consp' and Data1['Beh'][i] != 'defense action' and  Data1['Beh'][i] != 'Sniff'  and Data1['Beh'][i] != 'Sniff A\G'and Data1['Beh'][i] != 'Upright':
        listIndexD.append(Data1['Beh'][i])
Tf = Data1.Beh.isin(listIndexD)
Data1 ['tf'] = Tf
Tf = pd.Series.tolist(Tf)
Data1 = Data1[Data1.tf == False]
Data1 = Data1.drop('Beh',1)

#pre-elaboration d2
listIndexD = []
for i in range (len (Data2)):
    if Data2['Beh'][i] != 'Attack' and Data2['Beh'][i] != 'face the consp' and  Data2['Beh'][i] != 'Sniff'  and Data2['Beh'][i] != 'Sniff A\G'and Data2['Beh'][i] != 'Upright':
        listIndexD.append(Data2['Beh'][i])
Tf = Data2.Beh.isin(listIndexD)
Data2 ['tf'] = Tf
Tf = pd.Series.tolist(Tf)
Data2 = Data2[Data2.tf == False]
Data2 = Data2.drop('Beh',1)
Data3 = Data3.drop('Beh',1)

#drop the neurons that are present only in day 1 or 2 or 3
if len (Data1.transpose()) > len(Data2.transpose()):
    for col in Data1.columns.difference(Data2.columns):
        Data1 = Data1.drop(col,1)
else: 
    for col in Data2.columns.difference(Data1.columns):
        Data2 = Data2.drop(col,1)
        
if len (Data3.transpose()) > len(Data2.transpose()):
    for col in Data3.columns.difference(Data2.columns):
        Data3 = Data3.drop(col,1)
else: 
    for col in Data2.columns.difference(Data3.columns):
        Data2 = Data2.drop(col,1)
        
if len (Data1.transpose()) > len(Data3.transpose()):
    for col in Data1.columns.difference(Data3.columns):
        Data1 = Data1.drop(col,1)
else: 
    for col in Data3.columns.difference(Data1.columns):
        Data3 = Data3.drop(col,1)
        
DataFrameList = [Data1,Data2, Data3]
ActivityList = ['-1','1', '0']
ActivityList2 = ['sf', 'agg3', 'otherwise']
colors = ['k','y','r']
x, y = CreationDataFrame (DataFrameList,3,ActivityList)
pDf, var = Pls (x,y, 'state')
im = Graphing(pDf, 'pls SOCIAL FEAR VS AGGRESSION', ActivityList,ActivityList2, colors, 'state' )