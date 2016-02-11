#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'pjaeger'


"""
Analysis script.
This is actually performed in a Jupyter Notebook.
This way the DataFrame gets stashed in the first cell
and fast working is possible respectively direct plot outputs. 
"""

%matplotlib inline
import pandas as pd
from root_pandas import read_root
import b2stat
import b2plot 
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import functions
import argparse
from tabulate import tabulate
from collections import OrderedDict
from scipy import optimize
from functions import *

# Set nice searborn settings
seaborn.set(font_scale=2)
seaborn.set_style('whitegrid')

inputFile = '/local/scratch/hdd/pjaeger/files/3ks/ntup/allMC.root'
saveOut = '/autofs/users/users/pjaeger/masterarbeit/talks/latex/b2gm2016/pics/'
n = 1000000
df= read_root(inputFile)

############################# Stats #########################################################################
 """
 Get Info, Statistics and Background contents of the Tree.
 """

# Get list of variables in the tree.
for key in df:
    print (key)
print()
print (df["btruthtree"].groupby(["B0_mcPDG"]).size())
df["btree"].columns.values.tolist()

# Get ratio of KsCandidates/BCandidates per event.
ks = pd.DataFrame(df["kstree"][0:10000].groupby('evt_no').size())
ks.columns=["kscands"]
b = pd.DataFrame(df["btree"][0:100].groupby('evt_no').size())
b.columns=["bcands"]
print(b.join(ks))

getKssStats(df,n)
getBg(df,"kstree")
getBg(df,"btree")

############################# B0 deltaTdist #################################################################
  """ 
  Plot the two flavour-tagged deltaT histograms to show CP-Violation.
  """ 
dist=b2plot.Distribution(figure=plt.figure())
dist.bins= 50
dist.fill=True
dist.add(shortFrame,'K_S0____boBDTout__bc', shortFrame["K_S0_mcPDG"]==310)
dist.add(shortFrame,'K_S0____boBDTout__bc',shortFrame["K_S0_mcPDG"]!=310)
dist.labels=["signal","background"]
dist.set_plot_options( plot_kwargs={ 'marker':'.','fillstyle':'full',"alpha":"0.5",'linewidth':'4'})
dist.ymax=80000
dist.finish()
dist.axis.set_title('BDT Output')
dist.axis.xaxis.set_label_text('Classifier output')
dist.save('{}ksbdtout.eps'.format(saveOut))

############################# Delta T RESOLUTION ############################################################
  """ 
  Fit the finite Vertex reolution with a triple-Gaussian function in order to forward convolute the model.
  """ 

# prepare DataFrame
df["btree"]["reso"]=df["btree"].B0_DeltaT-df["btree"].B0_TruthDeltaT
shortFrame=df["btree"][(df["btree"]["B0_DeltaT"]>=-100) & (df["btree"]["B0_DeltaT"]<=100)]
mask_reso = (shortFrame["reso"]>-15) & (shortFrame["reso"]<=15)

delta=b2plot.Distribution(figure=plt.figure(),keep_first_binning=True)
delta.bins=120 
delta.unit = "ps"
delta.add(shortFrame[mask_reso],"reso")
delta.performFit()
delta.labels=["hist","triple-Gauss"]
delta.finish()
delta.axis.xaxis.set_label_text(r'$\Delta t - \Delta t_{truth}[ps]$' )
delta.axis.set_title(r'$\Delta t$ Resolution')
delta.save('{}resolution.eps'.format(saveOut))

# get result fit-values.
delta.getPopt()

############################# Convolution and Maximum Likelihood Fit ######################################

  """ 
  Numeric convolution to the model is performed according to: \int [P_{sig}(\Delta t ') R_{sig}(\Delta t_i - \Delta t ')] d(\Delta t ').
  Determine the time dependent CPV-parameters S,A from the fit.
  """ 

result = optimize.minimize(deltaTFit, (-0.7, 0.),args= shortFrame, method='Nelder-Mead')
S,A = result.x[0],result.x[1]
print(result)
print("sin(2phi)= -S :", -S)
print("A: ", A)

############################# Get Uncertainty ##########################################################
 
  """ 
  Calculate numeric 1 sigma uncertainty in order to feed it back to the BDToutput.
  """ 

# Perform numeric scan over NegLogLikelihood to find the min+0.5= 1 sigma value
X = np.linspace(-1.5, 0.5, 1000)
sarray = np.array([deltaTFit([x,A]) for x in X])
aarray = np.array([deltaTFit([S,x]) for x in X])                   

getUnc(sarray,-20,30)
getUnc(aarray,-20,30)
