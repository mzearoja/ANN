#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 16:58:26 2021

@author: mpzr44
"""

Artificial Neural Network (ANN) Model development using CEPF data
Augusto Souza - Purdue University, Institute for Plant Sciences Edited by Maria Zea


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score,confusion_matrix, ConfusionMatrixDisplay

import scipy.signal as sp

# Reading the data from the excel file
df = pd.read_excel(r'C:\Users\amagalha\Documents\HS - Processed Data\Maria - Basil and Kale\KaleBasilSide .xlsx','mixed')

# For Cadmium prediction, use df.Cd. This way, the unit is "per 500 mg of dry weight"
y = df.Cd


X = df.filter(like='WL_') # X is the VNIR reflectance data

# Getting rid of rows with empty data
ind = np.isnan(y)

y = y[~ind];
X = X[~ind];

# Deleting and Selecting just a portion of the wavelengths:
# the first wavelengths can be very noisy, so the first part of them are deleted. 
#For example, if wlCutOff is equal to 47, the first 47 wavelengths will be deleted from X
wlCutOff = 47
X = X.drop(X.columns[np.arange(wlCutOff)], axis = 1)

# Also, if you want to use just some wavelengths, not all of them, you can adjust the step size.
# This will divide the wavelengths by the step value. 
# For example, if step is 10 and the total wavelengths is 450, then 45 will be used
step = 1
X = X.iloc[:,np.arange(0,X.shape[1],step)]

wlName = X.columns
wlName = wlName.str.replace('WL_','')
wlName = wlName.str.replace('_','.')
wlNum = wlName.astype(float)

# %% Data pre-processing
# Some sources state that pre-processing helps in the estimation model
# Below are some of them or a combination of two methods


# One source convert the reflectance in absorbance before the pre-processing. This equation is common in other spectroscopy sources
X_corr = np.log10(1/X)

# Standard Normal Variate (SNV)
uX = np.mean(X_corr,1)
deltaX = np.std(X_corr,1)

X1 = (np.array(X_corr)-uX.values.reshape(-1,1))/deltaX.values.reshape(-1,1)


# First derivative using the Savitzky-Golay filter
X2 = sp.savgol_filter(X_corr,window_length = 5,polyorder = 3,deriv = 1, axis=0)


# Detrend method
X3 = sp.detrend(X_corr,axis=0)

# If you want to take a look at this data, just comment/uncomment the following plot lines
fig, ax1 = plt.subplots()
color = 'red'
ax1.set_xlabel('Wavelength (nm)')
ax1.plot(wlNum,X.iloc[1,:],color='red')
ax1.plot(wlNum,X_corr.iloc[1,:],color='red',linestyle='dashed')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(axis='both',lw=0.5)
ax2 = ax1.twinx()

color = 'blue'
ax2.set_ylabel('Normalized Absorbance (dec.)',color=color)  # we already handled the x-label with ax1
ax2.plot(wlNum,X1[1][:],color='blue', ls = 'dashed')
ax2.tick_params(axis='y', labelcolor=color)

ax1.legend(labels=('Reflectance','Absorbance'))

ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax1.get_yticks())))

plt.show()




# Combination of SNV and the other two methods
X2norm = sp.savgol_filter(X1,window_length = 5,polyorder = 3,deriv = 1, axis=0)

X3norm = sp.detrend(X1,axis=0)


plt.figure()
plt.plot(wlNum,X2[1][:],'magenta')
plt.plot(wlNum,X2norm[1][:],'green')
plt.grid(axis='both',lw=0.5)
plt.xlabel('Wavelentgh (nm)')
plt.legend(labels = ('1st derivative','Norm. 1st derivative'))
ax = plt.gca()
ax.set_facecolor((0.75,0.75,0.75))
plt.show()

plt.show()


plt.figure()
plt.plot(wlNum,X3[1][:],'cyan')
plt.plot(wlNum,X3norm[1][:],'yellow')
plt.grid(axis='both',lw=0.5)
plt.xlabel('Wavelentgh (nm)')
plt.legend(labels = ('Detrended','Norm. Detrended'))
ax = plt.gca()
ax.set_facecolor((0.75,0.75,0.75))
plt.show()

# %%

# Choose the preferred method by changing the X variable below
# X is the raw reflectance
# X_corr is the raw absorbance
# X1 is the normalized absorbance
# X2 is the first derivative of the absorbance
# X3 is the detrended absorbance
# X2norm is the first derivative of the normalized absorbance
# X3norm is the normalized detrended absorbance

X = X3norm
indy = y >= 1 # threshold for toxicity

y = indy

X_train, X_test , y_train, y_test = model_selection.train_test_split(X, y, stratify = y, test_size=0.25, random_state=1)

# Change the mode parameters here, such as learning rate, number of layers, maximum interations before the training stops...
# regr = MLPRegressor(solver='lbfgs', alpha=0.01,hidden_layer_sizes=(50,), random_state=1, max_iter = 10000)
regr = MLPClassifier(solver='lbfgs', alpha=1e-10,hidden_layer_sizes=(50,), random_state=1, max_iter = 1000)

regr.fit(X_train, y_train)


# Predictring the results based on the model
pred_train = regr.predict(X_train)
pred_test = regr.predict(X_test)

# %% Plotting the results
# plt.figure()
# plt.scatter(pred_train,y_train)
# plt.scatter(pred_test,y_test)

# plt.legend(labels=('Training','Testing'))

# plt.title('Neural Network')
# plt.ylabel('Measured Scaled Cadmium')
# plt.xlabel('Estimated Scaled Cadmium')
# plt.grid(axis='both',lw=0.5)
# # plt.xlim(xmin=0.40,xmax = 1.55)
# # plt.ylim(ymin=0.40,ymax = 1.55)
# plt.axis('auto')
# plt.show()

cm = confusion_matrix(y_train, pred_train)
cm = confusion_matrix(y_test, pred_test)
cm_display = ConfusionMatrixDisplay(cm,2).plot()

print('Score testing: ',regr.score(X_test, y_test))
print('Score training: ',regr.score(X_train, y_train))

# print('MSE testing: ',mean_squared_error(y_train, pred_train))
# print('MSE training: ',mean_squared_error(y_test, pred_test))


