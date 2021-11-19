#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 01:20:17 2021

successful build [pseudocode]:

    1. Be able to parse through data sets (year, players, etc)
    2. Rank by any statisical category and any given input window
    3. Apply regression algorithims to project future performance
    4. Build a GUI
 

@author: christopher.vinton
"""

import time
import numpy as np
import os
import pandas as pd
import random
import matplotlib.pyplot as plt


from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as r2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LogisticRegression as logit
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

# Data Import
# Choose your data years
#################################################

# 2019

data_dir = r'/Users/christopher.vinton/desktop/Fantasy_data_v2/yearly'
fname = r'2019.csv'
path = os.path.join(data_dir,fname) #139 data points
data2019 = pd.read_csv(path)
#players = 2019_data.Player


# 2018

fname = r'2018.csv'
path = os.path.join(data_dir,fname) #139 data points
data2018 = pd.read_csv(path)

# 2017

fname = r'2017.csv'
path = os.path.join(data_dir,fname) #139 data points
data2017 = pd.read_csv(path)

# 2016

fname = r'2016.csv'
path = os.path.join(data_dir,fname) #139 data points
data2016 = pd.read_csv(path)

# 2015

fname = r'2015.csv'
path = os.path.join(data_dir,fname) #139 data points
data2015 = pd.read_csv(path)

# 2014

fname = r'2014.csv'
path = os.path.join(data_dir,fname) #139 data points
data2014 = pd.read_csv(path)

# 2013

fname = r'2013.csv'
path = os.path.join(data_dir,fname) #139 data points
data2013 = pd.read_csv(path)

# 2012

fname = r'2012.csv'
path = os.path.join(data_dir,fname) #139 data points
data2012 = pd.read_csv(path)

# 2011

fname = r'2011.csv'
path = os.path.join(data_dir,fname) #139 data points
data2011 = pd.read_csv(path)# 2018

# 2010
fname = r'2010.csv'
path = os.path.join(data_dir,fname) #139 data points
data2010 = pd.read_csv(path)

# 2009

data_dir = r'/Users/christopher.vinton/desktop/Fantasy_data_v2/yearly'
fname = r'2009.csv'
path = os.path.join(data_dir,fname) #139 data points
data2009 = pd.read_csv(path)
#players = 2019_data.Player


# 2008

fname = r'2008.csv'
path = os.path.join(data_dir,fname) #139 data points
data2008 = pd.read_csv(path)

# 2007

fname = r'2007.csv'
path = os.path.join(data_dir,fname) #139 data points
data2007 = pd.read_csv(path)

# 2006

fname = r'2006.csv'
path = os.path.join(data_dir,fname) #139 data points
data2006 = pd.read_csv(path)

# 2005

fname = r'2005.csv'
path = os.path.join(data_dir,fname) #139 data points
data2005 = pd.read_csv(path)

# 2004

fname = r'2004.csv'
path = os.path.join(data_dir,fname) #139 data points
data2004 = pd.read_csv(path)

# 2003

fname = r'2003.csv'
path = os.path.join(data_dir,fname) #139 data points
data2003 = pd.read_csv(path)

# 2002

fname = r'2002.csv'
path = os.path.join(data_dir,fname) #139 data points
data2002 = pd.read_csv(path)

# 2001

fname = r'2001.csv'
path = os.path.join(data_dir,fname) #139 data points
data2001 = pd.read_csv(path)# 2018

# 2000
fname = r'2000.csv'
path = os.path.join(data_dir,fname) #139 data points
data2000 = pd.read_csv(path)






# joining data into one table
####################################
datalist = [data2019,data2018,data2017,data2016,data2015,data2014,data2013,data2012,data2011,data2010,
            data2009,data2008,data2007,data2006,data2005,data2004,data2003,data2002,data2001,data2000]
allplayers = pd.concat(datalist)



# Seperating players by positions for a single year. 
####################################
qbs = data2019.loc[data2019['Pos'] == 'QB']
rbs = data2019.loc[data2019['Pos'] == 'RB']
wrs = data2019.loc[data2019['Pos'] == 'WR']
tes = data2019.loc[data2019['Pos'] == 'TE']

# all years
####################################
allqb = allplayers.loc[allplayers['Pos'] == 'QB']
allrb = allplayers.loc[allplayers['Pos'] == 'RB']
allwr = allplayers.loc[allplayers['Pos'] == 'WR']
allte = allplayers.loc[allplayers['Pos'] == 'TE']
allskill = allplayers.drop(allplayers[allplayers.Pos == 'QB'].index)



# Rank/Sort Position by desired feature
####################################
#Can Create a custom rank feature from another data point
# qbs['PassYardRank'] = qbs['Yds'].rank(ascending = False)

# # Pass Yards
# Sorted_pass_yards = allqb.sort_values(by=['Yds'],ascending=False)
# print(Sorted_pass_yards)

# # Age
# Sorted_age = qbs.sort_values(by=['Age'],ascending=False)
# print(Sorted_age)

# # Pass TD

# Sorted_TD = qbs.sort_values(by=['PassingTD'],ascending=False)
# print(Sorted_TD)







# LINEAR REGRESSION MODEL - QUARTERBACKS
####################################


# Pre Processing
###########################################################
# selects a player and creates a table of just that players statistical data
specific_player = allplayers.loc[allplayers['Player'] == 'Tom Brady']
rodg_input = specific_player.drop(columns = ['Player', 'Tm', 'Pos', 'PassingTD','Unnamed: 0','FantasyPoints'])
rodg_target = specific_player.loc[:,['PassingTD']]

QB_input = allqb
QB_input = QB_input.drop(QB_input[QB_input.Player == 'Tom Brady'].index)
QB_target = QB_input.loc[:,['PassingTD']]
QB_input = QB_input.drop(columns = ['Player', 'Tm', 'Pos', 'PassingTD', 'Unnamed: 0', 'FantasyPoints'])


xtrain, xtest, ytrain, ytest = train_test_split(QB_input, QB_target, test_size=0.3, random_state=42)


# #########################################
# Linear Regression
reg = linear_model.LinearRegression()


# Fit regression  model and coefficients
#reg.fit(xtrain.values.reshape(-1,1),ytrain.values.reshape(-1,1))
reg.fit(xtrain, ytrain)
coeff = reg.coef_
interp = reg.intercept_
linear_model.LinearRegression()

# Prediction
lin_pred = reg.predict(rodg_input)

# RMSE predict
rms_lin = np.sqrt(MSE(lin_pred,rodg_target))
print('\nThe RMSE of my Linear Regession Model is: '+str(rms_lin))

# #########################################
# Multi-Layered Perceptron
# Creating the classifier model
clf_multi_perc = MLPRegressor(activation = "relu", solver = 'adam', alpha = 1e-5, hidden_layer_sizes = (2,100), max_iter =1000, random_state = 1) 
clf_multi_perc.fit(xtrain,ytrain)

# Prediction
mlp_predict = clf_multi_perc.predict(rodg_input)


# RMSE MLP
rms_mlp = np.sqrt(MSE(mlp_predict,rodg_target))
print('\nThe RMSE of my MLP is: '+str(rms_mlp))

# #########################################
# Random Forrest

clf_random_forrest = RandomForestClassifier(n_estimators=100)
clf_random_forrest.fit(xtrain, ytrain)

RF_pred = clf_random_forrest.predict(rodg_input)


# RMSE RF
rms_rf = np.sqrt(MSE(RF_pred,rodg_target))
print('\nThe RMSE of my Random Forrest is: '+str(rms_rf))


# #########################################
# Support Vector Machine
clf_svm = svm.SVC(kernel = 'rbf',C=1)
clf_svm.kernel
clf_svm.fit(xtrain,ytrain)

SVM_pred = clf_svm.predict(rodg_input)

# RMSE SVM
rms_SVM = np.sqrt(MSE(SVM_pred,rodg_target))
print('\nThe RMSE of my SVM is: '+str(rms_SVM))


#xax = np.linspace(1,20, 20)
xax = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
plt.figure(1)
plt.title('Tom Brady Pass Yard Prediction')
plt.xlabel('Year')
plt.ylabel('Touchdowns')
plt.plot(xax,lin_pred, color = 'red', label = "Predicted") 
plt.scatter(xax,rodg_target, color = 'black', label = "Actual") 
plt.plot(xax, SVM_pred, color = 'blue', label = 'SVM')
plt.plot(xax, RF_pred, color = 'green', label = 'Random Forrest')
plt.plot(xax, mlp_predict, color = 'yellow', label = 'MLP')
plt.legend()


RMSE_dict = {'Linear': rms_lin, 'MLP': rms_mlp, 'Random Forrest':rms_rf, 'SVM': rms_SVM }
rms_type = list(RMSE_dict.keys())
rms_val = list(RMSE_dict.values())
plt.figure(2)
plt.bar(rms_type, rms_val, color = 'blue', width = 0.5 )
plt.xlabel('Type')
plt.ylabel('RMSE Value')



# #########################################
# Create CSV

csv_qb = allqb.drop(columns = ['Tm', 'Pos', 'Unnamed: 0','FantasyPoints', 'Tgt', 'Rec', 'Yds.2', 'Y/R', 'Att', 'Yds', 'Att.1', 'Yds.1'])
csv_qb.to_csv('qbdata.csv')



# RMSE predict
#rms_lin = np.sqrt(MSE(lin_pred,ryan_target))
# print('\nThe RMSE of my Linear reg is: '+str(rms_lin))

# plt.figure(1)
# plt.title('Tom Brady Pass Yard Prediction')
# plt.xlabel('Year')
# plt.ylabel('Pass Yards')
# plt.plot(xax,lin_pred, color = 'red', label = "Predicted") 
# plt.scatter(xax,ryan_target, color = 'black', label = "Actual") 
# plt.legend()


## Prediction (usage) scenarios


# #########################################
# Polynomial Regression


# Numbers of data samples
# n_samples = 20

# # Number of polynomial degrees for CV
# poly_deg = range(1,4)


# # Initalize cv scores
# cv_scores = [0]*len(poly_deg)


# xlin = np.linspace(23,42,20)





# poly_features = PolynomialFeatures(degree = 1, include_bias= True)
    

    
# # Setup Model Pipeline
# pipeline = Pipeline([("poly_features", poly_features),("lin_reg",reg)])
    
# # Fit Model
# pipeline.fit(QB_input,QB_target)

    
# # Plotting

# poly_pred = pipeline.predict(ryan_input)

    
# plt.figure()
# plt.title("Tom Brady TDs")
# plt.xlabel('Age')
# plt.ylabel('Number of TDs')
# plt.plot(xlin,poly_pred, color = 'black', linewidth=3, label = "Model")
# plt.plot(xlin,ryan_target, 'r--', label="Real Life")
# #plt.scatter(xlin,QB_target, edgecolor = 'b', label="Samples")
# plt.legend()
    





# LINEAR REGRESSION MODEL - RUNNINGBACKS
####################################


# Pre Processing
###########################################################
# specific_playerRB = allplayers.loc[allplayers['Player'] == 'LeSean McCoy']
# lesean_input = specific_playerRB.drop(columns = ['Player', 'Tm', 'Pos', 'RushingYds','Yds.1','Unnamed: 0','FantasyPoints'])
# lesean_target = specific_playerRB.loc[:,['RushingYds']]


# RB_input = allrb
# RB_input = RB_input.drop(RB_input[RB_input.Player == 'LeSean McCoy'].index)
# RB_target = RB_input.loc[:,['RushingYds']]
# RB_input = RB_input.drop(columns = ['Player', 'Tm', 'Pos', 'RushingYds','Yds.1', 'Unnamed: 0', 'FantasyPoints'])


# xtrain, xtest, ytrain, ytest = train_test_split(RB_input, RB_target, test_size=0.3, random_state=42)


# # #########################################
# # Linear Regression
# regRB = linear_model.LinearRegression()


# # Fit regression  model and coefficients
# #reg.fit(xtrain.values.reshape(-1,1),ytrain.values.reshape(-1,1))
# regRB.fit(xtrain, ytrain)
# #coeff = regRB.coef_
# #interp = regRB.intercept_
# #linear_model.LinearRegression()


# lin_predRB = regRB.predict(lesean_input)
# xax = np.linspace(1,20, 20)
# # y_test_plot = np.linspace(1,6, 6)
# # y_test_plot = y_test_plot.reshape(1,6)


# # RMSE predict
# rms_lin = np.sqrt(MSE(lin_predRB,lesean_target))
# print('\nThe RMSE of my Linear reg is: '+str(rms_lin))

# plt.figure(1)
# plt.title('Tom Brady Pass Yard Prediction')
# plt.xlabel('Year')
# plt.ylabel('Pass Yards')
# plt.plot(xax,lin_pred, color = 'red', label = "Predicted") 
# plt.scatter(xax,brady_target, color = 'black', label = "Actual") 
# plt.legend()
























##  drop function
##  allqb.drop(columns = ['Player', 'Tm', 'Pos', 'Unnamed: 0', 'PassingYds'])
















# processing data and assigning numerical classifiers for various columns of the dataframe so
# the decision tree can compute them

## Team
###########################################################
# team_column = 'Tm'
# AZ = allplayers.Tm == 'ARI'
# ATL  = allplayer.Tm == 'ATL'
# BAL = allplayers.Tm == 'BAL'
# BUF  = allplayer.Tm == 'BUF'
# CAR = allplayers.Tm == 'CAR'
# CIN  = allplayer.Tm == 'ATL'
# CHI = allplayers.Tm == 'ARI'
# CLE  = allplayer.Tm == 'ATL'
# DAL = allplayers.Tm == 'ARI'
# DEN  = allplayer.Tm == 'ATL'
# DET = allplayers.Tm == 'ARI'
# HOU  = allplayer.Tm == 'ATL'
# GB = allplayers.Tm == 'ARI'
# IND  = allplayer.Tm == 'ATL'
# LAR = allplayers.Tm == 'ARI'
# LAC  = allplayer.Tm == 'ATL'
# JAC = allplayers.Tm == 'ARI'
# MIN  = allplayer.Tm == 'ATL'
# KC = allplayers.Tm == 'ARI'
# NO  = allplayer.Tm == 'ATL'
# LVR = allplayers.Tm == 'ARI'
# NYG  = allplayer.Tm == 'ATL'
# PHI = allplayers.Tm == 'ARI'
# MIA  = allplayer.Tm == 'ATL'
# SF = allplayers.Tm == 'ARI'
# NWE  = allplayer.Tm == 'ATL'
# SEA = allplayers.Tm == 'ARI'
# NYJ  = allplayer.Tm == 'ATL'
# TB = allplayers.Tm == 'ARI'
# PIT  = allplayer.Tm == 'ATL'
# TEN = allplayers.Tm == 'ARI'
# WFT  = allplayer.Tm == 'ATL'

# married_yes = 1
# married_no = 0

# X.loc[yes_condition, married_column] = married_yes
# X.loc[no_condition, married_column] = married_no






















## Dead code for ideas:
    
    
    
# Get functions for ranking
#def get_rank(position):
#    return position.get('Player')

#def get_rank(position):
#    return position.get('Yds')

#def get_rank(qbs):
#    return qbs.get("Player","Yds")


# qbs2 = data2018.loc[data2019['Pos'] == 'QB']


# testtable = [qbs,qbs2]
# qb = pd.concat(testtable)


