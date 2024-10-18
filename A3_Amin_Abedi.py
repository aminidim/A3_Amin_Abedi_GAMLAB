# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 20:58:22 2024

@author: amin abedi
"""
#==========the goal is finding the best model to classify the dataset==========

#importing and setting necessarry elements  
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
kf=KFold(n_splits=5,shuffle=True,random_state=42)

#preparing dataset, data and the targets
dataset=load_breast_cancer()
data=dataset.data
targets=dataset.target

#============================1-LogisticRegression=============================
from sklearn.linear_model import LogisticRegression
model1=LogisticRegression()
gs1=GridSearchCV(model1,{}, cv=kf,scoring='accuracy')
gs1.fit(data,targets)
model1_best_score=gs1.best_score_
#==============================================================================

#========================2-K-Nearest-Neighbor==================================
from sklearn.neighbors import KNeighborsClassifier
model2=KNeighborsClassifier()
KNN_Params={'n_neighbors':[1,2,3,4,7,10,11,12,20,30]}
gs2=GridSearchCV(model2, KNN_Params,cv=kf,scoring='accuracy')
gs2.fit(data,targets)
model2_best_parameters=gs2.best_params_
model2_best_score=gs2.best_score_
#==============================================================================

#=============================3-DecisionTree===================================
from sklearn.tree import DecisionTreeClassifier
model3=DecisionTreeClassifier()
DT_Params={'max_depth':[1,2,3,4,5,6,7,8,9]}
gs3=GridSearchCV(model3, DT_Params, cv=kf, scoring='accuracy')
gs3.fit(data,targets)
model3_best_parameters=gs3.best_params_
model3_best_score=gs3.best_score_
#==============================================================================

#==============================4-RandomForest==================================
from sklearn.ensemble import RandomForestClassifier
model4=RandomForestClassifier(random_state=42)
RF_Params={'n_estimators':[10,20,30,40,50,60,67,68,69,70,71,80,90]}
gs4=GridSearchCV(model4, RF_Params,cv=kf,scoring='accuracy')
gs4.fit(data,targets)
model4_best_parameters=gs4.best_params_
model4_best_score=gs4.best_score_
#==============================================================================

#===================================5-SVC======================================
from sklearn.svm import SVC
model5=SVC()
SVC_Params={'kernel':['poly','rbf','sigmoid'],'degree':[2,3,4,5],'C':[0.01,0.1,1]}
gs5=GridSearchCV(model5, SVC_Params,cv=kf,scoring='accuracy')
scaler=MinMaxScaler()
scaled_data=scaler.fit_transform(data)
gs5.fit(scaled_data,targets)
model5_best_parameters=gs5.best_params_
model5_best_score=gs5.best_score_
#==============================================================================

#======================the best score of each model============================
print('best score of LR: ',model1_best_score)
#best score of LR:  0.9402111473373699

print('best score of KNN: ',model2_best_score)
#best score of KNN:  0.9402111473373699

print('best score of DT: ',model3_best_score)
#best score of DT:  0.9455208818506444

print('best score of RF: ',model4_best_score)
#best score of RF:  0.9613414066138798

print('best score of SVC: ',model5_best_score)
#best score of SVC:  0.9806551777674274
#==============================================================================

#===============================conclusion=====================================
'''
by comparing the best score of each 5 models it's obvious that the best 
model is SVC which has the highest score
you cna see the best parameters and score of this model below:
'''
print('best parameters of best model: ',model5_best_parameters)
#best parameters of best model:  {'C': 1, 'degree': 2, 'kernel': 'rbf'}
#best score of SVC:  0.9806551777674274
#==============================================================================









