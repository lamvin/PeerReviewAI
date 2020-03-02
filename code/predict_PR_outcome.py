# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:52:48 2019

@author: User1
"""
import pandas as pd
import numpy as np
from scipy import stats
import os

path = "C:/Users/philv/Documents/Projects/PeerRead/"
output_path = path + 'results/prediction/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
with open('{}/data/processed/df_data_stem.csv'.format(path),'r',encoding='utf-8') as f:
    papers_stem = pd.read_csv(f,sep='\t',index_col=0)


fields = ['title','abstract','introduction']
platforms = ['arxiv.cs.ai_2007-2017','arxiv.cs.cl_2007-2017','arxiv.cs.lg_2007-2017','iclr_2017']

from sklearn.feature_extraction.text import TfidfVectorizer
data = {}
tfidf_data = {}
for field in fields:
    data[field] = papers_stem[[field,'accepted','platform']]
    data[field] = data[field].dropna()
    
    vect_word = TfidfVectorizer( lowercase=True, analyzer='word',
                        stop_words= 'english',dtype=np.float32)
    tfidf_data[field] = vect_word.fit_transform(data[field][field])


from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression

k = 10
kfold = KFold(k, True, 1)
metrics = ['Precision','Recall','F-score']
model_stats = pd.DataFrame(columns=metrics)
platforms = ['arxiv.cs.ai_2007-2017','arxiv.cs.cl_2007-2017','arxiv.cs.lg_2007-2017','iclr_2017']
platforms_keys = {'arxiv.cs.ai_2007-2017':'AI','arxiv.cs.cl_2007-2017':'CL','arxiv.cs.lg_2007-2017':'LG','iclr_2017':'ICLR'}
keys = list(platforms_keys.values())
keys.append('All')

for field in fields:
    model_stats = pd.DataFrame(columns=metrics)
    for i in range(len(keys)):
        results = np.zeros((k,4)) 
        if i == 4:
             predictors = tfidf_data[field]
             outcome  = np.array(data[field]['accepted'].copy())
             key = 'All'
        else:
            platform = platforms[i]
            predictors = tfidf_data[field]
            predictors = predictors[np.where(data[field]['platform']==platform)[0],:]
            outcome = np.array(data[field]['accepted'].copy())
            outcome = outcome[data[field]['platform']==platform]
            key = platforms_keys[platform]
        j = 0
        for train, test in kfold.split(outcome):
            X_train = predictors[train]
            y_train = outcome[train]
            X_test = predictors[test]
            y_test =  outcome[test]
    
            model = LogisticRegression().fit(X_train, y_train)  
            predicted = model.predict(X_test)
            
            results[j,:] = precision_recall_fscore_support(y_test,predicted,average='macro')
            j+=1
            
            
        avg_results = np.mean(results,axis=0)
        avg_sem = stats.sem(results,axis=0)
        model_avg = np.char.add(np.round(avg_results[:-1],3).astype(np.str),' ± ')
        model_sem = np.round(avg_sem[:-1],3).astype(np.str)
        model_res = np.core.defchararray.add(model_avg, model_sem)
        model_stats = model_stats.append(pd.Series(model_res,name=key,index=metrics))
        model_stats.to_csv(path + '{}{}_stats.csv'.format(output_path,field))   
   
# =============================================================================
# Prediction accuracy for permutated labels
# =============================================================================
for field in fields:
    model_stats = pd.DataFrame(columns=metrics)
    for i in range(len(keys)):
        results = np.zeros((k,4)) 
        if i == 4:
             predictors = tfidf_data[field]
             outcome  = np.array(data[field]['accepted'].copy())
             key = 'All'
        else:
            platform = platforms[i]
            predictors = tfidf_data[field]
            predictors = predictors[np.where(data[field]['platform']==platform)[0],:]
            outcome = np.array(data[field]['accepted'].copy())
            outcome = outcome[data[field]['platform']==platform]
            key = platforms_keys[platform]
        j = 0
        for train, test in kfold.split(outcome):
            X_train = predictors[train]
            y_train = outcome[train]
            X_test = predictors[test]
            y_test =  outcome[test]
            np.random.shuffle(y_test)
    
            model = LogisticRegression().fit(X_train, y_train)  
            predicted = model.predict(X_test)
            
            results[j,:] = precision_recall_fscore_support(y_test,predicted,average='macro')
            j+=1
            
            
        avg_results = np.mean(results,axis=0)
        avg_sem = stats.sem(results,axis=0)
        model_avg = np.char.add(np.round(avg_results[:-1],3).astype(np.str),' ± ')
        model_sem = np.round(avg_sem[:-1],3).astype(np.str)
        model_res = np.core.defchararray.add(model_avg, model_sem)
        model_stats = model_stats.append(pd.Series(model_res,name=key,index=metrics))
        model_stats.to_csv('{}{}_stats_shuffle.csv'.format(output_path,field))   
