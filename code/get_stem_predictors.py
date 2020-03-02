# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:59:11 2020

@author: philv
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import os

path = "C:/Users/philv/Documents/Projects/PeerRead/"
output_path = path + 'data/prediction/fig_tfidf'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
with open('{}/data/processed/df_data_stem.csv'.format(path),'r',encoding='utf-8') as f:
    papers_stem = pd.read_csv(f,sep='\t',index_col=0)



fields = ['title','abstract','introduction']
platforms  = {'arxiv.cs.ai_2007-2017':'AI','arxiv.cs.cl_2007-2017':'CL','arxiv.cs.lg_2007-2017':'LG','iclr_2017':'ICLR'}
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
data = {}
count_data = {}
vec_data = {}
for field in fields:
    data[field] = papers_stem[[field,'accepted','platform']]
    data[field] = data[field].dropna()
    
    vec_data[field] = TfidfVectorizer( lowercase=True, analyzer='word',
                        stop_words= 'english',dtype=np.float32)
    count_data[field] = vec_data[field].fit_transform(data[field][field])
    
    
keys = list(platforms.values())
keys.append('All')
outcomes = ['Stem','Score accepted','Score rejected']
variables = [x+' ' +y for x in keys for y in outcomes]
top_plot = 15
top = 50

for field in fields:
    counts = count_data[field]
    model_words_acc = pd.DataFrame(np.nan,columns=variables,index=np.arange(top))
    model_words_rej = pd.DataFrame(np.nan,columns=variables,index=np.arange(top))
    for i in range(5):
        counts = count_data[field]
        df = data[field]
        key = keys[i]
        if i < 4:
            platform = list(platforms.keys())[i]
            df = df[df['platform']==platform]
            counts = counts[np.where(df['platform']==platform)[0],:]

            
        total_count = np.sum(counts,axis=0)
        ft_names = np.array(vec_data[field].get_feature_names())
        n_items = counts.shape[0]
        idx = np.where(df['accepted'])[0]
        counts = np.array(counts.todense())
    
        count_accepted = counts[np.isin(np.arange(n_items),idx),:]
        count_rejected = counts[~np.isin(np.arange(n_items),idx),:]
        
        avg_acc=np.mean(count_accepted,axis=0)
        avg_rej=np.mean(count_rejected,axis=0)
        var_acc=np.var(count_accepted,axis=0)
        var_rej=np.var(count_rejected,axis=0)
        sem_acc=stats.sem(count_accepted,axis=0)
        sem_rej=stats.sem(count_rejected,axis=0)
        
        diff = avg_acc-avg_rej
        comb_var = var_acc+var_rej
        comb_sem = comb_var/np.sqrt(n_items)
        w_sort = diff.argsort()
        
        diff_sorted = diff[w_sort]
        var_sorted = comb_var[w_sort]
        ft_sorted = ft_names[w_sort]
        
        avg_acc_sorted = avg_acc[w_sort]
        avg_rej_sorted = avg_rej[w_sort]
        sem_acc_sorted = sem_acc[w_sort]
        sem_rej_sorted = sem_rej[w_sort]
        
        
        
        #ACCEPTED
        model_avg_acc = np.char.add(np.round(avg_acc_sorted[-top:],4).astype(np.str),' ± ')
        model_sem_acc = np.round(sem_acc_sorted[-top:],4).astype(np.str)
        model_res_acc = np.core.defchararray.add(model_avg_acc, model_sem_acc)
        model_avg_rej = np.char.add(np.round(avg_rej_sorted[-top:],4).astype(np.str),' ± ')
        model_sem_rej = np.round(sem_rej_sorted[-top:],4).astype(np.str)
        model_res_rej = np.core.defchararray.add(model_avg_rej, model_sem_rej)
        importance= np.char.add(np.round(diff[-top:],4).astype(np.str),' ± ')
        importance = np.core.defchararray.add(importance, np.round(comb_sem[-top:],4).astype(np.str))
        
        
        model_words_acc['{} {}'.format(key,outcomes[0])] = np.flip(ft_sorted[-top:])
        model_words_acc['{} {}'.format(key,outcomes[1])] = np.flip(model_res_acc)
        model_words_acc['{} {}'.format(key,outcomes[2])] = np.flip(model_res_rej)
        
        fig = plt.figure(figsize=(15,8))
        ax = plt.axes()
        index = np.arange(top_plot)
        bar_width = 0.35
        opacity = 0.4
        error_config = {'ecolor': '0.3'}
        
        rects1 = ax.bar(index, avg_acc_sorted[-top_plot:], bar_width,
                        alpha=opacity, color='g',
                        yerr=sem_acc_sorted[-top_plot:], error_kw=error_config,
                        label='Accepted')
        
        rects2 = ax.bar(index + bar_width, avg_rej_sorted[-top_plot:], bar_width,
                        alpha=opacity, color='r',
                        yerr=sem_rej_sorted[-top_plot:], error_kw=error_config,
                        label='Rejected')
        ax.set_xticklabels(ft_sorted[-top_plot:],fontsize=17)
        ax.set_xticks(index + bar_width / 2)
        plt.xticks(rotation=45)
        plt.yticks(fontsize=17)
        plt.ylabel('Average tf-idf score',fontsize=20)
        #plt.ylabel('Average count per document',fontsize=20)
        plt.xlabel('Word stem',fontsize=20)
        plt.title(key)
        plt.savefig('{}/{}_{}_acc.svg'.format(output_path,key,field))
        plt.savefig('{}/{}_{}_acc.png'.format(output_path,key,field))
        plt.close()
        
        #REJECTED
        model_avg_acc = np.char.add(np.round(avg_acc_sorted[:top],4).astype(np.str),' ± ')
        model_sem_acc = np.round(sem_acc_sorted[:top],4).astype(np.str)
        model_res_acc = np.core.defchararray.add(model_avg_acc, model_sem_acc)
        model_avg_rej = np.char.add(np.round(avg_rej_sorted[:top],4).astype(np.str),' ± ')
        model_sem_rej = np.round(sem_rej_sorted[:top],4).astype(np.str)
        model_res_rej = np.core.defchararray.add(model_avg_rej, model_sem_rej)
        importance= np.char.add(np.round(diff[:top],4).astype(np.str),' ± ')
        importance = np.core.defchararray.add(importance, np.round(comb_sem[:top],4).astype(np.str))
        
        model_words_rej['{} {}'.format(key,outcomes[0])] = ft_sorted[:top]
        model_words_rej['{} {}'.format(key,outcomes[1])] = model_res_acc
        model_words_rej['{} {}'.format(key,outcomes[2])] = model_res_rej
        
        fig = plt.figure(figsize=(15,8))
        ax = plt.axes()
    
        index = np.arange(top_plot)
        bar_width = 0.35
        
        opacity = 0.4
        error_config = {'ecolor': '0.3'}
        
        rects1 = ax.bar(index, np.flip(avg_acc_sorted[:top_plot]), bar_width,
                        alpha=opacity, color='g',
                        yerr=np.flip(sem_acc_sorted[:top_plot]), error_kw=error_config,
                        label='Accepted')
        
        rects2 = ax.bar(index + bar_width, np.flip(avg_rej_sorted[:top_plot]), bar_width,
                        alpha=opacity, color='r',
                        yerr=np.flip(sem_rej_sorted[:top_plot]), error_kw=error_config,
                        label='Rejected')
        ax.set_xticklabels(np.flip(ft_sorted[:top_plot]),fontsize=17)
        ax.set_xticks(index + bar_width / 2)
        plt.xticks(rotation=45)
        plt.yticks(fontsize=17)
        plt.ylabel('Average tf-idf score',fontsize=20)
        plt.xlabel('Word stem',fontsize=20)
        plt.title(key)
        plt.savefig('{}/{}_{}_rej.svg'.format(output_path,key,field))
        plt.savefig('{}/{}_{}_rej.png'.format(output_path,key,field))
        plt.close()
        
    model_words_acc.to_csv('{}/{}_words_acc.csv'.format(output_path,field))  
    model_words_rej.to_csv('{}/{}_words_rej.csv'.format(output_path,field))      
