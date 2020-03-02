# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:22:10 2020

@author: philv
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import os

path = "C:/Users/philv/Documents/Projects/PeerRead/"
output_path = path + 'results/Linguistic/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

with open('{}/data/processed/df_data.csv'.format(path),'r',encoding='utf-8') as f:
    papers = pd.read_csv(f,sep='\t',index_col=0)
    
with open('{}/data/processed/linguistic_data.csv'.format(path),'r',encoding='utf-8') as f:
    linguistic_data = pd.read_csv(f,sep='\t')
    
linguistic_data = linguistic_data.merge(papers[['artID','accepted','platform']],on='artID')


platforms = ['arxiv.cs.ai_2007-2017','arxiv.cs.cl_2007-2017','arxiv.cs.lg_2007-2017','iclr_2017']
acronyms = ['AI','CL','LG','ICLR']

linguistic_data = linguistic_data[linguistic_data["platform"].isin(platforms)]
linguistic_data['platform'] = linguistic_data.apply(lambda x: acronyms[platforms.index(x['platform'])],axis=1)
       

variables = ['frequency','aoa','concreteness','frequency_set','aoa_set','concreteness_set','nb_tokens','nb_types','TTR']  
parts = ['title','abstract','introduction']

width = 15
height = 8
width_bar = 0.35
acronyms.append('All')
results = pd.DataFrame(columns=['part','var','p','diff','avg_rej','std_rej','count_rej',
                              'avg_acc','std_acc','count_acc','eff_size'])


scores = linguistic_data.groupby(['accepted','platform']).agg(['mean','sem','std','count']).reset_index()
all_platforms = linguistic_data.groupby(['accepted']).agg(['mean','sem','std','count']).reset_index()
all_platforms['platform'] = 'All'
scores = scores.append(all_platforms).reset_index()
scores['accepted'] = pd.to_numeric(scores['accepted'])
cat1 = scores[scores['accepted']==0]
cat2 = scores[scores['accepted']==1]
colors = [[253,94,94],[102,226,106]]
colors = [np.array(x)/255 for x in colors]
psycho_vars = ['frequency','aoa','concreteness','frequency_set','aoa_set','concreteness_set']
psycho_vars_title = ['Frequency tokens','AOA tokens','Concreteness tokens','Frequency types','AOA types','Concreteness types']
lex_vars = ['nb_tokens','nb_types','TTR']
lex_vars_title = ['# tokens','# types','TTR']
for var_i in range(len(psycho_vars)):
    var = psycho_vars[var_i]
    sem = np.zeros(len(acronyms)*len(parts)*2)
    mean = np.zeros(len(acronyms)*len(parts)*2)
    ind = np.zeros(len(acronyms)*len(parts)*2)
    ind_labels = []
    labels = []
    j = 0
    bar_i = 0
    for part_i in range(len(parts)):
        part = parts[part_i]
        for plat_i in range(len(acronyms)):
            plat = acronyms[plat_i]
            ind[j:j+2] = [bar_i,bar_i+width_bar]
            ind_labels.append(ind[j]+width_bar/2)
            labels.append(plat)
            bar_i += 2*width_bar
            mean[j] = cat2.loc[cat2[('platform','')]==plat,('{}_{}'.format(part,var),'mean')]
            sem[j] = cat2.loc[cat2[('platform','')]==plat,('{}_{}'.format(part,var),'sem')]
            j+=1
            mean[j] = cat1.loc[cat1[('platform','')]==plat,('{}_{}'.format(part,var),'mean')]
            sem[j] = cat1.loc[cat1[('platform','')]==plat,('{}_{}'.format(part,var),'sem')]
            j+=1
            
        
        mean_rej = cat1.loc[cat1[('platform','')]=='All',('{}_{}'.format(part,var),'mean')].values[0]
        mean_acc = cat2.loc[cat2[('platform','')]=='All',('{}_{}'.format(part,var),'mean')].values[0]
        std_rej = cat1.loc[cat1[('platform','')]=='All',('{}_{}'.format(part,var),'std')].values[0]
        std_acc = cat2.loc[cat2[('platform','')]=='All',('{}_{}'.format(part,var),'std')].values[0]
        count_rej = cat1.loc[cat1[('platform','')]=='All',('{}_{}'.format(part,var),'count')].values[0]
        count_acc = cat2.loc[cat2[('platform','')]=='All',('{}_{}'.format(part,var),'count')].values[0]
        
        grp1 = linguistic_data.loc[linguistic_data['accepted']==0,'{}_{}'.format(part,var)].dropna()
        grp2 = linguistic_data.loc[linguistic_data['accepted']==1,'{}_{}'.format(part,var)].dropna()
        ttest = stats.ttest_ind(grp1,grp2, equal_var=False)
        pooled_std = np.sqrt( ((count_rej-1)*std_rej**2 + (count_acc-1)*std_acc**2)/(count_rej+count_acc))
        cohens_d = np.abs((mean_rej-mean_acc)/pooled_std)
        results.loc[len(results)] = [part,var,ttest[1],mean_acc-mean_rej,mean_rej,std_rej,count_rej,
                   mean_acc,std_acc,count_acc,cohens_d]

        
        bar_i += 1
   
    fig, ax = plt.subplots(figsize=(width,height))         
    nb_bars = len(ind)
    acc = np.arange(0,nb_bars,2)
    rej = np.arange(1,nb_bars,2)
    rects1 = ax.bar(ind[rej], mean[rej], width_bar, yerr=sem[rej], color=colors[0],label='Rejected')
    rects2 = ax.bar(ind[acc], mean[acc],  width_bar, yerr=sem[acc], color=colors[1],label='Accepted')
    ax.set_xticks(ind_labels)
    ax.set_xticklabels(labels)
    if var == 'aoa' or var == 'aoa_set':
        plt.ylim([6,9.5])
    elif var == 'concreteness' or var == 'concreteness_set':
        plt.ylim([2.3,2.7])
    elif var == 'frequency' or var == 'frequency_set':
        plt.ylim([2.5,4])
    plt.title(psycho_vars_title[var_i],size=30)
    plt.legend()
    plt.tick_params(labelsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(2.5)
    ax.get_legend().remove()
    plt.savefig('{}{}.png'.format(output_path,var))
    plt.savefig('{}{}.svg'.format(output_path,var))
    plt.close()
    
#Lexical    
for var_i in range(len(lex_vars)):
    var = lex_vars[var_i]

    for part_i in range(len(parts)):
        part = parts[part_i]
        sem = np.zeros(len(acronyms)*2)
        mean = np.zeros(len(acronyms)*2)
        ind = np.zeros(len(acronyms)*2)
        ind_labels = []
        labels = []
        j = 0
        bar_i = 0
        for plat_i in range(len(acronyms)):
            plat = acronyms[plat_i]
            ind[j:j+2] = [bar_i,bar_i+width_bar]
            ind_labels.append(ind[j]+width_bar/2)
            labels.append(plat)
            bar_i += 2*width_bar
            mean[j] = cat2.loc[cat2[('platform','')]==plat,('{}_{}'.format(part,var),'mean')]
            sem[j] = cat2.loc[cat2[('platform','')]==plat,('{}_{}'.format(part,var),'sem')]
            j+=1
            mean[j] = cat1.loc[cat1[('platform','')]==plat,('{}_{}'.format(part,var),'mean')]
            sem[j] = cat1.loc[cat1[('platform','')]==plat,('{}_{}'.format(part,var),'sem')]
            j+=1
            
        
        mean_rej = cat1.loc[cat1[('platform','')]=='All',('{}_{}'.format(part,var),'mean')].values[0]
        mean_acc = cat2.loc[cat2[('platform','')]=='All',('{}_{}'.format(part,var),'mean')].values[0]
        std_rej = cat1.loc[cat1[('platform','')]=='All',('{}_{}'.format(part,var),'std')].values[0]
        std_acc = cat2.loc[cat2[('platform','')]=='All',('{}_{}'.format(part,var),'std')].values[0]
        count_rej = cat1.loc[cat1[('platform','')]=='All',('{}_{}'.format(part,var),'count')].values[0]
        count_acc = cat2.loc[cat2[('platform','')]=='All',('{}_{}'.format(part,var),'count')].values[0]
        
        grp1 = linguistic_data.loc[linguistic_data['accepted']==0,'{}_{}'.format(part,var)].dropna()
        grp2 = linguistic_data.loc[linguistic_data['accepted']==1,'{}_{}'.format(part,var)].dropna()
        ttest = stats.ttest_ind(grp1,grp2, equal_var=False)
        pooled_std = np.sqrt( ((count_rej-1)*std_rej**2 + (count_acc-1)*std_acc**2)/(count_rej+count_acc))
        cohens_d = np.abs((mean_rej-mean_acc)/pooled_std)
        results.loc[len(results)] = [part,var,ttest[1],mean_acc-mean_rej,mean_rej,std_rej,count_rej,
                   mean_acc,std_acc,count_acc,cohens_d]
   
        fig, ax = plt.subplots(figsize=(width/3,height))         
        nb_bars = len(ind)
        acc = np.arange(0,nb_bars,2)
        rej = np.arange(1,nb_bars,2)
        rects1 = ax.bar(ind[rej], mean[rej], width_bar, yerr=sem[rej], color=colors[0],label='Rejected')
        rects2 = ax.bar(ind[acc], mean[acc],  width_bar, yerr=sem[acc], color=colors[1],label='Accepted')
        ax.set_xticks(ind_labels)
        ax.set_xticklabels(labels)
        if var == 'TTR':
            if part == 'title':
                plt.ylim([0.95,1.01])
                plt.yticks([0.95,0.975,1])
            elif part == 'abstract':
                plt.ylim([0.6,0.72])
                plt.yticks([0.6,0.65,0.7])
            elif part == 'introduction':
                plt.ylim([0.4,0.55])
                plt.yticks([0.4,0.45,0.5])
        
        plt.title(lex_vars_title[var_i],size=30)
        plt.legend()
        plt.tick_params(labelsize=20)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(2.5)
        ax.get_legend().remove()
        plt.savefig('{}{}{}.png'.format(output_path,var,part))
        plt.savefig('{}{}{}.svg'.format(output_path,var,part))
        plt.close()