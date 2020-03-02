# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 10:44:48 2020

@author: User1
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import os

def load_jargon(path,section):
    with open('{}/data/processed/jargon_{}.csv'.format(path,section),'r') as f:
        jargon = pd.read_csv(f,sep='\t',index_col=0)
    return jargon

#Path to root PeerRead directory
path = "C:/Users/philv/Documents/Projects/PeerRead/"
output_path = path + 'results/readability/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
with open('{}/data/processed/df_data.csv'.format(path),'r',encoding='utf-8') as f:
    papers = pd.read_csv(f,sep='\t',index_col=0)


with open('{}/data/processed/readability_data.csv'.format(path),'r') as f:
    read = pd.read_csv(f,sep='\t',index_col=0)
    

papers = pd.merge(papers, load_jargon(path,'title'), on='artID')     
papers = pd.merge(papers, load_jargon(path,'abs'), on='artID')   
papers = pd.merge(papers, load_jargon(path,'int'), on='artID')  
papers = pd.merge(papers, read, on='artID')    
papers.loc[pd.isnull(papers['has_edu']),'has_edu'] = False
papers = papers[~pd.isnull(papers['accepted'])]    

platforms = ['arxiv.cs.ai_2007-2017','arxiv.cs.cl_2007-2017','arxiv.cs.lg_2007-2017','iclr_2017']
acronyms = ['AI','CL','LG','ICLR']

papers = papers[papers["platform"].isin(platforms)]
papers['platform'] = papers.apply(lambda x: acronyms[platforms.index(x['platform'])],axis=1)
       

parts = ['title','abs','int']
variables = ['FRE','NDC','AI','general','specific']
var_title = ['FRE','NDC','AI','General','Specific']
width = 15
height = 8
width_bar = 0.35
acronyms.append('All')
results = pd.DataFrame(columns=['part','var','p','diff','avg_rej','std_rej','count_rej',
                              'avg_acc','std_acc','count_acc','eff_size','has_edu'])

for i in range(2):
    if i == 0:
        data = papers.loc[papers['has_edu']]
        has_edu = '_has_edu'
    else:
        data = papers.loc[~papers['has_edu']]
        has_edu = ''
        
    scores = data.groupby(['accepted','platform']).agg(['mean','sem','std','count']).reset_index()
    all_platforms = data.groupby(['accepted']).agg(['mean','sem','std','count']).reset_index()
    all_platforms['platform'] = 'All'
    scores = scores.append(all_platforms).reset_index()
    scores['accepted'] = pd.to_numeric(scores['accepted'])
    cat1 = scores[scores['accepted']==0]
    cat2 = scores[scores['accepted']==1]
    colors = [[253,94,94],[102,226,106]]
    colors = [np.array(x)/255 for x in colors]
    for var_i in range(len(variables)):
        var = variables[var_i]
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
                mean[j] = cat2.loc[cat2[('platform','')]==plat,('{}{}'.format(part,var),'mean')]
                sem[j] = cat2.loc[cat2[('platform','')]==plat,('{}{}'.format(part,var),'sem')]
                j+=1
                mean[j] = cat1.loc[cat1[('platform','')]==plat,('{}{}'.format(part,var),'mean')]
                sem[j] = cat1.loc[cat1[('platform','')]==plat,('{}{}'.format(part,var),'sem')]
                j+=1
                
            
            mean_rej = cat1.loc[cat1[('platform','')]=='All',('{}{}'.format(part,var),'mean')].values[0]
            mean_acc = cat2.loc[cat2[('platform','')]=='All',('{}{}'.format(part,var),'mean')].values[0]
            std_rej = cat1.loc[cat1[('platform','')]=='All',('{}{}'.format(part,var),'std')].values[0]
            std_acc = cat2.loc[cat2[('platform','')]=='All',('{}{}'.format(part,var),'std')].values[0]
            count_rej = cat1.loc[cat1[('platform','')]=='All',('{}{}'.format(part,var),'count')].values[0]
            count_acc = cat2.loc[cat2[('platform','')]=='All',('{}{}'.format(part,var),'count')].values[0]
            
            grp1 = data.loc[data['accepted']==0,'{}{}'.format(part,var)].dropna()
            grp2 = data.loc[data['accepted']==1,'{}{}'.format(part,var)].dropna()
            ttest = stats.ttest_ind(grp1,grp2, equal_var=False)
            pooled_std = np.sqrt( ((count_rej-1)*std_rej**2 + (count_acc-1)*std_acc**2)/(count_rej+count_acc))
            cohens_d = np.abs((mean_rej-mean_acc)/pooled_std)
            results.loc[len(results)] = [part,var,ttest[1],mean_acc-mean_rej,mean_rej,std_rej,count_rej,
                       mean_acc,std_acc,count_acc,cohens_d,has_edu]

            
            bar_i += 1
       
        fig, ax = plt.subplots(figsize=(width,height))         
        nb_bars = len(ind)
        acc = np.arange(0,nb_bars,2)
        rej = np.arange(1,nb_bars,2)
        rects1 = ax.bar(ind[rej], mean[rej], width_bar, yerr=sem[rej], color=colors[0],label='Rejected')
        rects2 = ax.bar(ind[acc], mean[acc],  width_bar, yerr=sem[acc], color=colors[1],label='Accepted')
        ax.set_xticks(ind_labels)
        ax.set_xticklabels(labels)
        if var == 'General':
            pass
            #plt.ylim([0.3,0.5])
        elif var == 'NDC':
            plt.ylim([10,18])
        plt.title(var_title[var_i],size=30)
        plt.legend()
        plt.tick_params(labelsize=20)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(2.5)
        ax.get_legend().remove()
        plt.savefig('{}{}{}.png'.format(output_path,var,has_edu))
        plt.savefig('{}{}{}.svg'.format(output_path,var,has_edu))
        plt.close()