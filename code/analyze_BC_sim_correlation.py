# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 09:25:57 2020

@author: philv
"""
import pandas as pd
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
import os


def get_corr():
    for field in fields:
        for platform in platforms:
            if platform == 'all':
                data = papers.copy()
                scores = citations.copy()
            else:
                data = papers[papers['platform']==platform]
                data_ids = data['artID']
                scores = citations.loc[(citations['art1'].isin(data_ids) & citations['art2'].isin(data_ids))]
            scores = scores.merge(text_sim_dict[field],how='left',left_on=['art1','art2'],right_on=['art2','art1'])
            for type_cit in types_cit:
                x, y = scores['sim'].values, scores[type_cit].values
                nas = np.logical_or(np.isnan(x), np.isnan(y))
                corr = stats.pearsonr(x[~nas], y[~nas])
                R2Scores.loc[len(R2Scores)] = [platform,field,type_cit,corr[1],corr[0]**2,corr[0]]
    return R2Scores        

path = "C:/Users/philv/Documents/Projects/PeerRead/"
output_path = path + 'results/corr_BC_sim/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
with open('{}/data/processed/df_data.csv'.format(path),'r',encoding='utf-8') as f:
    papers = pd.read_csv(f,sep='\t',index_col=0)

platforms_keys = {'arxiv.cs.ai_2007-2017':'AI','arxiv.cs.cl_2007-2017':'CL',
                  'arxiv.cs.lg_2007-2017':'LG','iclr_2017':'ICLR',
                  'conll_2016':'CoNLL','acl_2017':'ACL','all':'All'}
platforms = list(platforms_keys.keys())
fields = ['title','abstract','introduction']
types_cit = ['CitationSim','CitationInt']

citations = pd.read_csv(path+'data/processed/citation_scores.txt',sep='\t',encoding='utf-8')
citations = citations.loc[citations['CitationInt']>0]
citations['art1'] = citations['art1'].astype(str)
citations['art2'] = citations['art2'].astype(str)  

text_sim_dict = {}
for field in fields:
    text_sim_dict[field] = pd.read_csv('{}/data/processed/TextScores{}_stem.csv'.format(path,field),sep='\t',encoding='utf-8')
    text_sim_dict[field]['art1'] = text_sim_dict[field]['art1'].astype(str)
    text_sim_dict[field]['art2'] = text_sim_dict[field]['art2'].astype(str)  
    
  
R2Scores = pd.DataFrame(columns=['platform','field','type_cit','p','r2','r'])
R2Scores = get_corr()
width = 12
height = 8
for platform in platforms:
    df = R2Scores.loc[R2Scores['platform']==platform]
    sim = np.array([df.loc[((df['field']==fields[0]) & (df['type_cit']==types_cit[0])),'r'].values[0],
                  df.loc[((df['field']==fields[1]) & (df['type_cit']==types_cit[0])),'r'].values[0],
                  df.loc[((df['field']==fields[2]) & (df['type_cit']==types_cit[0])),'r'].values[0]])
    inter = np.array([df.loc[((df['field']==fields[0]) & (df['type_cit']==types_cit[1])),'r'].values[0],
                  df.loc[((df['field']==fields[1]) & (df['type_cit']==types_cit[1])),'r'].values[0],
                  df.loc[((df['field']==fields[2]) & (df['type_cit']==types_cit[1])),'r'].values[0]])
    fig = plt.figure(figsize=(width,height))
    ax = plt.axes()
    labels = ['Title','Abstract','Introduction']
    ind = np.arange(len(sim))
    width_bar = 0.25
    rects1 = ax.bar(ind, sim, width_bar, color='r',alpha=0.5)
    rects2 = ax.bar(ind+width_bar, inter, width_bar, color='b',alpha=0.5)
    ax.set_ylabel('Pearson r',fontsize=30)
    ax.set_title('{}'.format(platforms_keys[platform]),fontsize=35)
    ax.set_xticks(ind + width_bar / 2)
    ax.set_xticklabels(labels)
    ax.legend((rects1[0], rects2[0]), ('Jaccard', 'Intersection'),loc='lower right',fontsize=25)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', which='major', labelsize=30)
    major_ticks = np.arange(0, 0.7, 0.2)
    minor_ticks = np.arange(0, 0.7, 0.1)

    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    

    ax.grid(which='both',axis='x')
    ax.grid(which='minor', alpha=0.5)
    ax.grid(which='major', alpha=0.5)
    plt.ylim([0,0.7])
    ax.xaxis.grid(False) # vertical lines
    [j[1].set_linewidth(0.5) for j in list(ax.spines.items())]
    plt.savefig(output_path+'platforms{}_stem.png'.format(platform))
    plt.savefig(output_path+'platforms{}_stem.svg'.format(platform))
    plt.close()
