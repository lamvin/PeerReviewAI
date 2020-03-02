# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:32:17 2020

@author: philv
"""
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from statsmodels.formula.api import ols
import os

path = "C:/Users/philv/Documents/Projects/PeerRead/"
with open('{}/data/processed/df_data.csv'.format(path),'r',encoding='utf-8') as f:
    papers = pd.read_csv(f,sep='\t',index_col=0)

output_path = path + 'results/Semantic/'
if not os.path.exists(output_path):
    os.makedirs(output_path)    
                
platforms = ['arxiv.cs.ai_2007-2017','arxiv.cs.cl_2007-2017','arxiv.cs.lg_2007-2017','iclr_2017','all']
labels = ['AI','CL','LG','ICLR','All']
fields = ['title','abstract','introduction']
width =15
height = 8
effect_size = []
stats_plat = {}
for field in fields:
    results = np.zeros((3,3,5))
    text_sim = pd.read_csv('{}/data/processed/TextScores{}_stem.csv'.format(path,field),sep='\t',encoding='utf-8')
    for i in range(len(platforms)):
        platform = platforms[i]
        if platform != 'all':
            artIDs = papers.loc[papers['platform']==platform,'artID']
        else:
            artIDs = papers['artID']
        sim_plat = text_sim.loc[(text_sim['art1'].isin(artIDs) & text_sim['art2'].isin(artIDs))]
        results[:,:,i] = sim_plat.groupby('comb_type').agg({'sim':['mean','sem','std']})
        if platform == 'all':
            stats_plat[field] = ols('sim ~ C(comb_type)', data=sim_plat).fit()

    
    ind = np.arange(len(labels))
    width_bar = 0.25
    fig, ax = plt.subplots(figsize=(width,height))
    rects1 = ax.bar(ind, results[1,0,:], width_bar, color=(127/255,192/255,127/255), yerr=results[1,1,:])
    rects2 = ax.bar(ind+width_bar, results[0,0,:], width_bar,color=(255/255,127/255,127/255), yerr=results[0,1,:])
    rects3 = ax.bar(ind+width_bar*2, results[2,0,:], width_bar, color=(192/255,192/255,192/255), yerr=results[2,1,:])
    ax.set_title(labels,fontsize=35)
    ax.set_xticks(ind + width_bar)
    ax.set_xticklabels(labels)
    ax.legend((rects1[0], rects2[0],rects3[0]), ('Accepted','Rejected','Mixed'),loc='lower right',fontsize=25)
    ax.set_ylabel(field,fontsize=25)
    ax.set_xticklabels(labels)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', which='major', labelsize=30)

    plt.title('')
    plt.savefig(output_path+'PR_{}.svg'.format(field))
    plt.savefig(output_path+'PR_{}.png'.format(field))
    plt.close()
    
    print(stats_plat[field].summary())
