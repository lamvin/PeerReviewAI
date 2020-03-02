# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 07:30:59 2020

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

output_path = path + 'results/BC/'
if not os.path.exists(output_path):
    os.makedirs(output_path)    
    
citations = pd.read_csv(path+'data/processed/citation_scores.txt',sep='\t',encoding='utf-8')
citations = citations.loc[citations['CitationInt']>0]

            
platforms = ['arxiv.cs.ai_2007-2017','arxiv.cs.cl_2007-2017','arxiv.cs.lg_2007-2017','iclr_2017','all']
labels = ['AI','CL','LG','ICLR','All']
citations_type = ['CitationInt','CitationSim']
width =15
height = 8
effect_size = []
stats_plat = {}
for cit_type in citations_type:
    results = np.zeros((3,3,5))
    for i in range(len(platforms)):
        platform = platforms[i]
        if platform != 'all':
            artIDs = papers.loc[papers['platform']==platform,'artID']
        else:
            artIDs = papers['artID']
        cits_plat = citations.loc[(citations['art1'].isin(artIDs) & citations['art2'].isin(artIDs))]
        results[:,:,i] = cits_plat.groupby('comb_type').agg({cit_type:['mean','sem','std']})
        if platform == 'all':
            stats_plat[cit_type] = ols(cit_type + ' ~ C(comb_type)', data=cits_plat).fit()

    
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
    ax.set_ylabel(cit_type,fontsize=25)
    ax.set_xticklabels(labels)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', which='major', labelsize=30)

    plt.title('')
    plt.savefig(output_path+'PR_{}.svg'.format(cit_type))
    plt.savefig(output_path+'PR_{}.png'.format(cit_type))
    plt.close()
    
    print(stats_plat[cit_type].summary())




