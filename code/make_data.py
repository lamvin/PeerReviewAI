# -*- coding: utf-8 -*-
import pandas as pd
import processing
import pickle
import os

def peerread_to_df(load = False):
    if load:
        with open('{}/data/processed/df_data.csv'.format(path),'r',encoding='utf-8') as f:
            papers = pd.read_csv(f,sep='\t',index_col=0)
    else:
        papers = processing.get_data(path)
        with open('{}/data/processed/df_data.csv'.format(path),'w', encoding='utf-8') as f:
            papers.to_csv(f,sep='\t')
    return papers

def text_to_stem(papers=None,load=False):
    if load:
        with open('{}/data/processed/df_data_stem.csv'.format(path),'r',encoding='utf-8') as f:
            papers_stem = pd.read_csv(f,sep='\t',index_col=0)
    else:
        papers_stem = processing.convert_to_stem(papers)
        with open('{}/data/processed/df_data_stem.csv'.format(path),'w', encoding='utf-8') as f:
            papers_stem.to_csv(f,sep='\t')
    return papers_stem

def get_similarity_tfidf(papers):
    fields = ['title','abstract','introduction']
    for field in fields:
        print('Calculating distance scores for {}.'.format(field))
        pairingData = processing.setTextScores(papers,field)
        pairingData.to_csv(path+'/data/processed/TextScores{}_stem.csv'.format(field),encoding='utf-8',sep='\t')

def get_references(load=False):
    if load:
        with open('{}data/processed/references.p'.format(path),'rb') as f:
            references = pickle.load(f)
    else:
        references = processing.get_references(path)
        with open(path+'/data/processed/references.p','wb') as f:
            pickle.dump(references,f)
    return references

def group_citations(references,load=False):
    if load:
        with open('{}data/processed/grouped_citations.p'.format(path),'rb') as f:
            grouped_citations = pickle.load(f)
    else:
        citations = []
        for ref in references:
            citations = citations + references[ref]
        grouped_citations = processing.group_citations(citations)
        with open(path+'/data/processed/grouped_citations.p','wb') as f:
            pickle.dump(grouped_citations,f)
    return grouped_citations

def get_refs_IDs(papers,grouped_citations,references,load=False):
    if load:
        with open('{}data/processed/references_ID.p'.format(path),'rb') as f:
            references_ID = pickle.load(f)
    else:
        references_ID = processing.tag_citations_papers(papers,grouped_citations,references)
        with open(path+'/data/processed/references_ID.p','wb') as f:
            pickle.dump(references_ID,f)
    return references_ID

def get_citations_scores(references_ID,load=False):
    if load:
        citation_data = pd.read_csv(path+'data/processed/citation_scores.txt',sep='\t',encoding='utf-8')
    else:
        citation_data = processing.setCitationScores(papers,references_ID)
        citation_data.to_csv(path+'data/processed/citation_scores.txt',sep='\t',encoding='utf-8')
    return citation_data

def get_readability(papers,load=False):
    if load:
        with open('{}/data/processed/readability_data.csv'.format(path),'r',encoding='utf-8') as f:
            read = pd.read_csv(f,sep='\t',index_col=0)
    else:
        read = processing.get_readability(papers)
        with open('{}/data/processed/readability_data.csv'.format(path),'w') as f:
            read.to_csv(f,sep='\t')
    return read

def create_jargon_data(papers):
    sections = {'title':'title','abstract':'abs','introduction':'int'}
    for section in sections.keys():   
        df = processing.match_words_list(papers,section,sections[section])
        with open('{}/data/processed/jargon_{}.csv'.format(path,sections[section]),'w') as f:
            df.to_csv(f,sep='\t')

def get_linguistic(papers,load=False):
    if load:
        with open('{}/data/processed/linguistic_data.csv'.format(path),'r',encoding='utf-8') as f:
            linguistic_data = pd.read_csv(f,sep='\t',index_col=0)
    else:
        linguistic_data = processing.get_linguistic(papers)
        with open('{}/data/processed/linguistic_data.csv'.format(path),'w') as f:
            linguistic_data.to_csv(f,sep='\t')
    return linguistic_data

#Path to root PeerRead directory
path = "C:/Users/philv/Documents/Projects/PeerRead/"
output_path = path + 'data/processed/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
'''
Preprocessing of the textual data
'''
#Upload and preprocess PeerRead data to a dataframe
papers = peerread_to_df(load=True)
#Get word stem
papers_stem = text_to_stem(papers,load=True)
#Get similarity tf-idf
get_similarity_tfidf(papers_stem)


'''
Get references, assign unique ID to alternative writing of the same citation,
replace references with this unique ID for every manuscript.
'''
#Get references from PeerRead data
references = get_references()
#Group citations and assign them unique IDs (~24 hours)
grouped_citations = group_citations(references)    
#Replace references with unique IDs 
references_ID = get_refs_IDs(papers,grouped_citations,references)
#Get coupling scores between articles
citation_data = get_citations_scores(references_ID)

'''
Get readability scores, jargons proportion and lexical scores for every 
manuscript
'''
#Get readability 
read = get_readability(papers)
#Create jargon
create_jargon_data(papers)
#Get psycholinguistic scores
linguistic_data = get_linguistic(papers)

