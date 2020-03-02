# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:21:16 2020

@author: User1
"""
import json
import pandas as pd
import os
import abstract_cleanup as absclean
import readabilityFunctions as rf
import numpy as np
import csv
import string
from nltk import PorterStemmer
punct = string.punctuation
digits = string.digits
from fuzzywuzzy import fuzz
import itertools
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim import similarities
import numpy as np

def check_edu(emails):
    has_edu = False
    if len(emails) > 0:
        for email in emails:
            if email.lower().split('.')[-1] == 'edu':
                has_edu = True
    else:
        has_edu = None
    return has_edu

def get_data(path):
    platform_keys = {'arxiv.cs.lg_2007-2017':'a', 'arxiv.cs.cl_2007-2017':'b',
       'arxiv.cs.ai_2007-2017':'c', 'conll_2016':'d', 'acl_2017':'e', 'iclr_2017':'f'}
    data_types = ['test','dev','train']
    data_dirs = list(platform_keys.keys())
    articles = []
    for data_type in data_types:
        for data_dir in data_dirs:
            path_papers = '{}data/{}/{}/{}/'.format(path,data_dir,data_type,'parsed_pdfs')
            path_reviews = '{}data/{}/{}/{}/'.format(path,data_dir,data_type,'reviews')
            data_files = [f for f in os.listdir(path_papers)]
            for file_name in data_files:
                path_file = path_papers+file_name
                with open(path_file,'rb') as f:
                    data = json.loads(f.read())
                paper = pd.Series()
                artID = '.'.join(data['name'].split('.')[:-1])
                paper['artID'] = artID+platform_keys[data_dir]
                with open(path_reviews+artID+'.json') as f:
                    review = json.loads(f.read())
                metadata = data['metadata']
                paper['title'] = metadata['title'] if 'title' in metadata else None
                paper['abstract'] = metadata['abstractText'] if 'abstractText' in  metadata else None
                paper['year'] = metadata['year'] if 'year' in metadata else None
                
                paper['introduction'] = ' '.join([section['text'] for section in metadata['sections'] if section['heading'] and ("intro" in section['heading'].lower() or section['heading'].startswith('1') )]) if metadata['sections'] is not None else None
                paper['accepted'] = review['accepted'] if 'accepted' in review else None
                paper['keywords'] = metadata['keywords'] if 'keywords' in metadata else None
                paper['platform'] = data_dir
                paper['has_edu'] = check_edu(metadata['emails'])
                articles.append(paper)
    punct = string.punctuation.replace('.','')
    punct = punct.replace('!','')
    punct = punct.replace('?','')
    papers = pd.DataFrame(articles)
    papers['abstract'] = papers.apply(lambda x: absclean.cleanup_pretagger(x['abstract'],punct) if x['abstract'] is not None else None,axis=1)
    papers['introduction'] = papers.apply(lambda x: absclean.cleanup_pretagger(x['introduction'],punct) if x['introduction'] is not None else None,axis=1)
    papers['title'] = papers.apply(lambda x: absclean.cleanup_pretagger(x['title'],punct) if x['title'] is not None else None,axis=1)
    papers['accepted'] = papers.apply(lambda x: 1 if x['accepted'] else 0,axis=1)
    return papers
    

def get_readability(papers):
    colnames = ['artID','absFRE','absNDC','intFRE','intNDC','titleFRE','titleNDC']
    read = pd.DataFrame(columns=colnames)
    for i in range(papers.shape[0]):
        row = papers.iloc[i]
        read.loc[i,'artID'] = row['artID']
        abstract = row['abstract']        
        if isinstance(abstract,str):
            results = rf.analyze(abstract,doPreprocessing=0)
            if results:
                read.loc[i,'absFRE'] = results['fre']
                read.loc[i,'absNDC'] = results['NDC']
        introduction = row['introduction']
        if isinstance(introduction,str):
            results = rf.analyze(introduction,doPreprocessing=0)
            if results:
                read.loc[i,'intFRE'] = results['fre']
                read.loc[i,'intNDC'] = results['NDC']  
        title = row['title']
        if isinstance(title,str):
            results = rf.analyze(title,doPreprocessing=0)
            if results:
                read.loc[i,'titleFRE'] = results['fre']
                read.loc[i,'titleNDC'] = results['NDC']
    return read
       
def load_ngrams(file_name):
    if file_name == "AI":
        with open('../data/AI_jargon_processed.csv', 'r') as f:
            ng_list = f.readlines()
        ng_list = [x.rstrip().split(' ') for x in ng_list]
    else:
        content = []
        with open('../data/jargonListFinal.csv') as f:
            csv_reader = csv.reader(f,delimiter=',')
            for row in csv_reader:
                content.append(row)
        if file_name == 'general':
            ng_list = [[x[3]] for x in content[1:] if int(x[1])==1]
        else:
            ng_list = [[x[3]] for x in content[1:] if int(x[1])==0]
    return ng_list

def match_words_list(papers,section,sname):
    list_fnames = ['AI','general','specific']
    df = pd.DataFrame()
    df['artID'] = papers['artID']
    for i in range(len(papers)):
        artID = papers.iloc[i]['artID']
        text = papers[papers['artID']==artID][section].values[0]
        if isinstance(text, str):
            text = text.strip().lower().replace('.','').split(' ')
            matched_ng = 0
            len_text = len(text)
            for list_i in range(len(list_fnames)):
                jargon_type = list_fnames[list_i]
                ng_list = load_ngrams(jargon_type) 
                for ngram in ng_list:
                    ng_len = len(ngram)
                    for k in range(len_text):
                        if ngram == text[k:k+ng_len]:
                            matched_ng+=ng_len
                df.loc[df['artID']==artID,sname+jargon_type] = matched_ng/len_text        
    return df

def load_var():
    variables=pd.read_csv('../data/variables.csv',sep='\t',encoding='utf-8')
    variables=variables.set_index(['word'])
    return variables

def get_linguistic(papers):
    variables = load_var()
    fields = ['title','abstract','introduction']
    linguistic_data = pd.DataFrame(columns=['artID'])
    linguistic_data['artID'] = papers['artID']

    for field in fields:
        papers['{}_nb_tokens'.format(field)] = papers.apply(lambda x: len(x[field].split(' ')) if not pd.isnull(x[field]) else np.nan,axis=1)
        papers['{}_nb_types'.format(field)]= papers.apply(lambda x: len(set(x[field].split(' '))) if not pd.isnull(x[field]) else np.nan,axis=1)
        papers['{}_TTR'.format(field)] = papers['{}_nb_types'.format(field)]/papers['{}_nb_tokens'.format(field)]
        linguistic_data = pd.merge(linguistic_data,papers[['artID','{}_nb_tokens'.format(field),
                                                           '{}_nb_types'.format(field),'{}_TTR'.format(field)]],how='left',on='artID') 
    linguistic_data.set_index('artID',inplace=True)
    for var in variables.columns:
        var_dict = {key:variables.iloc[x][var] for x,key in enumerate(variables.index)}
        for field in fields:
            print('{} {}'.format(field, var))
            for row_i in range(len(papers)):
                row = papers.iloc[row_i]
                text = row[field]
                if not pd.isnull(text):
                    text= text.translate(str.maketrans('', '', punct))
                    text_tokens = text.split(' ')
                    text_set = set()
                    scores = []
                    scores_set = []
                    for tok in text_tokens:
                        if tok in var_dict.keys():
                            score = var_dict[tok]
                            scores.append(score)
                            if tok not in text_set:
                                text_set.add(tok)
                                scores_set.append(score)
                    scores_avg = np.nanmean(np.array(scores))
                    scores_set_avg = np.nanmean(np.array(scores_set))
                else: 
                    scores_avg = np.nan
                    scores_set_avg = np.nan
                
                linguistic_data.loc[row['artID'],'{}_{}'.format(field,var)] = scores_avg
                linguistic_data.loc[row['artID'],'{}_{}_set'.format(field,var)] = scores_set_avg
                
    return(linguistic_data)

def get_word_stem(text,stemmer):
    if not pd.isna(text):
        text = text.translate(str.maketrans('', '', punct))
        stemmed = ' '.join([stemmer.stem(x) for x in text.split(' ')])
    else:
        stemmed = np.nan
    return stemmed

def rm_words(text):
    if isinstance(text,str) and len(text) > 0:
        text = text.translate(str.maketrans('', '', digits))
        text = ' '.join([x for x in text.split(' ') if len(x) > 2])
    else:
        text = np.nan
    return(text)

def convert_to_stem(papers):
    stemmer = PorterStemmer()
    sections = ['title','abstract','introduction']
    for section in sections:
        papers[section] = papers.apply(lambda row: get_word_stem(row[section],stemmer),axis=1)
        papers[section] = papers.apply(lambda row: rm_words(row[section]),axis=1)
    return papers

def get_references(path):
    platform_keys = {'arxiv.cs.lg_2007-2017':'a', 'arxiv.cs.cl_2007-2017':'b',
       'arxiv.cs.ai_2007-2017':'c', 'conll_2016':'d', 'acl_2017':'e', 'iclr_2017':'f'}
    data_types = ['test','dev','train']
    data_dirs = list(platform_keys.keys())
    citations = []
    references = {}
    for data_type in data_types:
        for data_dir in  data_dirs:
            path_papers = '{}/data/{}/{}/{}/'.format(path,data_dir,data_type,'parsed_pdfs')
            data_files = [f for f in os.listdir(path_papers)]
            for file_name in data_files:
                path_file = path_papers+file_name
                with open(path_file,'rb') as f:
                    data = json.loads(f.read())
                citations += list(data['metadata']['references'])
                references['.'.join(data['name'].split('.')[:-1])+platform_keys[data_dir]] = list(data['metadata']['references'])
    return references

def group_citations(citations):
    citID = 0
    grped_citations = {}
    
    while len(citations) > 0:
        print('{} citations remaining'.format(len(citations)))
        base_citation = citations[0]
        del citations[0]
        base_year = base_citation['year']
        base_author = base_citation['author']
        base_author_str = ' '.join(base_author)
        base_nb_author = len(base_author)
        base_title = base_citation['title']
        grped_citations[citID] = [base_citation]
        del_cit = []
        for cit_i in range(len(citations)):
            citation = citations[cit_i]
            year = citation['year']
            if year != base_year:
                continue
            author = citation['author']
            if len(author) != base_nb_author:
                continue
            author_str = ' '.join(author) if isinstance(author,list) else author
            match_author = fuzz.token_set_ratio(author_str, base_author_str)
            if match_author < 70:
                continue
            title = citation['title']
            match_title = fuzz.token_set_ratio(title, base_title)

            if match_title < 70:
                continue
            grped_citations[citID].append(citation)
            del_cit.append(cit_i)
        for c in sorted(del_cit,reverse=True):
            del citations[c] 
        citID += 1
    return grped_citations

def match_citation(grouped_citations,references):
    citIDs = []
    for ref in references:
        matched_ref = False
        for citID in grouped_citations:
            for version_cit in grouped_citations[citID]:
                if version_cit == ref:
                    citIDs.append(citID)
                    matched_ref = True
                    break
            if matched_ref:
                break
    return citIDs
    
def tag_citations_papers(papers,grouped_citations,references):
    references_ID = {}
    ref_keys = list(references.keys())
    nb_refs = len(references)
    for i in range(nb_refs):
        artID = ref_keys[i]
        if i % 10 == 0:
            print('Matching paper {}/{}'.format(i,nb_refs))
        references_ID[artID] = match_citation(grouped_citations,references[artID])
    return references_ID

def get_comb_type(status1,status2):
    if status1 == 1 and status2 == 1:
        comp = 1
    elif status1 == 0 and status2 ==0:
        comp = 0
    else:
        comp = 2
    return comp

def setTextScores(data,field):   
    data = data.dropna(subset=[field])
    status_data = {x[1]['artID']:x[1]['accepted'] for x in data.iterrows()}
    dct = Dictionary([[line for line in doc.split()] for doc in data[field]])
    corpus = [dct.doc2bow(line.split(' ')) for line in data[field]]
    model = TfidfModel(corpus)
    tfidf = model[corpus]
    num_features = len(dct)
    index_cols = list(data['artID'])
    index = similarities.MatrixSimilarity(tfidf,num_features=num_features)
    sim = index[tfidf]
    pairs_id = np.tril_indices(sim.shape[0],-1)
    pairs_sim = []
    for j in range(len(pairs_id[0])):
        art1 = pairs_id[0][j]
        art2 = pairs_id[1][j]
        comb_type = get_comb_type(status_data[index_cols[art1]],status_data[index_cols[art2]])
        pairs_sim.append([index_cols[art1],index_cols[art2],sim[art1,art2],comb_type])
    return pd.DataFrame(pairs_sim,columns=['art1','art2','sim','comb_type'])
    
def setCitationScores(papers,references_ID):
    status_data = {x[1]['artID']:x[1]['accepted'] for x in papers.iterrows()}
    idx = list(references_ID.keys())
    print('Generating index')
    citationData=pd.DataFrame([combo for combo in itertools.combinations(idx,2)],columns=['art1','art2'])
    #citationData=citationData.set_index(['art1','art2'])
    print('Computing Citation Intersection Sets')
    citationData['CitationInt']= citationData.apply(
                lambda x: len(set(references_ID[x['art1']]).intersection(references_ID[x['art2']])),axis=1)
    print('Computing Citation Union Sets')
    citationData['CitationUnion']= citationData.apply(
                lambda x: len(set(references_ID[x['art1']]+references_ID[x['art2']])),axis=1)
    print('Computing Citation Similarity Scores')
    citationData['CitationSim']=citationData['CitationInt']/citationData['CitationUnion']
    #citationData=citationData[citationData['CitationInt']>=1]
    
    citationData=citationData.drop('CitationUnion',axis=1)
    citationData['comb_type'] = citationData.apply(lambda x: get_comb_type(status_data[x['art1']],status_data[x['art2']]),axis=1)
    
    return citationData
    
