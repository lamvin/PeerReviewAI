# PeerReviewAI

## This repository
This code can be used to reproduce the results of [this manuscript](https://arxiv.org/abs/1911.02648). You first need to clone [PeerRead](https://github.com/allenai/PeerRead) repository to obtain the data used in this analysis.

## Using the scripts
The first script you'll need is the make_data.py, which will load the json files in the PeerRead repository, preprocess the text content and generate the data files necessary for the analysis in the manuscript. Running all the script might take 36 to 48 hours on a standard computer.

The preprocessing.py and abstract_cleanup.py contain functions used by make_data.py to preprocess the data.

### analyze_BC.py
This scripts will generate the data and figures showing the relationship between bibliographic coupling and peer review outcome.

### analyze_semantic.py
This generates the results related to the similairity of rejected and accepted manuscripts.

### analyze_BC_sim_correlation.py
This scripts analyzes the correlation between the bibliographic coupling of manuscripts and their semantic similairity as determined by their tf-idf representations.

### analyze_linguistic.py
This generates the results related to the psycholinguistic (age of acquisition, concreteness and frequency of words) and lexical (# tokens, # types and TTR) correlates of peer review.

### analyze_readability.py
This will generate the figure showing the relationship between the scientific jargon and readability of manuscripts.

### predict_PR_outcome.py
This script build a logistic regression to predict the peer review outcome of manuscripts based on their word content. 

### get_stem_predictors.py
This script will extract the most relevant stems to predict manuscript acceptance and rejection. 

