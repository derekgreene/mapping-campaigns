import logging as log
import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import cross_validate, GridSearchCV
# BERT-related imports
import torch
import transformers as ppb 
from tqdm import tqdm
from evaluation import scoring_measures, grid_measure
# sentence BERT imports
from sentence_transformers import SentenceTransformer, util

# --------------------------------------------------------------

# Pre-processing settings
reduce_len = True
strip_handles=False
min_term_length=2
max_term_length=20

# --------------------------------------------------------------

def clean_text(s):
    tweet_tokenizer = TweetTokenizer(strip_handles=strip_handles, reduce_len=reduce_len)
    tokens = []
    for tok in tweet_tokenizer.tokenize(s):
        tok = tok.lower().replace("'","")
        if len(tok) == 0:
            continue
        if not (tok[0].isalpha() or tok[0] == '#' or tok[0] == '@'):
            continue
        if len(tok) < min_term_length or len(tok) > max_term_length:
            continue
        tokens.append(tok)
    return " ".join(tokens)

def apply_bert(df, use_distilBERT=True):
	if use_distilBERT:
		log.info("Using use_distilBERT")
		model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
	else:
		log.info("Using full BERT")
		model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')	
	# load pretrained model/tokenizer
	tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
	model = model_class.from_pretrained(pretrained_weights)	    
	# apply tokenization
	tokenized = df["text"].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
	# apply padding. we need to pad all lists to the same size, so we can represent the input as one 2-d array
	max_len = 0
	for i in tokenized.values:
	    if len(i) > max_len:
	        max_len = len(i)
	padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
	# create another variable to tell it to ignore (mask) the padding we've added when processing its input
	attention_mask = np.where(padded != 0, 1, 0)
	# apply the model
	log.info("Applying model ...")
	input_ids = torch.tensor(padded)  
	attention_mask = torch.tensor(attention_mask)
	with torch.no_grad():
	    last_hidden_states = model(input_ids, attention_mask=attention_mask)
	# generate the features for the documents
	features = last_hidden_states[0][:,0,:].numpy()
	return features

def apply_sentence_bert(df, model_id):
	log.info("Applying model %s..." % model_id)
	model = SentenceTransformer(model_id)
	features = model.encode(df["text"], convert_to_tensor=True)
	log.info("Generated embedding of size: %s" % str(features.shape))
	return features

# --------------------------------------------------------------

def compile_cv_results(cv_scores):
    results = {}
    for key in cv_scores:
        if key.startswith("test_"):
            results[key.replace("test_","")] = cv_scores[key].mean()
    return results    

def apply_cv(features, labels, clf, cv_folds):
    cv_scores = cross_validate(clf, features, labels, cv=cv_folds, scoring=scoring_measures)
    return compile_cv_results(cv_scores)

def apply_grid_cv(features, labels, clf, cv_folds, param_grid):
    grid = GridSearchCV(clf, cv=cv_folds, scoring=grid_measure, param_grid=param_grid)   
    grid.fit(features, labels)
    log.info("Grid CV best score: %.4f, %s" % (grid.best_score_, grid.best_params_))
    # perform classification with the best parameters
    log.info("Performing final classification ...")
    return apply_cv(features, labels, grid.best_estimator_, cv_folds)



