import logging as log
import numpy as np
from nltk.tokenize import TweetTokenizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from evaluation import scoring_measures, grid_measure

# --------------------------------------------------------------

# Pre-processing Settings
strip_handles=False
strip_urls=True
reduce_len=True
min_term_length=2
max_term_length=20
min_doc_freq=2

# --------------------------------------------------------------

def load_word_list(in_path):
    stopwords = set()
    with open(in_path) as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip().lower()
            if len(l) >= min_term_length:
                stopwords.add(l)
    return list(stopwords)

def custom_tokenizer(s):
    """ Tokenizer for tweet text """
    tknzr = TweetTokenizer(strip_handles=strip_handles, reduce_len=reduce_len)
    tokens = []
    for tok in tknzr.tokenize(s):
        tok = tok.lower().replace("'","")
        if len(tok) == 0:
            continue
        if not (tok[0].isalpha() or tok[0] == '#' or tok[0] == '@'):
            continue
        if strip_urls and (tok.startswith("http:") or tok.startswith("https:") or tok.startswith("www.")):
            continue
        if len(tok) < min_term_length or len(tok) > max_term_length:
            continue
        tokens.append(tok)
    return tokens

def vision_tokenizer(s):
    """ Tokenizer for Google vision labels """
    tokens = []
    for tok in s.lower().strip().split(","):
        if len(tok) < min_term_length:
            continue
        tokens.append(tok)
    return tokens

def create_pipeline(clf, stopwords, tokenizer=custom_tokenizer, tfidf=True):
    if tfidf:
        return Pipeline([
            ('vec', CountVectorizer(min_df=min_doc_freq, stop_words=stopwords, tokenizer=tokenizer, strip_accents="unicode", token_pattern=None),),
            ('tfidf', TfidfTransformer(norm='l2')),
            ('clf', clf)
        ])
    return Pipeline([
        ('vec', CountVectorizer(min_df=min_doc_freq, stop_words=stopwords, tokenizer=tokenizer, strip_accents="unicode", token_pattern=None)),
        ('norm', Normalizer(norm='l2')),
        ('clf', clf)
    ])

# --------------------------------------------------------------

def compile_cv_results(cv_scores):
    results = {}
    for key in cv_scores:
        if key.startswith("test_"):
            results[key.replace("test_","")] = cv_scores[key].mean()
    return results    

def apply_cv(df, pipeline, cv_folds, text_col="text"):
    documents, target = df[text_col], df["label"]
    cv_scores = cross_validate(pipeline, documents, target, cv=cv_folds, scoring=scoring_measures)
    return compile_cv_results(cv_scores)
    
def apply_grid_cv(df, pipeline, cv_folds, param_grid, text_col="text"):
    documents, target = df[text_col], df["label"]
    log.info("Performing grid search ...")
    grid = GridSearchCV(pipeline, cv=cv_folds, scoring=grid_measure, param_grid=param_grid)   
    grid.fit(documents, target)
    log.info("Grid CV best score: %.4f, %s" % (grid.best_score_, grid.best_params_))
    # perform classification with the best parameters
    log.info("Performing final classification ...")
    pipeline.clf = grid.best_estimator_
    return apply_cv(df, pipeline, cv_folds, text_col=text_col)
