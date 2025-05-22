#!/usr/bin/env python
"""
Tool to perform Sentence BERT-based document classification on tweet text

Sample usage:
python code/classify-sbert.py data/dataset1/policy.csv --seed=101 -m all-MiniLM-L6-v2 -o results/dataset1/policy-sbert.csv

Models include:
all-MiniLM-L6-v2 all-mpnet-base-v2 all-distilroberta-v1
"""
from pathlib import Path
import random
from optparse import OptionParser
import logging as log
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import bert
from evaluation import ScoreCollection
from util import read_tweets, seed_random

# --------------------------------------------------------------
# Settings
tidy_text = True
cv_folds = 5

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] data_path")
	parser.add_option("-o", action="store", type="string", dest="out_path", help="output path", default=None)
	parser.add_option("--seed", action="store", type="int", dest="seed", help="initial random seed", default=101)
	parser.add_option("-m","--model", action="store", type="string", dest="model_id", help="model name to apply", default="all-MiniLM-L6-v2")
	(options, args) = parser.parse_args()
	if len(args) != 1:
		parser.error( "Must specify data file" )	
	log.basicConfig(level=log.INFO, format='%(message)s')

	# read the data
	in_path = Path(args[0])
	log.info("Reading data from %s" % in_path)
	df = read_tweets(in_path)
	log.info("Read dataset with %d rows" % len(df))

	# output directory for results
	if options.out_path is None:
		out_path = Path("results.csv")
	else:
		out_path = options.out_path

	# Clean up tweet text?
	if tidy_text:
		log.info("Tidying text ...")
		df["text"] = df.apply(lambda x: bert.clean_text(x["text"]), axis=1)

	# Build the model
	features = bert.apply_sentence_bert(df, options.model_id).cpu()
	log.info("Generated embedding of size: %s" % str(features.shape))

	# Create the classifiers
	labels = df["label"]
	clf_lr = LogisticRegression(max_iter=1000)
	clf_knn = KNeighborsClassifier(n_neighbors=1)
	clf_sgd = SGDClassifier(loss="hinge")
	
	scores = ScoreCollection()

	# Apply CV + KNN
	log.info("- Applying KNN + CV")
	seed_random(options.seed)
	experiment = bert.apply_cv(features, labels, clf_knn, cv_folds)
	log.info(experiment)
	scores.add("knn-cv", experiment)

	# Apply CV + SVM
	log.info("- Applying SVM + CV")
	seed_random(options.seed)
	experiment = bert.apply_cv(features, labels, clf_sgd, cv_folds)
	log.info(experiment)
	scores.add("svm-cv", experiment)

	# Apply CV + Logistic Regression
	log.info("- Applying Logistic Regression + CV")
	seed_random(options.seed)
	experiment = bert.apply_cv(features, labels, clf_lr, cv_folds)
	log.info(experiment)
	scores.add("lr-cv", experiment)

	# # Apply GridCV + KNN
	log.info("- Applying KNN + GridCV")
	seed_random(options.seed)
	params = {"n_neighbors":list(range(1, 21))}
	experiment = bert.apply_grid_cv(features, labels, clf_knn, cv_folds, params)
	log.info(experiment)
	scores.add("knn-gridcv", experiment)

	# Apply GridCV + SVM
	log.info("- Applying SVM + GridCV")
	seed_random(options.seed)
	params = {'l1_ratio': [.05, .1, .15, .2, .25, .3, .4, .5, .6, .7, .8, .9, .95, .99, 1],
              'alpha': np.power(10, np.arange(-4, 1, dtype=float))}
	experiment = bert.apply_grid_cv(features, labels, clf_sgd, cv_folds, params)
	log.info(experiment)
	scores.add("svm-gridcv", experiment)

	# Apply GridCV + Logistic Regression
	log.info("- Applying Logistic Regression + GridCV")
	seed_random(options.seed)
	params = {"C":np.logspace(-3,3,7)}
	experiment = bert.apply_grid_cv(features, labels, clf_lr, cv_folds, params)
	log.info(experiment)
	scores.add("lr-gridcv", experiment)

	# display the results
	log.info(str(scores))
	# export the results
	log.info("Writing %s" % out_path)
	scores.save(out_path)

# --------------------------------------------------------------

if __name__ == "__main__":
	main()
