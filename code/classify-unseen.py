#!/usr/bin/env python
"""
Tool to perform Sentence BERT-based document classification on tweet text for an unseen set of tweets.

Sample usage:
python code/classify-unseen.py data/raw/data_tweets_all.csv models/sbert_distilrobertav1.bin -t policy --seed=101 -o results/unseen/policy-sbert.csv 
"""
from pathlib import Path
import random, joblib, sys
from optparse import OptionParser
import logging as log
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from evaluation import ScoreCollection, scoring_measures, grid_measure
from util import read_tweets, seed_random

# --------------------------------------------------------------
# Settings
cv_folds = 5
include_training_data = True

# --------------------------------------------------------------

def apply_grid_cv(features, labels, clf, cv_folds, param_grid):
	log.info("Applying grid search: %s" % str(clf))
	grid = GridSearchCV(clf, cv=cv_folds, scoring=grid_measure, param_grid=param_grid)   
	grid.fit(features, labels)
	log.info("Grid CV best score: %.4f, %s" % (grid.best_score_, grid.best_params_))
	return grid.best_estimator_

def predictions_to_series(task, predictions, test_ids):
	labels = {}
	for pos in range(len(predictions)):
		if predictions[pos] == 1:
			labels[test_ids[pos]] = task
		else:
			labels[test_ids[pos]] = "non-" + task
	return pd.Series(labels)

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] data_path")
	parser.add_option("-t", action="store", type="string", dest="task", help="task name (policy, electionering)", default=None)
	parser.add_option("--seed", action="store", type="int", dest="seed", help="initial random seed", default=101)
	parser.add_option("-o", action="store", type="string", dest="out_path", help="output path", default=None)
	(options, args) = parser.parse_args()
	if len(args) != 2:
		parser.error( "Must specify dataset file and cached model file" )	
	log.basicConfig(level=log.INFO, format='%(message)s')
	if options.task is None:
		parser.error( "Must task name:  policy, electionering)")	
	task = options.task.lower()
	include_lr_probabilities = True

	# output directory for results
	if options.out_path is None:
		out_path = Path("predictions.csv")
	else:
		out_path = options.out_path

	# read the data
	in_path = Path(args[0])
	log.info("Reading data from %s" % in_path)
	df = read_tweets(in_path)
	num_rows = len(df)
	log.info("Read dataset with %d rows" % num_rows)

	# read the cache model
	model_path = Path(args[1])
	log.info("Reading cached model from %s" % model_path)
	embedding = joblib.load(model_path)
	log.info("Using embedding of size: %s" % str(embedding.shape))
	if num_rows != embedding.shape[0]:
		log.error("Number of rows in dataset does not match number of rows in embedding")
		sys.exit(1)

	# Approach 1: Separate training and test sets
	log.info("Creating test set (include_training_data=%s) ..." % include_training_data)
	pos = 0
	train_positions, test_positions = [], []
	train_ids, test_ids = [], []
	train_labels = []
	for tweet_id, row in df.iterrows():
		# an annotated tweet?
		if row["tweet_coded"] == 1:
			train_positions.append(pos)
			train_ids.append(tweet_id)
			label = None
			if task == "policy":
				if not pd.isnull(row["coded_policy_content_image_text"]):
					label = int(row["coded_policy_content_image_text"])
				else:
					label = int(row["coded_policy_content_text"])
			elif task == "electioneering":
				if not pd.isnull(row["coded_electioneering_image_text"]):
					label = int(row["coded_electioneering_image_text"])
				else:
					label = int(row["coded_electioneering_text"])
			else:
				log.warning("Unknown task %s" % task)
			if label is None:
				log.warning("No label for row %d" % (pos+1))
			else:
				train_labels.append(label)
		# add to the test set?
		if include_training_data or row["tweet_coded"] == 0:
			test_positions.append(pos)
			test_ids.append(tweet_id)
		pos += 1

	# get the subset of submissions
	train_embedding = embedding[train_positions]
	test_embedding = embedding[test_positions]
	log.info("Original embedding size: %s" % str(embedding.shape))
	log.info("Training embedding size: %s" % str(train_embedding.shape))
	log.info("Testing embedding size: %s" % str(test_embedding.shape))
	log.info("Training labels size: %s" % len(train_labels))
	log.info("Label counts: %s" % pd.Series(train_labels).value_counts().to_dict())
	
	# create the classifiers
	clf_lr = LogisticRegression(max_iter=1000)
	clf_knn = KNeighborsClassifier(n_neighbors=1)
	clf_sgd = SGDClassifier(loss="hinge")

	# # Apply GridCV + KNN
	# log.info("- Applying KNN + GridCV")
	# params = {"n_neighbors":list(range(1, 21))}
	# seed_random(options.seed)
	# best_knn = apply_grid_cv(train_embedding, train_labels, clf_knn, cv_folds, params)
	# log.info("Applying best model: %s" % best_knn)
	# predictions = best_knn.predict(test_embedding)
	# log.info("Generated %d predictions" % len(predictions))
	# predicted_labels = predictions_to_series(task, predictions, test_ids)
	# log.info("Prediction counts: %s" % predicted_labels.value_counts().to_dict())
	# df["prediction_%s_knn" % task] = predicted_labels

	# # Apply GridCV + SVM
	# log.info("- Applying SVM + GridCV")
	# params = {'l1_ratio': [.05, .1, .15, .2, .25, .3, .4, .5, .6, .7, .8, .9, .95, .99, 1],
 #              'alpha': np.power(10, np.arange(-4, 1, dtype=float))}
	# seed_random(options.seed)
	# best_svm = apply_grid_cv(train_embedding, train_labels, clf_sgd, cv_folds, params)
	# log.info("Applying best model: %s" % best_svm)
	# predictions = best_svm.predict(test_embedding)
	# log.info("Generated %d predictions" % len(predictions))
	# predicted_labels = predictions_to_series(task, predictions, test_ids)
	# log.info("Prediction counts: %s" % predicted_labels.value_counts().to_dict())
	# df["prediction_%s_svm" % task] = predicted_labels

	# Apply GridCV + LR
	log.info("- Applying Logistic Regression + GridCV")
	params = {"C":np.logspace(-3,3,7)}
	seed_random(options.seed)
	best_lr = apply_grid_cv(train_embedding, train_labels, clf_lr, cv_folds, params)
	log.info("Applying best model: %s" % best_lr)
	predictions = best_lr.predict(test_embedding)
	log.info("Generated %d predictions" % len(predictions))
	predicted_labels = predictions_to_series(task, predictions, test_ids)
	log.info("Prediction counts: %s" % predicted_labels.value_counts().to_dict())
	df["prediction_%s_lr" % task] = predicted_labels
	# add the probabilities
	if include_lr_probabilities:
		log.info("Calculating Logistic Regression probability values ...")
		class_names = best_lr.classes_
		log.info("Class names: %s" % class_names)
		class_probs = best_lr.predict_proba(test_embedding)
		log.info("Probability counts: %s" % str(class_probs.shape))
		classname_0, classname_1 = class_names[0], class_names[1]
		prob_0, prob_1 = {}, {}
		for row, tweet_id in enumerate(list(df.index)):
			prob_0[tweet_id] = class_probs[row,0]
			prob_1[tweet_id] = class_probs[row,1]
		df["prob_%s_lr%s" % (task,classname_0)] = pd.Series(prob_0)
		df["prob_%s_lr%s" % (task,classname_1)] = pd.Series(prob_1)

	# Export the results
	log.info("Writing %s" % out_path)
	df.to_csv(out_path)

# --------------------------------------------------------------

if __name__ == "__main__":
	main()
