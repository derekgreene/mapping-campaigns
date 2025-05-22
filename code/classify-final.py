#!/usr/bin/env python
"""
Tool to evaluate Sentence BERT-based document classification using a training set and a separate hold out set.

Sample usage:
python code/classify-final.py data/combined/policy.csv data/dataset3/policy.csv -t policy --seed=101 -m all-distilroberta-v1 -o results/final/policy-sbert.csv 
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
from sklearn.metrics import confusion_matrix
from evaluation import grid_measure, ScoreCollection, calc_evaluation_results

import bert
from util import read_tweets, seed_random

# --------------------------------------------------------------
# Settings
tidy_text = True
cv_folds = 5

# --------------------------------------------------------------

def apply_grid_cv(features, labels, clf, cv_folds, param_grid):
	log.info("Applying grid search: %s" % str(clf))
	grid = GridSearchCV(clf, cv=cv_folds, scoring=grid_measure, param_grid=param_grid)   
	grid.fit(features, labels)
	log.info("Grid CV best score: %.4f, %s" % (grid.best_score_, grid.best_params_))
	return grid.best_estimator_

def check_errors(df_test, test_labels, predictions, labels):
	log.info("Negative=%s Positive=%s" % (labels[0], labels[1]))
	M = confusion_matrix(test_labels, predictions, labels=labels)
	log.info(M)
	tn, fp, fn, tp = M.ravel()
	log.info("tn=%d, fp=%d, fn=%d, tp=%d" % (tn, fp, fn, tp))
	# check for individual errors
	rows = []
	num_errors, pos = 0, 0
	df_errors = df_test.copy().reset_index().set_index("tweet_id")
	for i, row in df_test.iterrows():
		df_errors["prediction"] = predictions[pos]
		if test_labels[pos] == predictions[pos]:
			df_errors.at[i, "correct"] = 1
		else:
			df_errors.at[i, "correct"] = 0
			num_errors += 1
		pos += 1
	# errors = []
	# pos = 0
	# df_test2 = df_test.copy().reset_index()
	# for i, row in df_test2.iterrows():
	# 	if test_labels[pos] != predictions[pos]:
	# 		errors.append(row)
	# 	pos +=1
	# df_errors = pd.DataFrame(errors).set_index("tweet_id")
	# log.info("Errors: %d/%d of test set" % (len(df_errors), len(df_test2)))
	log.info("Errors: %d/%d of test set" % (num_errors, len(df_test)))
	return df_errors

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] data_path1 data_path2")
	parser.add_option("-t", action="store", type="string", dest="task", help="task name (policy, electionering)", default=None)
	parser.add_option("--seed", action="store", type="int", dest="seed", help="initial random seed", default=101)
	parser.add_option("-o", action="store", type="string", dest="out_path", help="output path", default=None)
	parser.add_option("-m","--model", action="store", type="string", dest="model_id", help="model name to apply", default="all-MiniLM-L6-v2")
	(options, args) = parser.parse_args()
	if len(args) != 2:
		parser.error( "Must specify two dataset files" )	
	log.basicConfig(level=log.INFO, format='%(message)s')
	if options.task is None:
		parser.error( "Must task name:  policy, electionering)")	
	task = options.task.lower()

	# output directory for results
	if options.out_path is None:
		out_path = Path("predictions.csv")
	else:
		out_path = options.out_path

	# read the training data
	train_in_path = Path(args[0])
	log.info("Reading training data from %s" % train_in_path)
	df_train = read_tweets(train_in_path)
	num_train = len(df_train)
	log.info("Read training dataset with %d rows" % num_train)
	train_ids = list(df_train.index)
	train_positions = list(range(0, num_train))
	train_labels = list(df_train["label"])
	log.info("Training labels size: %s" % len(train_labels))
	log.info("Training label counts: %s" % pd.Series(train_labels).value_counts().to_dict())

	# read the test data
	test_in_path = Path(args[1])
	log.info("Reading training data from %s" % test_in_path)
	df_test = read_tweets(test_in_path)
	num_test = len(df_test)
	log.info("Read training dataset with %d rows" % num_test)
	test_ids = list(df_test.index)
	test_positions = list(range(num_train, num_train+num_test))
	test_labels = list(df_test["label"])
	log.info("Test labels size: %s" % len(test_labels))
	log.info("Test label counts: %s" % pd.Series(test_labels).value_counts().to_dict())

	# combine the datasets
	df = pd.concat([df_train, df_test])
	log.info("Concatenated dataset has %d rows" % len(df))

	# Clean up tweet text?
	if tidy_text:
		log.info("Tidying text ...")
		df["text"] = df.apply(lambda x: bert.clean_text(x["text"]), axis=1)

	# Build the model
	model_path = Path("/tmp/" + task + "embed.bin")
	if model_path.exists():
		log.info("Reading cached model from %s" % model_path)
		embedding = joblib.load(model_path)
	else:
		seed_random(options.seed)
		embedding = bert.apply_sentence_bert(df, options.model_id)
		log.info("Generated embedding of size: %s" % str(embedding.shape))
		joblib.dump(embedding, model_path)

	embedding = embedding.cpu()
	
	# seed_random(options.seed)
	# embedding = bert.apply_sentence_bert(df, options.model_id)
	# log.info("Generated embedding of size: %s" % str(embedding.shape))
	# # joblib.dump(embedding, model_path)
		
	train_embedding = embedding[train_positions]
	test_embedding = embedding[test_positions]
	log.info("Original embedding size: %s" % str(embedding.shape))
	log.info("Training embedding size: %s" % str(train_embedding.shape))
	log.info("Testing embedding size: %s" % str(test_embedding.shape))

	# create the classifiers
	clf_lr = LogisticRegression(max_iter=1000)
	clf_knn = KNeighborsClassifier(n_neighbors=1)
	clf_sgd = SGDClassifier(loss="hinge")

	scores = ScoreCollection()

	# Apply GridCV + KNN
	log.info("- Applying KNN + GridCV")
	params = {"n_neighbors":list(range(1, 21))}
	seed_random(options.seed)
	best_knn = apply_grid_cv(train_embedding, train_labels, clf_knn, cv_folds, params)
	log.info("Applying best model: %s" % best_knn)
	predictions = best_knn.predict(test_embedding)
	log.info("Generated %d predictions" % len(predictions))
	results_knn = calc_evaluation_results(test_labels, predictions)
	scores.add("knn", results_knn)
	# check where the errors lie
	df_errors = check_errors(df_test, test_labels, predictions, best_knn.classes_)
	errors_out_path = "errors_%s_knn.csv" % task
	log.info("Writing errors to %s" % errors_out_path)
	df_errors.to_csv(errors_out_path)

	# Apply GridCV + SVM
	log.info("- Applying SVM + GridCV")
	params = {'l1_ratio': [.05, .1, .15, .2, .25, .3, .4, .5, .6, .7, .8, .9, .95, .99, 1],
              'alpha': np.power(10, np.arange(-4, 1, dtype=float))}
	seed_random(options.seed)
	best_svm = apply_grid_cv(train_embedding, train_labels, clf_sgd, cv_folds, params)
	log.info("Applying best model: %s" % best_svm)
	predictions = best_svm.predict(test_embedding)
	log.info("Generated %d predictions" % len(predictions))
	results_svm = calc_evaluation_results(test_labels, predictions)
	scores.add("svm", results_svm)
	# check where the errors lie
	df_errors = check_errors(df_test, test_labels, predictions, best_svm.classes_)
	errors_out_path = "errors_%s_svm.csv" % task
	log.info("Writing errors to %s" % errors_out_path)
	df_errors.to_csv(errors_out_path)

	# Apply GridCV + LR
	log.info("- Applying Logistic Regression + GridCV")
	params = {"C":np.logspace(-3,3,7)}
	seed_random(options.seed)
	best_lr = apply_grid_cv(train_embedding, train_labels, clf_lr, cv_folds, params)
	log.info("Applying best model: %s" % best_lr)
	predictions = best_lr.predict(test_embedding)
	log.info("Generated %d predictions" % len(predictions))
	results_lr = calc_evaluation_results(test_labels, predictions)
	scores.add("lr", results_lr)
	# check where the errors lie
	df_errors = check_errors(df_test, test_labels, predictions, best_lr.classes_)
	errors_out_path = "errors_%s_lr.csv" % task
	log.info("Writing errors to %s" % errors_out_path)
	df_errors.to_csv(errors_out_path)

	# display the results
	log.info(str(scores))
	# export the results
	log.info("Writing %s" % out_path)
	scores.save(out_path)

# --------------------------------------------------------------

if __name__ == "__main__":
	main()
