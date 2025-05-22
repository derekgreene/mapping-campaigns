#!/usr/bin/env python
"""
Tool to build and store a Sentence BERT-based based on an input dataset of tweets.

Sample usage:
python code/build-sbert.py data/raw/data_tweets_all.csv --seed=101 -m all-MiniLM-L6-v2 -o models/combined/model-policy-sbert.bin

Models include:
all-MiniLM-L6-v2 all-mpnet-base-v2 all-distilroberta-v1
"""
from pathlib import Path
import random, joblib
from optparse import OptionParser
import logging as log
import pandas as pd
import numpy as np
import torch
import bert
from util import read_tweets

# --------------------------------------------------------------
# Settings
tidy_text = True

# --------------------------------------------------------------

def seed_random(random_seed):
	log.info("Resetting random seed %s" % random_seed)
	random.seed(random_seed)
	np.random.seed(random_seed)
	torch.manual_seed(random_seed)

# --------------------------------------------------------------

def main():
	parser = OptionParser(usage="usage: %prog [options] data_path")
	parser.add_option("-o", action="store", type="string", dest="out_path", help="model output path", default=None)
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

	# output path for stored model
	if options.out_path is None:
		out_path = Path("model.bin")
	else:
		out_path = options.out_path

	# Clean up tweet text?
	if tidy_text:
		log.info("Tidying text ...")
		df["text"] = df.apply(lambda x: bert.clean_text(x["text"]), axis=1)

	# Build the model
	embedding = bert.apply_sentence_bert(df, options.model_id)
	log.info("Generated embedding of size: %s" % str(embedding.shape))

	# Store the model
	log.info("Writing embedding to %s ..." % out_path)
	joblib.dump(embedding, out_path)

# --------------------------------------------------------------

if __name__ == "__main__":
	main()
