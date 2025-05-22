import random
import logging as log
import pandas as pd
import numpy as np
import torch

# --------------------------------------------------------------

def read_tweets(tweet_file_path):
    df_tweets = pd.read_csv(tweet_file_path)
    df_tweets['tweet_id'] = df_tweets['tweet_id'].str.lower()
    df_tweets['tweet_id'] = df_tweets['tweet_id'].str.replace('.jpg','', regex=False)
    df_tweets["text"] = df_tweets["text"].str.replace('\n',' ', regex=True)
    return df_tweets.set_index("tweet_id")
    
# --------------------------------------------------------------

def seed_random(random_seed):
    log.info("Resetting random seed %s" % random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)    