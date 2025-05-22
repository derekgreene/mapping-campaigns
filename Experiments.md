# Experiments

Python scripts correspond to list of experiments conducted in the paper.


## Experiments - Dataset 1 (Images + Text)

### Experiment 1: Bag-of-words text classification

Apply standard classifiers with a bag-of-words model:

	python code/classify-bow.py data/dataset1/policy.csv -s code/stopwords.txt -o results/dataset1/policy-bow.csv

	python code/classify-bow.py data/dataset1/electioneering.csv -s code/stopwords.txt -o results/dataset1/electioneering-bow.csv 

### Experiment 2: BERT classification

Use pre-trained embeddings in combination with classifiers:

	python code/classify-bert.py data/dataset1/policy.csv -o results/dataset1/policy-bert-cv.csv

	python code/classify-bert.py data/dataset1/electioneering.csv -o results/dataset1/electioneering-bert.csv

### Experiment 3: Sentence-BERT classification

Use pre-trained embeddings for sentence-length classification, using several different pre-trained models:

	python code/classify-sbert.py data/dataset1/policy.csv -m all-MiniLM-L6-v2 -o results/dataset1/policy-sbert_sentence+minilml6v2.csv

	python code/classify-sbert.py data/dataset1/policy.csv -m all-mpnet-base-v2 -o results/dataset1/policy-sbert_sentence+mpnetbasev2.csv

	python code/classify-sbert.py data/dataset1/policy.csv -m all-distilroberta-v1 -o results/dataset1/policy-sbert_sentence+distilrobertav1.csv

	python code/classify-sbert.py data/dataset1/electioneering.csv -m all-MiniLM-L6-v2 -o results/dataset1/electioneering-sbert_sentence+minilml6v2.csv

	python code/classify-sbert.py data/dataset1/electioneering.csv -m all-mpnet-base-v2 -o results/dataset1/electioneering-sbert_sentence+mpnetbasev2.csv

	python code/classify-sbert.py data/dataset1/electioneering.csv -m all-distilroberta-v1 -o results/dataset1/electioneering-sbert_sentence+distilrobertav1.csv


## Experiments - Dataset 2 (Text-Only)

### Experiment 1: Bag-of-words text classification

Apply standard classifiers with a bag-of-words model:

	python code/classify-bow.py data/dataset2/policy.csv -s code/stopwords.txt -o results/dataset2/policy-bow.csv

	python code/classify-bow.py data/dataset2/electioneering.csv -s code/stopwords.txt -o results/dataset2/electioneering-bow.csv 

### Experiment 2: BERT classification

Use pre-trained embeddings in combination with classifiers:

	python code/classify-bert.py data/dataset2/policy.csv -o results/dataset2/policy-bert.csv

	python code/classify-bert.py data/dataset2/electioneering.csv -o results/dataset2/electioneering-bert.csv

### Experiment 3: Sentence-BERT classification

Use pre-trained embeddings for sentence-length classification, using several different pre-trained models:

	python code/classify-sbert.py data/dataset2/policy.csv -m all-MiniLM-L6-v2 -o results/dataset2/policy-sbert_sentence+minilml6v2.csv

	python code/classify-sbert.py data/dataset2/policy.csv -m all-mpnet-base-v2 -o results/dataset2/policy-sbert_sentence+mpnetbasev2.csv

	python code/classify-sbert.py data/dataset2/policy.csv -m all-distilroberta-v1 -o results/dataset2/policy-sbert_sentence+distilrobertav1.csv

	python code/classify-sbert.py data/dataset2/electioneering.csv -m all-MiniLM-L6-v2 -o results/dataset2/electioneering-sbert_sentence+minilml6v2.csv

	python code/classify-sbert.py data/dataset2/electioneering.csv -m all-mpnet-base-v2 -o results/dataset2/electioneering-sbert_sentence+mpnetbasev2.csv

	python code/classify-sbert.py data/dataset2/electioneering.csv -m all-distilroberta-v1 -o results/dataset2/electioneering-sbert_sentence+distilrobertav1.csv


## Experiments - Combined Dataset

### Experiment 1: Bag-of-words text classification

Apply standard classifiers with a bag-of-words model:

	python code/classify-bow.py data/combined/policy.csv -s code/stopwords.txt -o results/combined/policy-bow.csv

	python code/classify-bow.py data/combined/electioneering.csv -s code/stopwords.txt -o results/combined/electioneering-bow.csv 

### Experiment 2: BERT classification

Use pre-trained embeddings in combination with classifiers:

	python code/classify-bert.py data/combined/policy.csv -o results/combined/policy-bert.csv

	python code/classify-bert.py data/combined/electioneering.csv -o results/combined/electioneering-bert.csv

### Experiment 3: Sentence-BERT classification

Use pre-trained embeddings for sentence-length classification, using several different pre-trained models:

	python code/classify-sbert.py data/combined/policy.csv -m all-MiniLM-L6-v2 -o results/combined/policy-sbert_sentence+minilml6v2.csv

	python code/classify-sbert.py data/combined/policy.csv -m all-mpnet-base-v2 -o results/combined/policy-sbert_sentence+mpnetbasev2.csv

	python code/classify-sbert.py data/combined/policy.csv -m all-distilroberta-v1 -o results/combined/policy-sbert_sentence+distilrobertav1.csv

	python code/classify-sbert.py data/combined/electioneering.csv -m all-MiniLM-L6-v2 -o results/combined/electioneering-sbert_sentence+minilml6v2.csv

	python code/classify-sbert.py data/combined/electioneering.csv -m all-mpnet-base-v2 -o results/combined/electioneering-sbert_sentence+mpnetbasev2.csv

	python code/classify-sbert.py data/combined/electioneering.csv -m all-distilroberta-v1 -o results/combined/electioneering-sbert_sentence+distilrobertav1.csv


## Experiments - Classification of Unseen Data

Build the language model for the unseen data:

	python code/build-sbert.py data/raw/data_tweets_all.csv --seed=101 -m all-distilroberta-v1 -o models/sbert_distilrobertav1.bin

Experiment on all unseen tweets:

	python code/classify-unseen.py data/raw/data_tweets_all.csv models/sbert_distilrobertav1.bin -t policy --seed=101 -o results/unseen/policy-sbert_distilrobertav1.csv 

	python code/classify-unseen.py data/raw/data_tweets_all.csv models/sbert_distilrobertav1.bin -t electioneering --seed=101 -o results/unseen/electioneering-sbert_distilrobertav1.csv 


## Experiments - Classifier Verification

Final verification experiment on third annotated hold out set (400 tweets with images+text, 400 tweets with text only):

 	python code/classify-final.py data/combined/policy.csv data/dataset3/policy.csv -t policy --seed=101 -m all-distilroberta-v1 -o results/final/policy-sbert_distilrobertav1.csv

 	python code/classify-final.py data/combined/electioneering.csv data/dataset3/electioneering.csv -t electioneering --seed=101 -m all-distilroberta-v1 -o results/final/electioneering-sbert_distilrobertav1.csv

