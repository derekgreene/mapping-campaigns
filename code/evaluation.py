import pandas as pd
from pathlib import Path
from tabulate import tabulate
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score

# --------------------------------------------------------------

scoring_measures = {'f1': 'f1', 'accuracy': 'accuracy', 'bar':'balanced_accuracy', 
           'precision':'precision', 'recall':'recall'}
ordered_measures=["accuracy", "bar", "f1", "precision", "recall"]
grid_measure = ordered_measures[0]
sort_measures = ordered_measures[0:2]

# --------------------------------------------------------------

class ScoreCollection:
	def __init__(self):
		self.all_results = []

	def add(self, method, scores):
		scores["method"] = method
		self.all_results.append(scores)

	def to_df(self, sort=True):
		df = pd.DataFrame(self.all_results, columns=["method"]+ordered_measures).set_index("method")
		if sort:
			df = df.sort_values(by=sort_measures, ascending=False)
		return df

	def __str__(self):
		return tabulate(self.to_df(), headers = 'keys', tablefmt = 'psql', floatfmt=".4f")

	def save(self, out_path):
		self.to_df(sort=False).to_csv(out_path, float_format="%.4f")


# --------------------------------------------------------------

def calc_evaluation_results(y_true, y_pred_classes):
	results = {}
	results["accuracy"] = accuracy_score(y_true, y_pred_classes)
	results["bar"] = balanced_accuracy_score(y_true, y_pred_classes)
	results["precision"] = precision_score(y_true, y_pred_classes, average='binary')
	results["recall"] = recall_score(y_true, y_pred_classes, average='binary')
	results["f1"] = f1_score(y_true, y_pred_classes, average='binary')
	return results
