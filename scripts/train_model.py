import argparse
import os
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from scripts.gene_preprocessor import GenePreprocessor


TARGET_COLUMN = "sample_type_id"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data/Liver RNA Data.csv")
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "pca_random_forest_model.joblib")
DEFAULT_PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, "pca_preprocessor.joblib")


def train_model(
	data_path: str = DEFAULT_DATA_PATH,
	model_path: str = DEFAULT_MODEL_PATH,
	preprocessor_path: str = DEFAULT_PREPROCESSOR_PATH,
	test_size: float = 0.2,
	random_state: int = 42,
) -> None:
	dataframe = pd.read_csv(data_path)

	preprocessor = GenePreprocessor().fit(dataframe)
	transformed_features = preprocessor.transform_dataset(dataframe)
	target = dataframe[TARGET_COLUMN]

	X_train, X_test, y_train, y_test = train_test_split(
		transformed_features,
		target,
		test_size=test_size,
		random_state=random_state,
		stratify=target,
	)

	model = RandomForestClassifier(
		n_estimators=300,
		random_state=random_state,
		n_jobs=-1,
	)
	model.fit(X_train, y_train)

	accuracy = model.score(X_test, y_test)
	predictions = model.predict(X_test)
	balanced_acc = balanced_accuracy_score(y_test, predictions)

	print(f"Accuracy: {accuracy:.4f}")
	print(f"Balanced Accuracy: {balanced_acc:.4f}")

	joblib.dump(model, model_path)
	joblib.dump(preprocessor, preprocessor_path)
	print(f"Saved model to: {model_path}")
	print(f"Saved preprocessor to: {preprocessor_path}")


def preprocess_new_gene_entry(
	gene_entry: Dict[str, float],
	preprocessor_path: str = DEFAULT_PREPROCESSOR_PATH,
) -> np.ndarray:
	preprocessor: GenePreprocessor = joblib.load(preprocessor_path)
	return preprocessor.transform_new_gene_entry(gene_entry)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train RNA classifier and save preprocessing artifacts.")
	parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to training CSV")
	parser.add_argument("--model-out", default=DEFAULT_MODEL_PATH, help="Output model path")
	parser.add_argument(
		"--preprocessor-out",
		default=DEFAULT_PREPROCESSOR_PATH,
		help="Output preprocessor path",
	)
	parser.add_argument("--test-size", type=float, default=0.2, help="Test set ratio")
	parser.add_argument("--random-state", type=int, default=42, help="Random seed")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	train_model(
		data_path=args.data,
		model_path=args.model_out,
		preprocessor_path=args.preprocessor_out,
		test_size=args.test_size,
		random_state=args.random_state,
	)