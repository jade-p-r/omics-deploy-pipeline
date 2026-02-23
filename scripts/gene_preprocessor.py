from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


TARGET_COLUMN = "sample_type_id"
EPSILON = 1e-5
DEFAULT_PCA_COMPONENTS = 3


@dataclass
class GenePreprocessor:
    target_column: str = TARGET_COLUMN
    n_components: int = DEFAULT_PCA_COMPONENTS
    epsilon: float = EPSILON
    no_variance_columns: List[str] = None
    model_feature_columns: List[str] = None
    pca: PCA = None

    def fit(self, dataframe: pd.DataFrame) -> "GenePreprocessor":
        if self.target_column not in dataframe.columns:
            raise ValueError(f"Missing target column: {self.target_column}")

        features = dataframe.drop(columns=[self.target_column])
        variances = features.var(axis=0)
        self.no_variance_columns = variances[variances == 0].index.tolist()
        self.model_feature_columns = [
            column for column in features.columns if column not in self.no_variance_columns
        ]

        transformed_features = self._log_transform(features[self.model_feature_columns])
        max_components = min(
            self.n_components,
            transformed_features.shape[0],
            transformed_features.shape[1],
        )
        if max_components < 1:
            raise ValueError("Not enough usable features to fit PCA.")

        self.pca = PCA(n_components=max_components)
        self.pca.fit(transformed_features)
        return self

    def transform_features(self, features_dataframe: pd.DataFrame) -> np.ndarray:
        self._assert_is_fitted()
        aligned = features_dataframe.reindex(columns=self.model_feature_columns, fill_value=0.0)
        transformed_features = self._log_transform(aligned)
        return self.pca.transform(transformed_features)

    def transform_dataset(self, dataframe: pd.DataFrame) -> np.ndarray:
        if self.target_column in dataframe.columns:
            features_dataframe = dataframe.drop(columns=[self.target_column])
        else:
            features_dataframe = dataframe
        return self.transform_features(features_dataframe)

    def transform_new_gene_entry(self, gene_entry: Dict[str, float]) -> np.ndarray:
        entry_dataframe = pd.DataFrame([gene_entry])
        return self.transform_features(entry_dataframe)

    def _log_transform(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        return np.log(feature_frame.astype(float) + self.epsilon)

    def _assert_is_fitted(self) -> None:
        if self.pca is None or self.model_feature_columns is None:
            raise ValueError("Preprocessor is not fitted. Fit or load it before transforming data.")
