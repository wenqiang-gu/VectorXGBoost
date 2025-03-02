import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score

class VectorXGBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, gen1_feature_groups, jagged_features=None, gen1_params=None, gen2_params=None, n_splits=5):
        """
        Dual-layer XGBoost classifier with vector (jagged array) feature support.

        Parameters:
        - gen1_feature_groups: dict, where keys are sub-model names and values are lists of feature indices.
        - jagged_features: list, feature indices that contain jagged arrays.
        - gen1_params: dict, parameters for generation-1 XGBoost models.
        - gen2_params: dict, parameters for the final XGBoost classifier.
        - n_splits: int, number of folds for cross-validation in gen-1 stage.
        """
        self.gen1_feature_groups = gen1_feature_groups  # Dict {"sub_model1": [idx1, idx2], ...}
        self.jagged_features = jagged_features or []  # Indices of jagged array features
        self.gen1_params = gen1_params or {"objective": "binary:logistic", "eval_metric": "logloss"}
        self.gen2_params = gen2_params or {"objective": "binary:logistic", "eval_metric": "logloss"}
        self.n_splits = n_splits
        self.gen1_models = {}  # Store Gen-1 models
        self.final_model = None  # Store final model

    def fit(self, X, y):
        """
        Train the dual-layer XGBoost classifier, expanding jagged arrays into multiple rows.
        """
        if not self.gen1_feature_groups:
            raise ValueError("gen1_feature_groups must be specified as a dictionary.")

        gen1_scores = np.zeros((X.shape[0], len(self.gen1_feature_groups)))  # Store Gen-1 BDT scores
        print("Training Generation-1 XGBoost models...")

        # Train each Gen-1 sub-model and generate out-of-fold predictions
        for i, (model_name, feature_indices) in enumerate(self.gen1_feature_groups.items()):
            print(f"  Training {model_name} using features {feature_indices}...")

            # Extract features
            X_sub = self._expand_jagged_features(X[:, feature_indices])

            # Train model and generate out-of-fold predictions
            gen1_model = xgb.XGBClassifier(**self.gen1_params, use_label_encoder=False)
            # [:, 1] means we are interested in the positive class (class 1)
            oof_predictions = cross_val_predict(gen1_model, X_sub["expanded"], y[X_sub["expanded_indices"]], cv=self.n_splits, method="predict_proba")[:, 1]

            # Aggregate scores (max for jagged features)
            gen1_scores[:, i] = self._aggregate_jagged_scores(oof_predictions, X_sub["original_row_indices"])

            # Train on fully expanded dataset
            gen1_model.fit(X_sub["expanded"], y[X_sub["expanded_indices"]])
            self.gen1_models[model_name] = gen1_model

        # Identify remaining features (not in Gen-1 models)
        all_gen1_indices = set(idx for indices in self.gen1_feature_groups.values() for idx in indices)
        remaining_indices = [i for i in range(X.shape[1]) if i not in all_gen1_indices]
        X_remaining = X[:, remaining_indices]

        # Train final model with Gen-1 scores + remaining features
        X_final_train = np.hstack((X_remaining, gen1_scores))
        print(f"Training Final XGBoost classifier with Gen-1 scores + remaining features {remaining_indices}...")

        self.final_model = xgb.XGBClassifier(**self.gen2_params, use_label_encoder=False)
        self.final_model.fit(X_final_train, y)

        return self
    
    def predict(self, X):
        """
        Predict class labels using the trained dual-layer XGBoost classifier.
        """
        X_final_test = self._transform_features(X)
        return self.final_model.predict(X_final_test)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the trained dual-layer XGBoost classifier.
        """
        X_final_test = self._transform_features(X)
        return self.final_model.predict_proba(X_final_test)

    def score(self, X, y):
        """
        Evaluate the model using accuracy.
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def _expand_jagged_features(self, X_subset):
        """
        Expands jagged array features into multiple rows.
        Returns:
        - "expanded": Expanded dataset where jagged features are flattened.
        - "expanded_indices": Indices corresponding to the expanded dataset.
        - "original_row_indices": Maps original row to its expanded entries.
        """
        expanded_rows = []
        expanded_indices = []
        original_row_indices = {}

        row_counter = 0
        for i, row in enumerate(X_subset):
            if isinstance(row[0], (list, np.ndarray)):  # Check if row contains jagged arrays
                for sub_row in zip(*row):  # Expand each jagged feature
                    expanded_rows.append(sub_row)
                    expanded_indices.append(i)
                original_row_indices[i] = list(range(row_counter, row_counter + len(row[0])))
                row_counter += len(row[0])
            else:
                expanded_rows.append(row)
                expanded_indices.append(i)
                original_row_indices[i] = [row_counter]
                row_counter += 1

        return {
            "expanded": np.array(expanded_rows),
            "expanded_indices": np.array(expanded_indices),
            "original_row_indices": original_row_indices
        }

    def _aggregate_jagged_scores(self, scores, original_row_indices):
        """
        Aggregates Gen-1 model scores by taking the maximum score for jagged features.
        """
        aggregated_scores = np.zeros(len(original_row_indices))
        for i, indices in original_row_indices.items():
            aggregated_scores[i] = np.max(scores[indices])  # Take max score for jagged features
        return aggregated_scores

    def _transform_features(self, X):
        """
        Transforms input features for prediction, handling jagged features.
        """
        if not self.gen1_models or self.final_model is None:
            raise ValueError("Model is not trained yet. Call fit() first.")

        gen1_scores = np.zeros((X.shape[0], len(self.gen1_models)))

        for i, (model_name, model) in enumerate(self.gen1_models.items()):
            X_sub = self._expand_jagged_features(X[:, self.gen1_feature_groups[model_name]])
            # [:, 1] means we are interested in the positive class (class 1)
            predictions = model.predict_proba(X_sub["expanded"])[:, 1]
            gen1_scores[:, i] = self._aggregate_jagged_scores(predictions, X_sub["original_row_indices"])

        all_gen1_indices = set(idx for indices in self.gen1_feature_groups.values() for idx in indices)
        remaining_indices = [i for i in range(X.shape[1]) if i not in all_gen1_indices]
        X_remaining = X[:, remaining_indices]

        return np.hstack((X_remaining, gen1_scores))