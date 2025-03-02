import numpy as np
from sklearn.model_selection import train_test_split
from vector_xgb.vector_xgb import VectorXGBoost

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 500

# Generate meaningful scalar features
scalar_features = np.zeros((num_samples, 2))
scalar_features[:, 0] = np.random.normal(loc=0, scale=1, size=num_samples)  # Feature 1 (Gaussian)
scalar_features[:, 1] = np.random.normal(loc=2, scale=1.5, size=num_samples)  # Feature 2 (Gaussian)

# Generate class labels (binary classification)
y = np.random.randint(0, 2, num_samples)

# Generate meaningful jagged features correlated with class labels
jagged_features = np.empty(num_samples, dtype=object)
jagged_features2 = np.empty(num_samples, dtype=object)

for i in range(num_samples):
    if y[i] == 0:
        jagged_features[i] = [np.random.uniform(0, 0.5) for _ in range(np.random.randint(2, 5))]
        jagged_features2[i] = [np.random.uniform(0, 0.3) for _ in range(np.random.randint(2, 5))]
    else:
        jagged_features[i] = [np.random.uniform(0.5, 1.0) for _ in range(np.random.randint(2, 5))]
        jagged_features2[i] = [np.random.uniform(0.7, 1.0) for _ in range(np.random.randint(2, 5))]

# Combine all features
X = np.column_stack((scalar_features, jagged_features, jagged_features2))

# Define Gen-1 models with jagged features
gen1_feature_groups = {
    "sub_model1": [0, 1],  # Scalar features
    "sub_model2": [2, 3]   # Jagged features
}

# Specify which features are jagged
jagged_features_indices = [2, 3]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define model parameters
gen1_params = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1}
gen2_params = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05}

# Train the model
vector_xgb = VectorXGBoost(
    gen1_feature_groups=gen1_feature_groups,
    jagged_features=jagged_features_indices,
    gen1_params=gen1_params,
    gen2_params=gen2_params
)
vector_xgb.fit(X_train, y_train)

# Evaluate model
accuracy = vector_xgb.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
