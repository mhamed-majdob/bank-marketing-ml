import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

# Load the preprocessed data
data = pd.read_csv('../data/preprocessed_data.csv')

# Separate features and target
X = data.drop('y', axis=1)
y = data['y']

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

# Plot
sorted_idx = np.argsort(importances)

plt.figure(figsize=(10, 12))
plt.barh(range(len(importances)), importances[sorted_idx], align='center')
plt.yticks(range(len(importances)), [feature_names[i] for i in sorted_idx])
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# Print top 10 features
for f in range(10):
    print(f"{f + 1}. {feature_names[indices[f]]} ({importances[indices[f]]:.4f})")
