import pandas as pd

# Load preprocessed data
data = pd.read_csv('../data/preprocessed_data.csv')

# Separate features and target
X = data.drop('y', axis=1)
y = data['y']
from sklearn.model_selection import train_test_split

# First split: Train (70%) and Temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)

# Second split: Temp (30%) → Validation (15%) and Test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# Check shapes
print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")
##########SVM MODEL#######
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Initialize the SVM model
svm_model = SVC(kernel='rbf', random_state=42)

# Train the model
svm_model.fit(X_train, y_train)

# Predict on validation set
y_val_pred = svm_model.predict(X_val)

# Evaluate on validation set
accuracy = accuracy_score(y_val, y_val_pred)
print(f"SVM Validation Accuracy: {accuracy:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_val, y_val_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Detailed Classification Report
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))
##########Naive Bayes Model#######
from sklearn.naive_bayes import GaussianNB

# Initialize Naive Bayes
nb_model = GaussianNB()

# Train the model
nb_model.fit(X_train, y_train)

# Predict on validation set
y_val_pred_nb = nb_model.predict(X_val)

# Evaluate
accuracy_nb = accuracy_score(y_val, y_val_pred_nb)
print(f"\nNaive Bayes Validation Accuracy: {accuracy_nb:.4f}")

# Confusion Matrix
conf_matrix_nb = confusion_matrix(y_val, y_val_pred_nb)
print("\nConfusion Matrix (Naive Bayes):")
print(conf_matrix_nb)

# Classification Report
print("\nClassification Report (Naive Bayes):")
print(classification_report(y_val, y_val_pred_nb))
###########Random Forest###########
from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on validation set
y_val_pred_rf = rf_model.predict(X_val)

# Evaluate
accuracy_rf = accuracy_score(y_val, y_val_pred_rf)
print(f"\nRandom Forest Validation Accuracy: {accuracy_rf:.4f}")

# Confusion Matrix
conf_matrix_rf = confusion_matrix(y_val, y_val_pred_rf)
print("\nConfusion Matrix (Random Forest):")
print(conf_matrix_rf)

# Classification Report
print("\nClassification Report (Random Forest):")
print(classification_report(y_val, y_val_pred_rf))
#########TEST########
# ----- Final Test Evaluation with Random Forest -----
print("\n--- Final Test Evaluation (Random Forest) ---")

# Predict on test set
y_test_pred = rf_model.predict(X_test)

# Evaluate
accuracy_test = accuracy_score(y_test, y_test_pred)
print(f"Random Forest Test Accuracy: {accuracy_test:.4f}")

# Confusion Matrix
conf_matrix_test = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix (Test):")
print(conf_matrix_test)

# Classification Report
print("\nClassification Report (Test):")
print(classification_report(y_test, y_test_pred))
