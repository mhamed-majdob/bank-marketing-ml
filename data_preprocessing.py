import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 🗂️ Load the data
data = pd.read_csv('../data/bank-full.csv', sep=';')

# 🔥 Remove leakage column
data = data.drop('duration', axis=1)

# 🔢 Label Encoding for binary columns
label_cols = ['y', 'housing', 'loan', 'default']
le = LabelEncoder()
for col in label_cols:
    data[col] = le.fit_transform(data[col])

# 🔸 One-Hot Encoding for multi-class columns
data = pd.get_dummies(data, columns=['job', 'marital', 'education', 'contact', 'month', 'poutcome'], drop_first=True)

# 🔧 Scaling numerical columns
scaler = StandardScaler()
num_cols = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
data[num_cols] = scaler.fit_transform(data[num_cols])

# 🔥 Convert boolean (True/False) to integer (1/0)
data = data.astype(int)

# ✅ Check the first rows after preprocessing
print("\n--- After Preprocessing ---")
print(data.head())

# ✅ Check the shape
print(f"Data shape after preprocessing: {data.shape}")
data.to_csv('../data/preprocessed_data.csv', index=False)
