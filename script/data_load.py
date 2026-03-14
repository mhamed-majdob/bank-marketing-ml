# Import the pandas library for data handling
import pandas as pd

# Load the CSV file
data = pd.read_csv('../data/bank-full.csv', sep=';')  # Note: sep=';' because the CSV uses semicolons

# Show the first 5 rows to check if it worked
print(data.head())

# Check the number of rows and columns
print(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

# Basic info about the dataset
print("\n--- Dataset Info ---")
print(data.info())

# Check for missing values
print("\n--- Missing Values ---")
print(data.isnull().sum())

# Check the distribution of the target variable 'y'
print("\n--- Target Variable Distribution ---")
print(data['y'].value_counts())
print("\n--- Target Variable Distribution (Percent) ---")
print(data['y'].value_counts(normalize=True) * 100)
print("\n--- Summary Statistics ---")
print(data.describe(include='all'))
import matplotlib.pyplot as plt
import seaborn as sns

# Plot target variable distribution
sns.countplot(data=data, x='y')
plt.title('Target Variable Distribution (y)')
plt.xlabel('Subscribed to Term Deposit')
plt.ylabel('Count')
plt.show()
# Example: Check job distribution
sns.countplot(data=data, x='job')
plt.title('Job Distribution')
plt.xticks(rotation=45)
plt.show()

