import pandas as pd

# Load dataset
df = pd.read_csv('data/raw/creditcard.csv')

# Print basic info
print(f"✅ Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Fraud cases: {df['Class'].sum()}")
print(f"Normal cases: {(df['Class']==0).sum()}")
print(f"\nFirst few rows:")
print(df.head())
