from src.load_data import load_breast_cancer_data

df = load_breast_cancer_data()

print(df.head())
print(df.shape)
print(df.info())
