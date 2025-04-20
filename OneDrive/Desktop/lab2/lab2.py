# 1. Load Dataset
import pandas as pd

df = pd.read_csv('lab2.csv')  
print("--- Dataset Info ---")
print(df.info())

# 2. Summary Statistics
print("\n--- Summary Statistics ---")
print(df.describe(include='all'))

# 3. Handle Missing or Duplicate Values
print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())

# Drop duplicate rows
df_cleaned = df.drop_duplicates()
print("\nAfter cleaning:")
print("Total rows:", len(df_cleaned))
print("Unique titles:", df_cleaned['title'].nunique())
print("Unique texts:", df_cleaned['text'].nunique())

# 4. Visualize Your Data
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram 
plt.figure(figsize=(10, 5))
sns.histplot(data=df_cleaned, x='text_length', hue='label', kde=True, palette="Set2", multiple="stack")
plt.title("Text Length Distribution by News Label")
plt.xlabel("Text Length")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Correlation Heatmap
numeric_cols = ['text_length', 'title_length', 'num_words', 'num_sentences', 'label_numeric']
correlation_matrix = df_cleaned[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Heatmap of Text Features")
plt.tight_layout()
plt.show()

# Boxplot
plt.figure(figsize=(6, 5))
sns.boxplot(x='label', y='text_length', data=df_cleaned)
plt.title("Boxplot of Text Length by News Label")
plt.xlabel("Label")
plt.ylabel("Text Length")
plt.show()

# 5. Feature Analysis
print("\n--- Average Text Length by Label ---")
print(df_cleaned.groupby('label')['text_length'].mean())
