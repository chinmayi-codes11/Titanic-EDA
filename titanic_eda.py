import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Load the Titanic dataset
df = pd.read_csv('train.csv')

# Show first 5 rows to verify
print("First 5 rows of data:")
print(df.head())

# Show summary info and data types
print("\nData info:")
print(df.info())

# Check missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked with most frequent value
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column because many missing
df.drop(['Cabin'], axis=1, inplace=True)

# Survival counts plot
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title('Survival Counts')
plt.savefig('survival_counts.png')
plt.close()

# Survival rate by Gender
plt.figure(figsize=(6,4))
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Gender')
plt.savefig('gender_survival.png')
plt.close()

# Survival rate by Passenger Class
plt.figure(figsize=(6,4))
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Passenger Class')
plt.savefig('class_survival.png')
plt.close()

# Age distribution histogram
plt.figure(figsize=(8,5))
df['Age'].hist(bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('age_distribution.png')
plt.close()

# Boxplot of Age by Survival
plt.figure(figsize=(6,4))
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Age Distribution by Survival')
plt.savefig('age_survival_boxplot.png')
plt.close()

# Correlation heatmap (only numeric columns)
numeric_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()

# --- Advanced Statistical Tests and Visualizations ---

# Chi-Square Test: Survival vs Gender
contingency_table = pd.crosstab(df['Sex'], df['Survived'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Chi-Square Test between Sex and Survived: chi2={chi2:.2f}, p-value={p:.4f}")

# T-Test: Age difference between survived and non-survived
survived_ages = df[df['Survived'] == 1]['Age']
not_survived_ages = df[df['Survived'] == 0]['Age']
t_stat, p_val = stats.ttest_ind(survived_ages, not_survived_ages, nan_policy='omit')
print(f"T-Test for Age difference: t-statistic={t_stat:.2f}, p-value={p_val:.4f}")

# FacetGrid: Age distribution by Survival and Gender
g = sns.FacetGrid(df, col='Survived', row='Sex', height=4)
g.map(plt.hist, 'Age', bins=20)
plt.savefig('age_facetgrid.png')
plt.close()

# Pairplot of numerical features colored by Survival
pairplot = sns.pairplot(df[['Age', 'Fare', 'Pclass', 'Survived']], hue='Survived')
pairplot.savefig('pairplot.png')
plt.close()

# Violin plot: Fare distribution by Survival
plt.figure(figsize=(6,4))
sns.violinplot(x='Survived', y='Fare', data=df)
plt.title('Fare Distribution by Survival')
plt.savefig('fare_violinplot.png')
plt.close()

# Heatmap of missing values
plt.figure(figsize=(8,6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.savefig('missing_data_heatmap.png')
plt.close()
