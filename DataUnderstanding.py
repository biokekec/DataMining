# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data set & check
iris_df = pd.read_csv('Iris.csv')

# Create new columns for combined variables
iris_df['width'] = iris_df['sepal_width'] + iris_df['petal_width']
iris_df['length'] = iris_df['sepal_length'] + iris_df['petal_length']

# Create a scatter plot with combined variables
sns.scatterplot(data=iris_df, x='width', y='length', hue='species')

# Set axis labels
plt.xlabel('Width (Sepal + Petal)')
plt.ylabel('Length (Sepal + Petal)')
plt.title('Clustered Scaterplot')

# Create the pairplot with different colors for different species
sns.pairplot(iris_df, hue='species', vars=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
plt.suptitle('Pairplot of Iris Dataset', y=1.02)
plt.show()
