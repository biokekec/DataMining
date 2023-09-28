# import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  # import PCA

# load the data set
iris_df = pd.read_csv('Iris.csv')

# GRAPH ONE
# create the pairplot with different colors for different species
sns.pairplot(iris_df, hue='species', vars=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
plt.suptitle('Pairplot of Iris Dataset', y=1.02)
plt.show()


# GRAPH TWO
# Create new columns for combined variables
iris_df['width'] = iris_df['sepal_width'] + iris_df['petal_width']
iris_df['length'] = iris_df['sepal_length'] + iris_df['petal_length']

# Create a scatter plot with combined variables
plt.figure(figsize=(8, 6))  # Create a new figure
sns.scatterplot(data=iris_df, x='width', y='length', hue='species')

# set labels
plt.xlabel('Width (Sepal + Petal)')
plt.ylabel('Length (Sepal + Petal)')
plt.title('Clustered Scatterplot')

# GRAPH THREE
# perform PCA
X_reduced = PCA(n_components=3).fit_transform(iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])

fig = plt.figure(figsize=(8, 6))  # create a new figure for the 3D scatter plot
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

# color code the different species
species_to_color = {'setosa': 'blue', 'versicolor': 'orange', 'virginica': 'green'}
colors = iris_df['species'].map(species_to_color)

# scatter plot for each species
for species, color in species_to_color.items():
    indices = iris_df['species'] == species
    ax.scatter(
        X_reduced[indices, 0],
        X_reduced[indices, 1],
        X_reduced[indices, 2],
        c=color,
        s=40,
        label=species  # set label for legend
    )
# set labels + legend
ax.set_title("3D Display")
ax.set_xlabel("1st Eigenvector")
ax.set_ylabel("2nd Eigenvector")
ax.set_zlabel("3rd Eigenvector")
ax.legend()
plt.show()
