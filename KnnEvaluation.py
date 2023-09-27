# impor the libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# load the Iris dataset
iris_df = pd.read_csv('Iris.csv', usecols=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

# create new columns for combined variables
iris_df['width'] = iris_df['sepal_width'] + iris_df['petal_width']
iris_df['length'] = iris_df['sepal_length'] + iris_df['petal_length']

# define parameter ranges
test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]  # different test sizes
random_states = [0, 42, 123]  # different random states
n_neighbors_values = [1, 3, 5, 7, 9]  # different n_neighbors values

#  initialized with a value of -1 
best_accuracy = -1

# create an empty dictionary to store the parameters 
best_params = {}

# create a grid of subplots for parameter variations
fig, axes = plt.subplots(len(test_sizes), len(random_states), figsize=(15, 12), sharex='col', sharey='row')

# loop through parameter combinations and plot accuracy
for i, test_size in enumerate(test_sizes):
    for j, random_state in enumerate(random_states):
        accuracy_scores = []
        for n_neighbors in n_neighbors_values:
            # split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(iris_df.drop('species', axis=1), iris_df['species'], test_size=test_size, random_state=random_state)

            # create kNN classifier
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)

            # train the model using the training sets
            knn.fit(X_train, y_train)

            # calculate accuracy
            accuracy = knn.score(X_test, y_test)
            accuracy_scores.append(accuracy)

            # check if this combination leads to the best accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {
                    'test_size': test_size,
                    'random_state': random_state,
                    'n_neighbors': n_neighbors,
                    'accuracy': accuracy
                }

        # plot the accuracy scores for this combination
        ax = axes[i, j]
        ax.plot(n_neighbors_values, accuracy_scores, marker='o')
        ax.set_title(f'Test Size = {test_size}, Random State = {random_state}')
        ax.set_xlabel('n_neighbors')
        ax.set_ylabel('Accuracy')
        ax.grid(True)

# add common labels
plt.suptitle('Accuracy vs. n_neighbors for Different test_size and random_state', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# print the best parameters and accuracy
print("Best Parameters:", best_params)

# Show the plot
plt.show()
