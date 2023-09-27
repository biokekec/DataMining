# load the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# load the Iris dataset
iris_df = pd.read_csv('Iris.csv')

# define X and y based on the dataset columns
X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris_df['species']

# create lists to store accuracy scores and parameter values
accuracy_scores = []
random_states = range(1, 101)  # added an range (possible to change)
test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]  # added a lsit (possible to change)
best_accuracy = -1 # initialized with a value of -1 
best_test_size = None # initialized with None as these values are yet to be determined 
best_random_state = None

# create subplots
fig, axs = plt.subplots(len(test_sizes), figsize=(10, 12))

# iterate through different test_size values
for i, test_size in enumerate(test_sizes):
    accuracy_values = []  # store accuracy values for this test_size
    
    for random_state in random_states:
        # split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # train a Gaussian Naive Bayes classifier on the training set
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)

        # evaluate the performance of the classifier on the testing set
        y_pred = gnb.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_values.append(accuracy)

        # check if this accuracy is the highest so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_test_size = test_size
            best_random_state = random_state

    # plot accuracy values for this test_size on a subplot
    axs[i].plot(random_states, accuracy_values, label=f'Test Size {test_size}')
    axs[i].set_title(f'Test Size {test_size}')
    axs[i].set_xlabel('Random State')
    axs[i].set_ylabel('Accuracy')
    axs[i].grid(True)
    axs[i].legend()

# print the best test_size, random_state, and accuracy
print(f"Best Test Size: {best_test_size}")
print(f"Best Random State: {best_random_state}")
print(f"Highest Accuracy: {best_accuracy}")

# adjust layout
plt.tight_layout()
plt.show()
