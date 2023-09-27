# import libraries 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd 

# load data and check 
iris_df = pd.read_csv('Iris.csv', usecols=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']) #had to specify the columns as python identified NaN values for uknown reasons 
iris_df.head()

# split data into features and target
X = iris_df.drop('species', axis=1) # features/variables
y = iris_df['species'] # target

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# create kNN classifier with k
knn = KNeighborsClassifier(n_neighbors=5)

# train the model using the training sets
knn.fit(X_train, y_train)

# predict the response for test dataset     
y_pred = knn.predict(X_test)

# print accuracy score and the predicted species
print("Accuracy:", knn.score(X_test, y_test))
print("Predicted Species", y_pred)

# create a new DataFrame with the set aside dataset 
new_samples = pd.DataFrame({
    'sepal_length': [5.4, 7.2, 6.5, 6.4, 4.9, 6.5, 5.7],
    'sepal_width': [3.9, 3.6, 3.2, 2.7, 3.1, 2.8, 2.8],
    'petal_length': [1.7, 6.1, 5.1, 5.3, 1.5, 4.6, 4.5],
    'petal_width': [0.4, 2.5, 2, 1.9, 0.1, 1.5, 1.3]
})

# create a series for the correct species of the new samples
correct_species = pd.Series(['setosa', 'virginica', 'virginica', 'virginica', 'setosa', 'versicolor', 'versicolor'])

# use the trained model to predict the species
new_prediction = knn.predict(new_samples)

# compare the predicted species with the correct species
accuracy = accuracy_score(correct_species, new_prediction)

# print the predicted species and accuracy score for the new samples
print("Predicted Species for New Samples:", new_prediction)
print("Accuracy for New Samples:", accuracy)





