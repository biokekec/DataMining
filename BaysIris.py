# import libraries 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd 

# load data and check
iris_df = pd.read_csv('Iris.csv')
iris_df.head()

# split the dataset into training and testing sets
X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris_df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

# train a Gaussian Naive Bayes classifier on the training set
gnb = GaussianNB()
gnb.fit(X_train, y_train)   

# get the unique class names
class_names = iris_df['species'].unique()

# evaluate the performance of the classifier on the testing set
y_pred = gnb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# print the accuracy
print('Accuracy:', accuracy)

# create a new DataFrame with the set aside dataset 
new_samples = pd.DataFrame({
    'sepal_length': [5.4, 7.2, 6.5, 6.4, 4.9, 6.5, 5.7],
    'sepal_width': [3.9, 3.6, 3.2, 2.7, 3.1, 2.8, 2.8],
    'petal_length': [1.7, 6.1, 5.1, 5.3, 1.5, 4.6, 4.5],
    'petal_width': [0.4, 2.5, 2, 1.9, 0.1, 1.5, 1.3]
})

# create a series for the correct species of the new samples
correct_species = pd.Series(['setosa', 'virginica', 'virginica', 'virginica', 'setosa', 'versicolor', 'versicolor'])

# predict the species for the new data using the trained model
new_prediction = gnb.predict(new_samples)

# compare the predicted species with the correct species
accuracy = accuracy_score(correct_species, new_prediction)

# print the predicted species and accuracy score for the new samples 
print('Predicted Species:')
for i, species in enumerate(new_prediction):
    print('{}="{}"'.format(i, species))
print("Accuracy for New Samples:", accuracy)