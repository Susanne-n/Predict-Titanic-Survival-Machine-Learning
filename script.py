import codecademylib3_seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
passengers = pd.read_csv('passengers.csv')
print(passengers.head())

# Update sex column to numerical
passengers['Sex'] = passengers['Sex'].map({'male': 0, 'female': 1})

# Fill the nan values in the age column
passengers['Age'].fillna(value=passengers.Age.mean(), inplace=True)

# Create a first class column
passengers['FirstClass'] = passengers.Pclass.apply(lambda x: 1 if x == 1 else 0)

# Create a second class column
passengers['SecondClass'] = passengers.Pclass.apply(lambda x: 1 if x == 2 else 0)

# Select the desired features for the model
features = passengers[['Sex', 'Age', 'FirstClass', 'SecondClass']]
survival = passengers['Survived']

# Split the data into training and test sets
train_features, test_features, train_labels, test_labels = train_test_split(features, survival, train_size=0.8, test_size=0.2)

# Scale the feature data so it has mean = 0 and standard deviation = 1
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# Create and evaluate the model
model = LogisticRegression()
model.fit(train_features, train_labels)

print(model.score(train_features, train_labels))

accuracy = model.score(test_features, test_labels)
print(accuracy)
# accuracy of the model seems to be around 80%

# Analyze the coefficients
coef = model.coef_
print(coef)
print(list(zip(['Sex', 'Age', 'FirstClass', 'SecondClass'],model.coef_[0])))
# the attribute Sex seems to be most important in predicting survival

# Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
Susanne = np.array([1.0,27,0.0,1.0])

# Preparing the passenger features
sample_passengers = np.array([Jack, Rose, Susanne])
sample_passengers = scaler.transform(sample_passengers)

# Making survival predictions
sample_prediction = model.predict(sample_passengers)
print(sample_prediction)

sample_probabilities = model.predict_proba(sample_passengers)
print(sample_probabilities)
# it was predicted that Jack won't survive, with a probability of around 88% of not surviving
# it was predicted that Rose will survive, with a probability of around 94% of surviving
# it was predicted that Susanne (me) will survive, with a probability of around 81% of surviving
