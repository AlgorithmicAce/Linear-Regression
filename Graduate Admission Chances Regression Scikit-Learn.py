import pandas as pd
import numpy as np
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv('Admission_Predict.csv')

X = df[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']]
Y = df[['Chance of Admit']]

msk = np.random.rand(len(X)) < 0.75
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)
print(len(X_train))

regr = lm.LinearRegression()
np.reshape(X_train, (-1, 1))
regr.fit(X_train, Y_train)

print("The coefficent of the model is", regr.coef_)
print("The intercept of the model is", regr.intercept_)

np.reshape(X_test, (-1, 1))
Y_hat = regr.predict(X_test)
loss = mean_squared_error(Y_test, Y_hat)
print("The loss of the model is", loss)