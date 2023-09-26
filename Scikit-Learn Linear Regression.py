import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model as lm
import numpy

X = np.linspace(0,6,60)
F = -3 * X + 2
Y = F + 1 * np.random.rand(len(X))

regr = lm.LinearRegression()
regr.fit(X.reshape(-1,1), Y)

print("Coefficients: ", regr.coef_)
print("Intercept: ", regr.intercept_)

eq = X.reshape(-1, 1) * regr.coef_ + regr.intercept_

plt.plot(X, Y, 'ro', label = 'Datapoints')
plt.plot(X, eq, label = 'Model')
plt.legend()
plt.show()

loss = numpy.mean((eq - Y) ** 2)
print(loss)