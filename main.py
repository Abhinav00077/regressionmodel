import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

breastcancer=datasets.load_breast_cancer()
print(breastcancer.keys())
#['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module']
print(breastcancer.DESCR)
breastcancer_x=breastcancer.data[:, np.newaxis ,2]# i did slicing to plot a graph using matplotlib

breastcancer_x_train=breastcancer_x[:-30]
breastcancer_x_test=breastcancer_x[-30:]

breastcancer_y_train=breastcancer.target[: -30]
breastcancer_y_test=breastcancer.target[-30 :]

model=linear_model.LinearRegression()
model.fit(breastcancer_x_train,breastcancer_y_train)

breastcancer_y_predicted=model.predict(breastcancer_x_test)
print("mean squared error:", mean_squared_error(breastcancer_x_test,breastcancer_y_predicted))
print("weights: ",model.coef_)
print("intercept: ",model.intercept_)

plt.scatter(breastcancer_x_test,breastcancer_y_test)
plt.plot(breastcancer_x_test,breastcancer_y_predicted)
plt.show()





