from os import scandir
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



#Question 4

# i) generate samples

X = np.random.normal(loc=0, scale=10,size=150)
# print(X)

# ii) design matrix for features 1 x and x^2

design_matrix = np.zeros((len(X), 3), dtype=float)

#set first col to ones

design_matrix[:, 0] = 1

x_func = lambda x: x
x_squared_func = lambda x: x**2
x_cubed_func = lambda x: x**3

#set second col
for i, x in enumerate(X):
    design_matrix[i, 1] = x_func(x)

#set last col
for i, x in enumerate(X):
    design_matrix[i, 2] = x_squared_func(x)

# print(design_matrix)
#iii)
theta_0 = np.random.uniform()
theta_1 = np.random.uniform()
theta_2 = np.random.uniform()

thetas = np.array([theta_0, theta_1, theta_2])

# print(thetas)

#iv)
cross_of_x = np.matmul(design_matrix.T, design_matrix)
moore_penrose_inverse = np.linalg.pinv(design_matrix)
cross_with_x_inverse = np.dot(cross_of_x, moore_penrose_inverse)
Y = np.dot(cross_with_x_inverse.T, thetas)


gaussian_noise = np.array([np.random.normal(loc=0, scale=8) for i in range(len(Y))])

Y = Y + gaussian_noise

# print(Y)

#v)

# plt.scatter(X, Y)
# plt.show()

#vi)

X_train, X_test, y_train, y_test = train_test_split(X, Y,  test_size=.2)

# print(X_train, X_test, y_train, y_test)


#b)
# i)
#
#set new design matrix on train data
#set second col
for i, x in enumerate(X_train):
    design_matrix[i, 1] = x_func(x)

#set last col
for i, x in enumerate(X_train):
    design_matrix[i, 2] = x_squared_func(x)

weights = np.dot(np.dot(np.linalg.pinv(np.matmul(design_matrix.T, design_matrix)), design_matrix.T),Y)
# print(weights)

#ii)

difference = thetas - weights
percentage_difference = [np.abs(difference[i]/thetas[i])*100 for i in range(len(thetas))]

# print(difference)


f = lambda x:weights[0] +  weights[1]*x_func(x) + weights[2]*x_squared_func(x)

#iii)
#we will first evaluate the training set
inferences = [f(x) for x in X_train]
square_mean_error = 0
for i in range(len(inferences)):
    square_mean_error += (y_train[i] - inferences[i])**2
square_mean_error *= 1/len(inferences)
# print(square_mean_error)



#and now the validation set

inferences = [f(x) for x in X_test]
square_mean_error = 0
for i in range(len(inferences)):
    square_mean_error += (y_test[i] - inferences[i])**2
square_mean_error *= 1/len(inferences)
# print(square_mean_error)


#iv)
data = np.stack((X_train, y_train))
idx = np.argsort(X_train)

y_points = [f(x) for x in X_train[idx]]

# plt.plot(X_train[idx],y_points, color="red")
# plt.scatter(X_train[idx], y_points)
# plt.show()


#v)

alpha = 1e-6


#init weights
theta_0 = np.random.uniform()
theta_1 = np.random.uniform()
theta_2 = np.random.uniform()

new_weights = np.zeros(3)
f_dynamic = lambda x, w: w[0] +  w[1]*x_func(x) + w[2]*x_squared_func(x)
error_x = []
errors = []
for epoch in range(101):
    for j, x in enumerate(X_train):
        for i in range(len(new_weights)):
            new_weights[i] = new_weights[i] - alpha*(f_dynamic(x, new_weights) - y_train[j])*design_matrix[j,i]
    if epoch % 20 == 0:
        inferences = [f_dynamic(x, new_weights) for x in X_train]
        square_mean_error = 0
        for i in range(len(inferences)):
            square_mean_error += (y_train[i] - inferences[i])**2
        square_mean_error *= 1/len(inferences)
        error_x.append(epoch)
        errors.append(square_mean_error)

plt.plot(error_x, errors)
plt.show()

#c)
#i)

#set first col to ones

design_matrix = np.zeros((len(X), 4), dtype=float)

design_matrix[:, 0] = 1

x_func = lambda x: x
x_squared_func = lambda x: x**2
x_cubed_func = lambda x: x**3

#set second col
for i, x in enumerate(X):
    design_matrix[i, 1] = x_func(x)

#set last col
for i, x in enumerate(X):
    design_matrix[i, 2] = x_squared_func(x)


#set last col
for i, x in enumerate(X):
    design_matrix[i, 3] = x_cubed_func(x)

# print(design_matrix)
f_new = lambda x, w: w[0] + w[1]*x_func(x) + w[2]*x_squared_func(x) + w[3]*x_cubed_func(x)


# ii)

alpha = 1e-6


#init weights
theta_0 = np.random.uniform()
theta_1 = np.random.uniform()
theta_2 = np.random.uniform()
theta_3 = np.random.uniform()

new_weights = np.array([theta_0, theta_1, theta_2, theta_3])
error_x = []
errors = []
for epoch in range(101):
    for j, x in enumerate(X_train):
        for i in range(len(new_weights)):
            new_weights[i] = new_weights[i] - alpha*(f_new(x, new_weights) - y_train[j])*design_matrix[j,i]
    if epoch % 20 == 0:
        inferences = [f_new(x, new_weights) for x in X_train]
        square_mean_error = 0
        for i in range(len(inferences)):
            square_mean_error += (y_train[i] - inferences[i])**2
        square_mean_error *= 1/len(inferences)
        error_x.append(epoch)
        errors.append(square_mean_error)

# plt.plot(error_x, errors)
# plt.show()
