import csv
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Generate the training set
with open('gpg.csv') as csvfile:

    reader = csv.reader(csvfile)

    next(reader, None)  # skips the header

    year = []
    GPG = []

    for row in reader:

        year.append(float(row[1]))
        GPG.append(float(row[4]))

#plt.plot(year, GPG, 'bx')  # plotting the original data


# Had to define this function to return the values of year squared
def square(a):

    squares = []

    for i in a:

        squares.append(i ** 2)

    return squares

linear_year_train = np.transpose([np.ones(72), year])  #linear
quadratic_year_train = np.transpose([np.ones(72), year, square(year)]) #quadratic
linear_theta = inv(np.transpose(linear_year_train) @ linear_year_train) @ np.transpose(linear_year_train) @ GPG
quadratic_theta = inv(np.transpose(quadratic_year_train) @ quadratic_year_train) @ np.transpose(quadratic_year_train) @ GPG

x_predict = np.arange(1930, 2040, 1)
linear_goal_predict = linear_theta[0] + (linear_theta[1] * x_predict)
quadratic_goal_predict = quadratic_theta[0] + (quadratic_theta[1] * x_predict) + (quadratic_theta[2] * (x_predict ** 2))
plt.figure()
plt.plot(year, GPG, 'bx')
plt.plot(x_predict, linear_goal_predict, 'y-')  # plotting the prediction data
plt.ylabel('Goals Per Game')
plt.xlabel('Year')
plt.figure()
plt.plot(year, GPG, 'bx')
plt.plot(x_predict, quadratic_goal_predict, 'y-')
plt.ylabel('Goals Per Game')
plt.xlabel('Year')
plt.show()

