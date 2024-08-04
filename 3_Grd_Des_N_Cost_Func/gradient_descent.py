import numpy as np
import math

def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 100000
    n = len(x)
    learning_rate = 0.08
    cost_previous = 0
    for i in range(iterations):
        # predicted value of y
        y_predicted = m_curr*x + b_curr
        # cost function MSE - mean squared error
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        if math.isclose(cost, cost_previous, rel_tol=1e-10):
            print("End loop at : ", i)
            return i
        cost_previous = cost
        # derivative of slope m
        md = -(2/n)* sum(x*(y - y_predicted))
        # derivative of intercept b
        bd = -(2/n)* sum(y - y_predicted)

        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd

        print("m {}, b {}, cost {}, iteration {}".format(m_curr, b_curr, cost, i))



x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])
gradient_descent(x, y)