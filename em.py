import csv
import math
import numpy as np


# 1. Read File
def read_file(filename):
    with open(filename, newline='') as csvfile:
        temp = csv.reader(csvfile)
        data_points = []
        for row in temp:
            data_points.append(float(row[0].strip(' ')))
    return data_points


def compute_gaussian(sigma, mu, x):
    return 1/(math.sqrt(2*math.pi)*sigma) * math.exp(-np.power((x-mu), 2)/(2*np.power(sigma, 2)))


def compute_gaussian2(sigma, mu, x):
    sqrt = 1/(math.sqrt(2*math.pi)*sigma)
    exp = math.exp(-np.power((x-mu), 2)/(2*np.power(sigma, 2)))
    return sqrt * exp


# 2. Compute the log-likelihood
def compute_log_likelihood():
    pass


def compute_mu_a():
    return np.dot(data_points, weights) / np.sum(weights)


def compute_mu_b():
    ones = np.ones(number_of_data_points)
    try:
        return np.dot(data_points, 1-weights) / np.sum(1-weights)
    except ZeroDivisionError:
        return 0


data_points = np.asarray(read_file('data.csv'))  # convert input data to numpy array
number_of_data_points = len(data_points)
#weights = np.ones(number_of_data_points)  # initialize weights = 1
weights = np.zeros(number_of_data_points, dtype=float)

p_a, p_b = 0.5, 0.5
mu_a, sigma_a = 1., 1.
mu_b, sigma_b = 4., 1.


def calculate_weights():
    for i, x in enumerate(data_points):
        weights[i] = compute_gaussian(mu_a, sigma_a, x) * p_a / (1/number_of_data_points)  # compute w_i


calculate_weights()
print(weights)
print(np.sum(weights))
print('mu_a: ', compute_mu_a())
print('mu_b: ', compute_mu_b())
print('mean(data_points): ', np.mean(data_points))
