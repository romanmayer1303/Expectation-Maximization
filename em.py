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


def compute_gaussian(mu, sigma_squared, x):
    return 1/(math.sqrt(2*math.pi*sigma_squared)) * math.exp(-np.power((x-mu), 2)/(2*sigma_squared))


# 2. Compute the log-likelihood
def compute_mu_a(data_points, weights):
    return np.dot(data_points, weights) / np.sum(weights)


def compute_mu_b(data_points, number_of_data_points, weights):
    ones = np.ones(number_of_data_points, dtype=np.float64)
    weights = np.subtract(ones, weights)
    try:
        return np.dot(data_points, weights) / np.sum(weights)
    except ZeroDivisionError:
        print("ZeroDivisionError")
        return 0


# call with mu_a, sigma_a_squared, p_a
def compute_weights(data_points, number_of_data_points, mu_a, sigma_a_squared, p_a):
    weights = np.zeros(number_of_data_points, dtype=np.float64)
    for i, x in enumerate(data_points):
        weights[i] = compute_gaussian(mu_a, sigma_a_squared, x) * p_a / (1/number_of_data_points)  # compute w_i
        if math.fabs(weights[i]) >= 1.0:
            print(str(i), ':')
            print(weights[i])
            print(mu_a)
            print(sigma_a_squared)
            print(x)
    return weights


def compute_sigma_a_squared(data_points, weights, mu):
    return np.dot(weights, np.power(np.subtract(data_points, mu), 2)) / sum(weights)


def compute_sigma_b_squared(data_points, number_of_data_points, weights, mu):
    ones = np.ones(number_of_data_points)
    weights_minus = ones - weights
    # print(weights)
    # print('my_test')
    # print('weights_minus: ', weights_minus)
    print('sum(weights_minus)', sum(weights_minus))
    # print('power:', np.power(np.subtract(data_points, mu), 2))
    test = np.dot(weights_minus, np.power(np.subtract(data_points, mu), 2)) / sum(weights_minus)
    # print('sigma_b_squared: ', test)
    return test


# changed the formula for log_likelihood, so that
# -400*log(2) + sum(log(gaussian(x_a) + gaussian(x_b))  =  a + b
# must be calculated.
# I can do that because p=0.5 for all data points --> log(2)*400
def compute_log_likelihood(data_points, number_of_data_points, log_likelihood_new,
                           mu_a, sigma_a_squared, mu_b, sigma_b_squared):
    log_likelihood_old = log_likelihood_new
    a = (-1)*number_of_data_points*(math.log(2))
    print(a)
    b = 0
    for data_point in data_points:
        b += math.log(compute_gaussian(mu_a, sigma_a_squared, data_point) + compute_gaussian(mu_b, sigma_b_squared,
                                                                                             data_point))
    print(b)
    log_likelihood_new = a + b
    return log_likelihood_old, log_likelihood_new


def main():
    data_points = np.asarray(read_file('data.csv'))  # convert input data to numpy array

    # initialize
    number_of_data_points = len(data_points)
    p_a, p_b = 0.5, 0.5
    mu_a, sigma_a_squared = 1., 1.
    mu_b, sigma_b_squared = 4., 1.
    log_likelihood = 0
    log_likelihood_old, log_likelihood_new = compute_log_likelihood(data_points, number_of_data_points,
                                                                    log_likelihood, mu_a, sigma_a_squared,
                                                                    mu_b, sigma_b_squared)
    print('log_likelihood_old: ', log_likelihood_old)
    print('log_likelihood_new: ', log_likelihood_new)

    i = 0
    while math.fabs(log_likelihood_old - log_likelihood_new) > 10:
        print(str(i) + ':')
        print('log_likelihood_old: ', log_likelihood_old)
        print('log_likelihood_new: ', log_likelihood_new)
        weights = compute_weights(data_points, number_of_data_points, mu_a, sigma_a_squared, p_a)
        mu_a = compute_mu_a(data_points, weights)
        sigma_a_squared = compute_sigma_a_squared(data_points, weights, mu_a)
        mu_b = compute_mu_b(data_points, number_of_data_points, weights)
        sigma_b_squared = compute_sigma_b_squared(data_points, number_of_data_points, weights, mu_b)
        log_likelihood_old, log_likelihood_new = compute_log_likelihood(data_points, number_of_data_points,
                                                                        log_likelihood, mu_a, sigma_a_squared, mu_b,
                                                                        sigma_b_squared)
        i += 1
        print()

    print('--RESULTS--')
    print('mu_a: ', mu_a)
    print('sigma_a: ', sigma_a_squared)
    print('mu_b: ', mu_b)
    print('sigma_b: ', sigma_b_squared)
    print('-----------')


main()

#    print(weights)
#    print('sum of weights: ', np.sum(weights))
#    print('mu_a: ', compute_mu_a(data_points, weights))
#    print('mu_b: ', compute_mu_b(data_points, number_of_data_points, weights))
#    print('mean(data_points): ', np.mean(data_points))

