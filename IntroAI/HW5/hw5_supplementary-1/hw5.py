import sys
import csv
import matplotlib.pyplot as plt
import numpy as np


def plot_data(filepath):
    with open(filepath, newline='') as csvfile:
        data = list(csv.DictReader(csvfile))
    csvfile.close()
    year = []
    days = []
    for i in range(len(data)):
        year.append(int(data[i]['year']))
        days.append(int(data[i]['days']))

    # plot
    plt.figure(figsize=(8, 6))
    plt.plot(year, days)
    plt.xlabel("Year")
    plt.ylabel("Number of Frozen Days")
    plt.savefig("plot.jpg")
    return year, days


def q3a(year):
    X = []
    for i in range(len(year)):
        X.append([1, year[i]])
    X = np.array(X)
    print("Q3a:")
    print(X)
    return X


def q3b(days):
    Y = np.array(days)
    print("Q3b:")
    print(Y)
    return Y


def q3c(X):
    Z = np.dot(X.T, X)
    print("Q3c:")
    print(Z)
    return Z


def q3d(Z):
    I = np.linalg.inv(Z)
    print("Q3d:")
    print(I)
    return I


def q3e(I, X):
    PI = np.dot(I, np.transpose(X))
    print("Q3e:")
    print(PI)
    return PI


def q3f(PI, Y):
    beta = np.dot(PI, Y)
    print("Q3f:")
    print(beta)
    return beta


def q4(beta, x_test):
    y_test = np.dot(beta.T, np.array([1, x_test]))
    print("Q4: " + str(y_test))


def q5a(beta):
    symbol = "="
    if beta[1] < 0:
        symbol = "<"
    elif beta[1] > 0:
        symbol = ">"
    print("Q5a: " + symbol)


def q5b():
    print("Q5b: If sign == '=', it means the year won't influence the number of frozen days, the number of days is "
          "fixed. If sign == '<', it means the larger the value of the year, the smaller the number of frozen days. "
          "If sign == '>', it means the larger the value of the year, the bigger the number of frozen days.")


def q6a(beta):
    x_star = -beta[0]/beta[1]
    print("Q6a: " + str(x_star))


def q6b():
    print("Q6b: This result is in line with expectations. The sign '<' means the larger the value of the year, "
          "the smaller the number of frozen days. The predicted number of days in 2022 is much bigger than 0, "
          "and the expected 'Y' is 0, so the expected year should be larger than 2022. So the calculation result 2455 "
          "is as expected.")


if __name__ == "__main__":
    year, days = plot_data(sys.argv[1])
    X = q3a(year)
    Y = q3b(days)
    Z = q3c(X)
    I = q3d(Z)
    PI = q3e(I, X)
    beta = q3f(PI, Y)
    q4(beta, 2022)
    q5a(beta)
    q5b()
    q6a(beta)
    q6b()

