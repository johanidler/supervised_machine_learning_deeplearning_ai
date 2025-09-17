# import math, copy
import numpy as np

data = np.loadtxt(
    "./data.csv",
    delimiter=",",
    skiprows=1,
)


def get_value(index: int, value):
    if value == "x":
        return data[index][0]
    elif value == "y":
        return data[index][1]
    else:
        return None


def get_m():
    return len(data)


def calculate_y_hat(x, w, b):
    y_hat = w * x + b
    return y_hat


# we square not just to get a positive number, that can't cancel out, but also to ensure large mistakes are treated as even worse
def calculate_one_cost(x, y, w, b):
    y_hat = calculate_y_hat(x, w, b)
    # cost = abs(y_hat - y) #small mistakes dont matter so much
    cost = (y_hat - y) * (y_hat - y)  # mistakes are a big problem
    return cost


# our J
def compute_total_cost(w, b):
    m = get_m()
    cost = 0
    for i in range(m):
        cost = cost + calculate_one_cost(get_value(i, "x"), get_value(i, "y"), w, b)

    cost = 1 / (2 * m) * cost
    return cost


def gradient_descent(w, b):
    m = get_m()
    alpha = 0.1
    iterations = 500
    for g in range(iterations):
        # compute gradients
        dj_dw = 0.0
        dj_db = 0.0
        for i in range(m):
            x_i = get_value(i, "x")
            y_i = get_value(i, "y")
            y_prediction = calculate_y_hat(x_i, w, b)
            err = y_prediction - y_i
            dj_dw = dj_dw + err * x_i
            dj_db = dj_db + err
        dj_dw /= get_m()
        dj_db /= get_m()

        # update
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if g % 10 == 0:
            print(f"Iteration: {g} -> w = {w}, b = {b}")

    return w, b


# print(data)

# print(get_value(2, "x"))
# print(get_m())

w = 0.8
b = 9.6

# x_1 = 8
# y_1 = 16

# print("y^ = ", calculate_y_hat(1, w, b))
# print("y = 11")
# print("cost = ", calculate_one_cost(1, 11, w, b))

# print(compute_total_cost(w, b))


print(gradient_descent(5, 9))
