import random
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import os
import json


def rastrigin_experiments(symbols: list):
    f = rastrigin_for_d2(2, symbols)
    rastrigin_domain = [-5.12, 5.12]

    learning_rates = [0.01, 0.001, 0.0001]
    experiments_data = []
    for lr in learning_rates:
        x0 = [0.2, 0.3]
        starting_point = x0
        steps, x0, mse, abs_err = calc_gradient_descent(
            symbols=symbols,
            f=f,
            x0=x0,
            epsilon=0.001,
            max_iter=10000,
            domain=rastrigin_domain,
            learning_rate=lr,
        )
        steps_first10 = steps[:10]
        steps_last10 = steps[-10:]
        exp_data = {
            "function_name": "rastrigin",
            "learning_rate": lr,
            "starting_point": starting_point,
            "steps_first10": steps_first10,
            "steps_last10": steps_last10,
            "best_x": x0,
            "mse": mse,
            "abs_err": abs_err,
        }
        experiments_data.append(exp_data)
        params = [lr, starting_point, "rastrigin"]
        plot_gradient_descent(steps, f, symbols, params)

    starting_points = [[-0.2, 0.15], [0.45, 0.5], [2.7, 3.4]]
    for i, x0 in enumerate(starting_points):
        lr = 0.001
        steps, x0, mse, abs_err = calc_gradient_descent(
            symbols=symbols,
            f=f,
            x0=x0,
            epsilon=0.001,
            max_iter=10000,
            domain=rastrigin_domain,
            learning_rate=lr,
        )
        steps_first10 = steps[:10]
        steps_last10 = steps[-10:]
        exp_data = {
            "function_name": "rastrigin",
            "learning_rate": lr,
            "starting_point": starting_point,
            "steps_first10": steps_first10,
            "steps_last10": steps_last10,
            "best_x": x0,
            "mse": mse,
            "abs_err": abs_err,
        }
        experiments_data.append(exp_data)
        params = [lr, starting_points[i], "rastrigin"]
        plot_gradient_descent(steps, f, symbols, params)

    cwd = os.getcwd()
    with open(f"{cwd}/lab1/rastrigin_exp.json", "w") as json_file:
        json.dump({"rastrigin_experiments": experiments_data}, json_file, indent=2)


def griewanka_experiments(symbols: list):
    f = griewanka_for_d2(2, symbols)
    griewanka_domain = [-5, 5]
    # x0 = generate_starting_point(rastrigin_domain)
    learning_rates = [0.1, 0.01, 0.001]
    experiments_data = []
    for lr in learning_rates:
        x0 = [0.2, 0.3]
        starting_point = x0
        steps, x0, mse, abs_err = calc_gradient_descent(
            symbols=symbols,
            f=f,
            x0=x0,
            epsilon=0.001,
            max_iter=10000,
            domain=griewanka_domain,
            learning_rate=lr,
        )
        steps_first10 = steps[:10]
        steps_last10 = steps[-10:]
        exp_data = {
            "function_name": "griewanka",
            "learning_rate": lr,
            "starting_point": starting_point,
            "steps_first10": steps_first10,
            "steps_last10": steps_last10,
            "best_x": x0,
            "mse": mse,
            "abs_err": abs_err,
        }
        experiments_data.append(exp_data)
        params = [lr, starting_point, "griewanka"]
        plot_gradient_descent(steps, f, symbols, params)

    cwd = os.getcwd()
    with open(f"{cwd}/lab1/griewank_exp.json", "w") as json_file:
        json.dump({"griewank_experiments": experiments_data}, json_file, indent=2)

    starting_points = [[-0.9, 0.85], [-3.45, 2.5], [4.5, 4.8]]
    lr = 0.1
    for i, x0 in enumerate(starting_points):
        steps, x0, mse, abs_err = calc_gradient_descent(
            symbols=symbols,
            f=f,
            x0=x0,
            epsilon=0.001,
            max_iter=10000,
            domain=griewanka_domain,
            learning_rate=lr,
        )
        steps_first10 = steps[:10]
        steps_last10 = steps[-10:]
        exp_data = {
            "function_name": "griewanka",
            "learning_rate": lr,
            "starting_point": starting_point,
            "steps_first10": steps_first10,
            "steps_last10": steps_last10,
            "best_x": x0,
            "mse": mse,
            "abs_err": abs_err,
        }
        experiments_data.append(exp_data)
        params = [lr, starting_points[i], "griewanka"]
        plot_gradient_descent(steps, f, symbols, params)


def x_squared_experiments(symbols: list):
    x1, x2 = symbols
    f = x1**2 + x2**2

    starting_point = [4, 4]
    steps, x0, mse, abs_err = calc_gradient_descent(
        symbols=symbols,
        f=f,
        x0=starting_point,
        epsilon=0.001,
        max_iter=10000,
        domain=[-50, 50],
    )
    params = [0.1, x0, "x_squared"]
    plot_gradient_descent(steps, f, symbols, params)


def calc_gradient(f, symbols):
    gradient = [sp.diff(f, symbol) for symbol in symbols]
    gradient_func = [sp.lambdify(symbols, diff) for diff in gradient]
    return gradient_func


def rastrigin_for_d2(d: int, symbols: list):
    x1, x2 = symbols
    return (
        20
        + x1**2
        - 10 * sp.cos(2 * np.pi * x1)
        + x2**2
        - 10 * sp.cos(2 * np.pi * x2)
    )


def griewanka_for_d2(d: int, symbols: list):
    x1, x2 = symbols
    return (
        1
        + x1**2 / 4000
        + x2**2 / 4000
        - (sp.cos(x1 / sp.sqrt(1)) * sp.cos(x2 / sp.sqrt(2)))
    )


def generate_starting_point(domain: list, d: int = 2):
    point = []
    for i in range(d):
        point.append(random.uniform(domain[0], domain[1]))
    return point


def calc_gradient_descent(
    symbols, f, x0, epsilon, max_iter, domain, learning_rate=0.001
):
    steps = [x0]
    gradient_func = calc_gradient(f, symbols)
    mse = []
    absolute_err = []

    for _ in range(max_iter):
        gradient = [func(*x0) for func in gradient_func]
        diff = [learning_rate * grad for grad in gradient]

        if all(abs(d) < epsilon for d in diff):
            break

        predicted_y = f.subs({symbols[0]: x0[0], symbols[1]: x0[1]})
        # true value (global optimum) is 0
        squared_error = predicted_y**2
        mse.append(squared_error)
        absolute_err.append(abs(predicted_y))
        x0 = [x0[i] - diff[i] for i in range(len(x0))]
        steps.append(x0)

    if len(mse) != 0:
        final_mse = str(round(sum(mse) / len(mse), 2))
        final_abs_err = str(round(sum(absolute_err) / len(absolute_err), 2))
    else:
        final_mse = "N/A"
        final_abs_err = "N/A"
    x0 = [round(x, 2) for x in x0]
    return steps, x0, final_mse, final_abs_err


def plot_gradient_descent(steps, f, symbols, exp_params):
    plt.figure()
    x1_vals, x2_vals = zip(*steps)
    X1, X2 = np.meshgrid(
        np.linspace(min(x1_vals) - 1, max(x1_vals) + 1, 100),
        np.linspace(min(x2_vals) - 1, max(x2_vals) + 1, 100),
    )
    Z = np.zeros_like(X1)
    for i in range(len(X1)):
        for j in range(len(X1[0])):
            Z[i, j] = f.subs({symbols[0]: X1[i, j], symbols[1]: X2[i, j]})

    lr, x0, fnc_name = exp_params
    plt.contourf(X1, X2, Z, levels=20, cmap="viridis")
    plt.colorbar(label="f(x1, x2)")
    plt.scatter(x1_vals, x2_vals, color="red", label="kroki", s=7)
    plt.title(
        f"Gradient prosty dla funkcji {fnc_name.capitalize()}(lr={lr}, start={x0})"
    )
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    cwd = os.getcwd()
    plt.savefig(
        f"{cwd}/lab1/experiments/{fnc_name}_gradient_descent_{lr}_{x0[0]}_{x0[1]}.png"
    )
    # plt.show()


rastrigin_domain = [-5.12, 5.12]

symbols = sp.symbols("x1 x2")

# rastrigin_experiments(symbols)
# griewanka_experiments(symbols)
x_squared_experiments(symbols)
