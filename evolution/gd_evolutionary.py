import sympy as sp
import numpy as np
import random
import configparser
import os

import matplotlib.pyplot as plt


def load_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config


def rastrigin_for_d2(symbols: list):
    x1, x2 = symbols
    return (
        20
        + x1**2
        - 10 * sp.cos(2 * np.pi * x1)
        + x2**2
        - 10 * sp.cos(2 * np.pi * x2)
    )


def griewanka_for_d2(symbols: list):
    x1, x2 = symbols
    return (
        1
        + x1**2 / 4000
        + x2**2 / 4000
        - (sp.cos(x1 / sp.sqrt(1)) * sp.cos(x2 / sp.sqrt(2)))
    )


def dropwave_for_d2(symbols: list):
    x1, x2 = symbols
    return -(1 + sp.cos(12 * sp.sqrt(x1**2 + x2**2))) / (
        0.5 * (x1**2 + x2**2) + 2
    )


def evaluate_population(population, function):
    fitness_scores = []
    for individual in population:
        fitness_scores.append(function(*individual))
    return fitness_scores


def generate_population(population_size, domain):
    population = []
    for _ in range(population_size):
        individual = [random.uniform(domain[0], domain[1]) for _ in range(2)]
        population.append(individual)
    return population


def tournament_selection_with_elitism(
    population, fitness_scores, tournament_size, elitism_size
):
    new_population = []
    for _ in range(elitism_size):
        best_individual = population[fitness_scores.index(min(fitness_scores))]
        new_population.append(best_individual)
        # population.remove(best_individual)
        # fitness_scores.remove(max(fitness_scores))
    for _ in range(len(population)):
        tournament = random.sample(
            list(zip(population, fitness_scores)), tournament_size
        )
        best_individual = min(tournament, key=lambda x: x[1])[0]
        new_population.append(best_individual)
        # population.remove(best_individual)
        # fitness_scores.remove(max(fitness_scores))
    return new_population


def crossover(population, crossover_rate):
    new_population = []

    for i in range(0, len(population) - 1, 2):
        if random.random() < crossover_rate:
            new_population.append([population[i][0], population[i + 1][1]])
            new_population.append([population[i + 1][0], population[i][1]])
        else:
            new_population.append(population[i])
            new_population.append(population[i + 1])
    return new_population


def mutation(population, mutation_rate, domain):
    for individual in population:
        if random.random() < mutation_rate:
            to_add_x1 = random.uniform(-0.5, 0.5)
            to_add_x2 = random.uniform(-0.5, 0.5)
            while abs(individual[0] + to_add_x1) > abs(domain[0]) or abs(
                individual[1] + to_add_x2
            ) > abs(domain[1]):
                to_add_x1 = random.uniform(-0.5, 0.5)
                to_add_x2 = random.uniform(-0.5, 0.5)
            individual[0] += to_add_x1

            individual[1] += to_add_x2
    return population


def find_best_individual(population, fitness_scores):
    return population[fitness_scores.index(min(fitness_scores))]


def run_algorithm(population, domain, general_cfg, function):
    scores = evaluate_population(population, function)
    best = find_best_individual(population, scores)
    for _ in range(general_cfg["num_generations"]):
        population = tournament_selection_with_elitism(
            population,
            scores,
            general_cfg["tournament_size"],
            general_cfg["elitism_size"],
        )
        population = crossover(population, general_cfg["crossover_rate"])
        population = mutation(population, general_cfg["mutation_rate"], domain)
        scores = evaluate_population(population, function)
        new_best = find_best_individual(population, scores)
        if function(*new_best) < function(*best):
            best = new_best
    return population, best


def get_experiment_settings(configurations, section):
    lower_bound = configurations.getfloat(section, "lower_bound")
    upper_bound = configurations.getfloat(section, "upper_bound")
    step = configurations.getfloat(section, "step")
    return lower_bound, upper_bound, step


def run_experiment(exp_function, general_cfg, configurations, domain):
    """run all experiments for the experiment function"""

    # move to run_algorithm
    population_size = general_cfg["population_size"]
    population = generate_population(population_size, rastrigin_dropwave_domain)
    fnc_name = exp_function[0]
    exp_function = exp_function[1]
    # scores = evaluate_population(population, rastrigin)
    # best = find_best_individual(population, scores)
    # start with the same population for all experiments

    for section in configurations.sections():
        if section != "general":
            lower_bound, upper_bound, step = get_experiment_settings(
                configurations, section
            )

            if section == "mutation_rate_experiment":
                mr = lower_bound
                exp_results = {}
                while mr <= upper_bound:
                    general_cfg["mutation_rate"] = mr
                    population, best = run_algorithm(
                        population, domain, general_cfg, exp_function
                    )
                    # print(exp_function(*best))
                    exp_results[mr] = [exp_function(*best), population]
                    mr += step
                    mr = round(mr, 2)
                # plot_parameter_vs_score(exp_results, "mutation_rate", fnc_name)
                # for key in exp_results.keys():
                # 	print(f"{key}: {exp_results[key][0]}")
            elif section == "crossover_rate_experiment":
                cr = lower_bound
                exp_results = {}
                while cr <= upper_bound:
                    general_cfg["crossover_rate"] = cr
                    population, best = run_algorithm(
                        population, domain, general_cfg, exp_function
                    )
                    exp_results[cr] = [exp_function(*best), population]
                    plot_population_for_cr(population, cr, fnc_name)
                    cr += step
                    cr = round(cr, 2)
                # plot_parameter_vs_score(exp_results, "crossover_rate", fnc_name)
                # for key in exp_results.keys():
                # 	print(f"{key}: {exp_results[key][0]}")
            elif section == "population_size_experiment":
                ps = int(lower_bound)
                exp_results = {}
                while ps <= upper_bound:
                    general_cfg["population_size"] = ps
                    population, best = run_algorithm(
                        population, domain, general_cfg, exp_function
                    )
                    exp_results[ps] = [exp_function(*best), population]
                    ps += step
                    ps = int(ps)
                # plot_parameter_vs_score(exp_results, "population_size", fnc_name)
            elif section == "num_generations_experiment":
                ng = int(lower_bound)
                exp_results = {}
                while ng <= upper_bound:
                    general_cfg["num_generations"] = ng
                    population, best = run_algorithm(
                        population, domain, general_cfg, exp_function
                    )
                    exp_results[ng] = [exp_function(*best), population]
                    ng += step
                    ng = int(ng)
                # plot_parameter_vs_score(exp_results, "num_generations", fnc_name)


def plot_parameter_vs_score(exp_results, parameter_name, fnc_name):
    parameter_values = list(exp_results.keys())
    best_scores = [result[0] for result in exp_results.values()]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(parameter_values, best_scores, marker="o", linestyle="-")
    plt.xlabel(parameter_name)
    plt.ylabel("Best Individual Score")
    plt.title(f"{parameter_name} vs. Best Individual Score")
    plt.grid(True)
    cwd = str(os.getcwd())
    filepath = f"{cwd}/lab2/plots/{fnc_name}_{parameter_name}.png"
    plt.savefig(filepath)

    # Save results to text file
    text_filepath = f"{cwd}/lab2/text/{fnc_name}_{parameter_name}.txt"
    with open(text_filepath, "w") as file:
        file.write(f"{parameter_name} vs. Best Individual Score\n")
        for param, score in zip(parameter_values, best_scores):
            file.write(f"{param}: {score}\n")


def plot_population_for_cr(population, cr, fnc_name):
    """Plot the final population for a given crossover rate."""
    plt.figure(figsize=(10, 6))
    x_vals = [ind[0] for ind in population]  # Extract x1 values
    y_vals = [ind[1] for ind in population]  # Extract x2 values
    plt.scatter(x_vals, y_vals, label=f"CR={cr}")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"Final Population Distribution for CR={cr} (Function: {fnc_name})")
    plt.legend()
    plt.grid(True)
    cwd = os.getcwd()
    filepath = f"{cwd}/lab2/plots/{fnc_name}_population_distribution_CR_{cr}.png"
    plt.savefig(filepath)
    plt.close()  # Close the plot to avoid displaying it inline if running in a notebook


cwd = os.getcwd()
configurations = load_config(f"{cwd}/lab2/experiments_config.ini")

# all move to main

general_settings = {
    "selection": configurations.get("general", "selection"),
    "tournament_size": configurations.getint("general", "tournament_size"),
    "elitism": configurations.getboolean("general", "elitism"),
    "elitism_size": configurations.getint("general", "elitism_size"),
    "population_size": configurations.getint("general", "population_size"),
    "num_generations": configurations.getint("general", "num_generations"),
    "mutation_rate": configurations.getfloat("general", "mutation_rate"),
    "crossover_rate": configurations.getfloat("general", "crossover_rate"),
}
# print(general_settings)


# population_size = 200
# num_generations = 200

rastrigin_dropwave_domain = [-5.12, 5.12]
griewanka_domain = [-50, 50]
symbols = sp.symbols("x1 x2")

rastrigin = sp.lambdify(symbols, rastrigin_for_d2(symbols))
griewanka = sp.lambdify(symbols, griewanka_for_d2(symbols))
dropwave = sp.lambdify(symbols, dropwave_for_d2(symbols))

rastrigin = ["rastrigin", rastrigin]
griewanka = ["griewanka", griewanka]
dropwave = ["dropwave", dropwave]
run_experiment(rastrigin, general_settings, configurations, rastrigin_dropwave_domain)
# run_experiment(griewanka, general_settings, configurations, griewanka_domain)
# run_experiment(dropwave, general_settings, configurations, rastrigin_dropwave_domain)


# print(rastrigin(*best))
