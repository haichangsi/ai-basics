from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn import svm, tree
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt

iris = fetch_ucirepo(id=53)

X = iris.data.features
y = iris.data.targets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True, random_state=42, stratify=y
)

random_states = [0, 23, 67]

scoring = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, average="weighted", zero_division=0),
    "recall": make_scorer(recall_score, average="weighted"),
    "f1": make_scorer(f1_score, average="weighted"),
}

all_svm_results = pd.DataFrame()
all_tree_results = pd.DataFrame()


def process_scores_svm(
    scores,
    random_state,
    kernel="linear",
    C=1,
    max_iter=-1,
    all_svm_results=all_svm_results,
    test_name="kernels",
):
    res = pd.DataFrame()

    for metric, scores in scores.items():
        if metric == "fit_time" or metric == "score_time":
            continue
        for i, score in enumerate(scores):
            results = pd.DataFrame(
                {
                    "Test Name": test_name,
                    "Random State": random_state,
                    "Kernel": kernel,
                    "Fold": i + 1,
                    "C": C,
                    "Max Iter": max_iter if max_iter != -1 else "Default",
                    "Metric": metric.split("_")[1],
                    "Score": score,
                },
                index=[0],
            )
            res = pd.concat([results, res], ignore_index=True)

    return pd.concat([res, all_svm_results], ignore_index=True)


def test_svm_kernels(all_svm_results=all_svm_results):
    for random_state in random_states:
        # print(f"Random state: {random_state}")
        for kernel in ["linear", "poly", "rbf"]:
            svc = svm.SVC(kernel=kernel, random_state=random_state)
            svc_scores = cross_validate(
                svc,
                X,
                y.values.ravel(),
                scoring=scoring,
                cv=5,
                return_train_score=False,
            )
            all_svm_results = process_scores_svm(
                svc_scores,
                random_state,
                kernel=kernel,
                all_svm_results=all_svm_results,
                test_name="kernels",
            )
    return all_svm_results


def test_svm_C(all_svm_results=all_svm_results):
    for random_state in random_states:
        for C in [0.1, 1, 10]:
            svc = svm.SVC(kernel="linear", C=C, random_state=random_state)
            svc_scores = cross_validate(
                svc,
                X,
                y.values.ravel(),
                scoring=scoring,
                cv=5,
                return_train_score=False,
            )
            all_svm_results = process_scores_svm(
                svc_scores,
                random_state,
                C=C,
                all_svm_results=all_svm_results,
                test_name="C",
            )

    return all_svm_results


def test_svm_max_iter(all_svm_results=all_svm_results):
    for random_state in random_states:
        for max_iter in [1000, 10000, 1000000]:
            svc = svm.SVC(
                kernel="linear",
                C=1,
                tol=1e-16,
                max_iter=max_iter,
                random_state=random_state,
            )
            svc_scores = cross_validate(
                svc,
                X,
                y.values.ravel(),
                scoring=scoring,
                cv=4,
                return_train_score=False,
            )
            all_svm_results = process_scores_svm(
                svc_scores,
                random_state,
                max_iter=max_iter,
                all_svm_results=all_svm_results,
                test_name="Max Iter",
            )

    return all_svm_results


def process_scores_tree(
    scores,
    random_state,
    criterion="gini",
    splitter="best",
    max_depth=None,
    test_name="Trees",
    all_tree_results=all_tree_results,
):
    res = pd.DataFrame()
    for metric, scores in scores.items():
        if metric == "fit_time" or metric == "score_time":
            continue
        for i, score in enumerate(scores):
            results = pd.DataFrame(
                {
                    "Test Name": test_name,
                    "Random State": random_state,
                    "Criterion": criterion,
                    "Splitter": splitter,
                    "Fold": i + 1,
                    "Max Depth": max_depth if max_depth is not None else "Default",
                    "Metric": metric.split("_")[1],
                    "Score": score,
                },
                index=[0],
            )
            res = pd.concat([results, res], ignore_index=True)
    return pd.concat([res, all_tree_results], ignore_index=True)


def test_tree_max_depth(all_tree_results=all_tree_results):
    for random_state in random_states:
        for max_depth in [1, 2, 5, 10]:
            dtc = tree.DecisionTreeClassifier(
                max_depth=max_depth, random_state=random_state
            )
            dtc_scores = cross_validate(
                dtc,
                X,
                y.values.ravel(),
                scoring=scoring,
                cv=5,
                return_train_score=False,
            )
            all_tree_results = process_scores_tree(
                dtc_scores,
                random_state,
                max_depth=max_depth,
                test_name="Max Depth",
                all_tree_results=all_tree_results,
            )
    return all_tree_results


def test_tree_criterion(all_tree_results=all_tree_results):
    for random_state in random_states:
        for criterion in ["gini", "entropy"]:
            dtc = tree.DecisionTreeClassifier(
                criterion=criterion, random_state=random_state
            )
            dtc_scores = cross_validate(
                dtc,
                X,
                y.values.ravel(),
                scoring=scoring,
                cv=5,
                return_train_score=False,
            )
            all_tree_results = process_scores_tree(
                dtc_scores,
                random_state,
                criterion=criterion,
                test_name="Criterion",
                all_tree_results=all_tree_results,
            )
    return all_tree_results


def test_tree_splitter(all_tree_results=all_tree_results):
    for random_state in random_states:
        for splitter in ["best", "random"]:
            dtc = tree.DecisionTreeClassifier(
                splitter=splitter, random_state=random_state
            )
            dtc_scores = cross_validate(
                dtc,
                X,
                y.values.ravel(),
                scoring=scoring,
                cv=5,
                return_train_score=False,
            )
            all_tree_results = process_scores_tree(
                dtc_scores,
                random_state,
                splitter=splitter,
                test_name="Splitter",
                all_tree_results=all_tree_results,
            )
    return all_tree_results


def calc_mean_and_std(results):
    mean = np.mean(results)
    std = np.std(results)
    a = 1


def compare_svcs(all_svm_results=all_svm_results):
    test_max_iter = all_svm_results[(all_svm_results["Test Name"] == "Max Iter")]
    grouped_max_iter = test_max_iter.groupby(["Metric", "Random State", "Max Iter"])
    # Calculate mean and standard deviation for each group
    results_max_iter = grouped_max_iter["Score"].agg(["mean", "std"]).reset_index()
    # print(results_max_iter)
    results_max_iter_latex = results_max_iter.to_latex(index=False)

    test_c = all_svm_results[(all_svm_results["Test Name"] == "C")]
    grouped_c = test_c.groupby(["Metric", "Random State", "C"])
    results_c = grouped_c["Score"].agg(["mean", "std"]).reset_index()
    results_c_latex = results_c.to_latex(index=False)

    test_kernels = all_svm_results[(all_svm_results["Test Name"] == "kernels")]
    grouped_kernels = test_kernels.groupby(["Metric", "Random State", "Kernel"])
    results_kernels = grouped_kernels["Score"].agg(["mean", "std"]).reset_index()
    results_kernels_latex = results_kernels.to_latex(index=False)

    with open("lab4/tables_latex/svm_results.tex", "w") as f:
        f.write(results_max_iter_latex + "\n\n")
        f.write(results_c_latex + "\n\n")
        f.write(results_kernels_latex + "\n\n")


def compare_trees(all_tree_results=all_tree_results):
    test_max_depth = all_tree_results[(all_tree_results["Test Name"] == "Max Depth")]
    grouped_max_depth = test_max_depth.groupby(["Metric", "Random State", "Max Depth"])
    results_max_depth = grouped_max_depth["Score"].agg(["mean", "std"]).reset_index()
    # print(results_max_depth)
    results_max_depth_latex = results_max_depth.to_latex(index=False)

    test_criterion = all_tree_results[(all_tree_results["Test Name"] == "Criterion")]
    grouped_criterion = test_criterion.groupby(["Metric", "Random State", "Criterion"])
    results_criterion = grouped_criterion["Score"].agg(["mean", "std"]).reset_index()
    results_criterion_latex = results_criterion.to_latex(index=False)

    test_splitter = all_tree_results[(all_tree_results["Test Name"] == "Splitter")]
    grouped_splitter = test_splitter.groupby(["Metric", "Random State", "Splitter"])
    results_splitter = grouped_splitter["Score"].agg(["mean", "std"]).reset_index()
    results_splitter_latex = results_splitter.to_latex(index=False)

    with open("lab4/tables_latex/tree_results.tex", "w") as f:
        f.write(results_max_depth_latex + "\n\n")
        f.write(results_criterion_latex + "\n\n")
        f.write(results_splitter_latex + "\n\n")


def generate_and_plot_svm_C_vs_accuracy(X, y, scoring):
    C_values = np.logspace(-1, 2, num=50)  # More dense range with 100 points
    accuracies = []

    for C in C_values:
        svc = svm.SVC(kernel="linear", C=C, random_state=42)
        scores = cross_validate(
            svc, X, y, scoring=scoring, cv=5, return_train_score=False
        )
        accuracies.append(np.mean(scores['test_accuracy']))
    
    # Fit a polynomial trend line
    z = np.polyfit(np.log10(C_values), accuracies, 3)  # Degree 3 polynomial fit
    p = np.poly1d(z)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(C_values, accuracies, marker='o', linestyle='-', color='b', label='Accuracy')
    plt.plot(C_values, p(np.log10(C_values)), linestyle='--', color='r', label='Trend Line')
    plt.xscale('log')
    plt.xlabel('C Value (log scale)')
    plt.ylabel('Mean Accuracy')
    plt.title('SVM Accuracy vs. C Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
# Assuming X and y are defined, and you have a scoring dictionary defined as well.
scoring = {"accuracy": "accuracy"}
generate_and_plot_svm_C_vs_accuracy(X, y, scoring=scoring)

# all_svm_results = test_svm_kernels(all_svm_results)
# all_svm_results = test_svm_C(all_svm_results)
# all_svm_results = test_svm_max_iter(all_svm_results)
# all_svm_results.to_csv("lab4/test_svm_results.csv", index=False, mode="w")
# a = 1
# all_tree_results = test_tree_criterion(all_tree_results)
# all_tree_results = test_tree_max_depth(all_tree_results)
# all_tree_results = test_tree_splitter(all_tree_results)
# a = 1
# all_tree_results.to_csv("lab4/test_tree_results.csv", index=False, mode="w")


# compare_svcs(all_svm_results)
# compare_trees(all_tree_results)
