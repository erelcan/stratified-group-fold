from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

from StratifiedGroupKFold import StratifiedGroupKFold
from StratifiedGroupSplit import StratifiedGroupSplit
from common_utils import create_data, check_splits


def test_mixed_kfold():
    data = create_data(1000, (1, 8), list(range(3)), 3, True)
    y = data["labels"]
    groups = data["groups"]
    X = data.drop(columns=["labels", "groups"])

    stratified_group_fold = StratifiedGroupKFold(n_splits=5, mixed_groups=True)

    for train_indices, test_indices in stratified_group_fold.split(X, y, groups):
        # print("TRAIN:", train_indices)
        # print("TEST:", test_indices)
        check_splits(data, train_indices, test_indices)
        print("--------------------")


def test_mixed_split():
    data = create_data(1000, (1, 8), list(range(3)), 3, True)
    y = data["labels"]
    groups = data["groups"]

    splitter = StratifiedGroupSplit(mixed_groups=True, ratio=0.8, slack=0)
    train_indices, test_indices = splitter.split(y, groups)
    check_splits(data, train_indices, test_indices)
    print("--------------------")


def test_flow():
    data = create_data(1000, (1, 8), list(range(3)), 3, True)
    y = data["labels"]
    groups = data["groups"]
    X = data.drop(columns=["labels", "groups"])

    splitter = StratifiedGroupSplit(mixed_groups=True, ratio=0.8, slack=0.001)
    train_indices, test_indices = splitter.split(y, groups)
    
    train_X = X.iloc[train_indices]
    train_y = y.iloc[train_indices]
    train_groups = groups.iloc[train_indices]
    test_X = X.iloc[test_indices]
    test_y = y.iloc[test_indices]
    
    stratified_group_fold = StratifiedGroupKFold(n_splits=5, mixed_groups=True)

    tuned_parameters = [{'kernel': ["linear", "rbf"]}]
    grid_setup = GridSearchCV(SVC(), tuned_parameters, cv=stratified_group_fold)
    grid_setup.fit(train_X, train_y, groups=train_groups)
    print(grid_setup.cv_results_)

    preds = grid_setup.predict(test_X)
    print(classification_report(test_y, preds))


# test_mixed_split()
# test_mixed_kfold()
# test_flow()
