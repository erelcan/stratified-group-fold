import numpy as np
import pandas as pd
import random

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def create_class_distributions(num_of_distributions, num_of_classes):
    alpha = [random.randint(1, 10) for i in range(num_of_classes)]
    return np.random.dirichlet(alpha, size=num_of_distributions)


def create_data(num_of_groups, group_size_bounds, class_ids, feature_length, is_mixed=False):
    if is_mixed:
        distributions = create_class_distributions(num_of_groups, len(class_ids))
    
    data_dict = {}
    data_dict["labels"] = []
    data_dict["groups"] = []
    for i in range(num_of_groups):
        group_size = random.randint(*group_size_bounds)
        if is_mixed:
            data_dict["labels"].extend(random.choices(class_ids, weights=distributions[i], k=group_size))
        else:
            data_dict["labels"].extend([random.choice(class_ids)] * group_size)

        data_dict["groups"].extend([i] * group_size)
 
    for i in range(feature_length):
        data_dict["feature" + str(i)] = np.random.normal(0, 1, len(data_dict["labels"])).tolist()

    return pd.DataFrame(data_dict)


def check_splits(data, train_indices, test_indices):
    train_data = data.iloc[train_indices]
    test_data = data.iloc[test_indices]

    train_groups = set(train_data["groups"].unique())
    test_groups = set(test_data["groups"].unique())
    group_intersection = train_groups.intersection(test_groups)
    group_union = train_groups.union(test_groups)
    print("# of unique groups: " + str(len(data["groups"].unique())))
    print("# of unique train groups: " + str(len(train_groups)))
    print("# of unique test groups: " + str(len(test_groups)))
    print("group intersection size: " + str(len(group_intersection)))
    print("group union size: " + str(len(group_union)))
    
    train_class_ratios = train_data["labels"].value_counts(normalize=True)
    test_class_ratios = test_data["labels"].value_counts(normalize=True)

    print("class distribution in training:")
    print(train_class_ratios.to_dict())
    print("class distribution in test:")
    print(test_class_ratios.to_dict())

    tr_size = len(train_data)
    t_size = len(test_data)
    tr_ratio = tr_size / (tr_size + t_size)
    print("training sample size: " + str(tr_size))
    print("test sample size: " + str(t_size))
    print("training ratio: " + str(tr_ratio))