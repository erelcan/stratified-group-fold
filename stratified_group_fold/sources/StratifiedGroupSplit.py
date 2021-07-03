import pandas as pd
from sklearn.model_selection._split import _BaseKFold
from partition_optimizers import weighted_split, mixed_weighted_split


# Designed to be used independent of sklearn base splitters~
# ratio represents the training-ratio..
class StratifiedGroupSplit():
    def __init__(self, ratio=0.8, mixed_groups=False, slack=0):
        self._ratio = ratio
        self._mixed_groups = mixed_groups
        self._slack = slack
    
    def split(self, labels, groups):
        # groups and labels must be series with same indexing!
        # Also, these indices should represent the (corresponding) sample indices (or must be carefully handled outside~)
        df = pd.DataFrame({"groups": groups, "labels": labels})
        if self._mixed_groups:
            return self._handle_mixed_groups(df)
        else:
            return self._handle_non_mixed_groups(df)
    
    def _handle_mixed_groups(self, df):
        # Optimize this in the future..
        weights = {}
        for c in df["labels"].unique():
            weights[c] = {}
        for i in range(len(df)):
            cur_label = df.iloc[i]["labels"].item()
            cur_group = df.iloc[i]["groups"].item()
            if cur_group not in weights[cur_label]:
                weights[cur_label][cur_group] = 0
            weights[cur_label][cur_group] += 1

        cur_splits = mixed_weighted_split(df["groups"].unique(), weights, self._ratio, self._slack)
        training_split = self._get_sample_ids_for_groups(df, cur_splits[0]["ids"])
        test_split = self._get_sample_ids_for_groups(df, cur_splits[1]["ids"])

        return training_split, test_split

    def _handle_non_mixed_groups(self, df):
        # Handles the problems where for a group all members having the same class!
        training_split = []
        test_split = []
        for c in df["labels"].unique():
            cur_df = df[df["labels"] == c]
            group_counts = cur_df["groups"].value_counts().to_dict()
            cur_splits = weighted_split(list(group_counts.keys()), list(group_counts.values()), self._ratio, self._slack)
            training_split.extend(self._get_sample_ids_for_groups(cur_df, cur_splits[0]["ids"]))
            test_split.extend(self._get_sample_ids_for_groups(cur_df, cur_splits[1]["ids"]))

        return training_split, test_split
    
    def _get_sample_ids_for_groups(self, df, group_ids):
        sample_ids = []
        for group in group_ids:
            sample_ids.extend(df[df["groups"] == group].index.tolist())
        return sample_ids