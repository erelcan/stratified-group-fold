import pandas as pd
from sklearn.model_selection._split import _BaseKFold
from partition_optimizers import equally_partition_into_bins, mixed_equally_partition_into_bins

# Using get_n_splits from the super class.
class StratifiedGroupKFold(_BaseKFold):
    def __init__(self, n_splits=5, mixed_groups=False, opt_type=0, reset_index=True):
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self._mixed_groups = mixed_groups
        self._opt_type = opt_type
        self._reset_index = reset_index
        
    def split(self, X, y=None, groups=None):
        folds = self._prepare_folds(y, groups)
        for i in range(len(folds)):
            yield self._train_test_from_folds(folds, [i])

    
    def _train_test_from_folds(self, folds, test_folds):
        test_folds_set = set(test_folds)
        train_ids = []
        test_ids = []
        for i in range(len(folds)):
            if i in test_folds_set:
                test_ids.extend(folds[i])
            else:
                train_ids.extend(folds[i])
        return train_ids, test_ids
    
    def _prepare_folds(self, labels, groups):
        # groups and labels must be series with same indexing!
        # Also, these indices should represent the (corresponding) sample indices (or must be carefully handled outside~)
        # Since sklearn cross validation converts dataframes/series to arrays, it expects 0-based index samples. In otherwords,
        # any sample id we return here is relative to location of the samples (arrays yielded by sklearn conversion).
        # Since sklearn looks up samples by array indices; the index values remaining from the actual dataframe becomes incompatible.
        # This raises "positional indexers are out-of-bounds" error.
        # To prevent such issues, we provide index_resetting option.
            
        df = pd.DataFrame({"groups": groups, "labels": labels})
        if self._reset_index:
            df["index"] = list(range(len(labels)))
            df = df.set_index("index", drop=True)

        if self._mixed_groups:
            return self._handle_mixed_groups(df)
        else:
            return self._handle_non_mixed_groups(df)
    
    def _handle_mixed_groups(self, df):
        result_folds = {}
        for i in range(self.n_splits):
            result_folds[i] = []

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
        
        cur_folds = mixed_equally_partition_into_bins(df["groups"].unique().tolist(), weights, self.n_splits)

        for i in range(len(cur_folds)):
            result_folds[i] = self._get_sample_ids_for_groups(df, cur_folds[i]["ids"])

        return result_folds

    def _handle_non_mixed_groups(self, df):
        # Handles the problems where for a group all members having the same class!
        result_folds = {}
        for i in range(self.n_splits):
            result_folds[i] = []
        for c in df["labels"].unique():
            cur_df = df[df["labels"] == c]
            group_counts = cur_df["groups"].value_counts().to_dict()
            cur_folds = equally_partition_into_bins(list(group_counts.keys()), list(group_counts.values()), self.n_splits, self._opt_type)
            for i in range(len(cur_folds)):
                result_folds[i].extend(self._get_sample_ids_for_groups(cur_df, cur_folds[i]["ids"]))
        return result_folds
    
    def _get_sample_ids_for_groups(self, df, group_ids):
        sample_ids = []
        for group in group_ids:
            sample_ids.extend(df[df["groups"] == group].index.tolist())
        return sample_ids