# STRATIFIED GROUP FOLDING/SPLITTING

When we would like to keep class distribution same among splits, we sample in a stratified manner. Sklearn has several folding/splitting mechanism to handle stratified sampling. It also provides handling groups. Sometimes, we need to ensure that all samples coming from the same group to occur in the same split to prevent information leakage. However, sklearn does not provide handling stratified sampling along with group-handling. There is some work on this issue in the experimental version, but not official. Also, both this solution and the group-only sampling seems like they are greedy solutons which may probably handle many cases; but can we do better?

In this project, we provide folding and splitting solutions for **stratified group sampling**. We present them under two cases; namely mixed groups and non-mixed groups such that a group has samples with the same class label for the non-mixed case while class labels can vary for the mixed case. Former is a special case of the latter. However, we provide them separately since the former can be solved with less constraints; and that may be solved more efficiently.

Ok, but what is a **group**? Let's say we would like to classify a network activity/package as a fraud or not. Dataset may consist of multiple samples coming from same users. Having these samples both in training and test set may cause **information leakage**. Personalized factors may have identifying effects and if we have such samples in the training set; the model may "memorize". Therefore, we should consider such samples coming from the same user as a **group** and ensure that all these samples are in the same split/fold.

## Non-mixed Stratified Sampling

In this case, we know that samples from a group has the same class label.

**How to prepare K-folds having the group constraint and also having same class distribution among folds?**

*Observation*: Given K-Folds, we should have same number of samples for a given class in each fold. That makes approximately (ceil of) **# of Ci (class i) / k** samples on each fold having class label Ci.

Let's say samples from groups G1, G4, G6, G7, ... have class label Ci.

Assume that the # of samples for each group is like: {G1: 3, G4: 2, G6: 1, G7: 5, ...}

Then, let's make lists such as id_list=[G1, G4, G6, G7, ...] and weights=[3, 2, 1, 5, ...]

Then, we can formulize the problem such that assing groups (of the id_list) to bins and satisfy weight total in each fold is smaller or equal to **# of Ci (class i) / k**.

For instance, if you place 3 into fold1; you know for sure that all the samples coming from G1 is in the fold1. Also, capacity constraint forces bins to have equal number of samples so that we can keep the distribution of class_i same for all folds. When we repeat such assignment process for each class; we have all folds having the same class distribution and group members are restricted into the same folds.

For train-test splitting; we have two bins/folds. The constraint we have is that the total weight in the training fold must be between an interval computed by a training-test ratio and some slack. For example, let's say training-test ratio is 80-20 and slack is 0.01. Then, the constraint forces the solution into the interval of [79.99, 80.01]. Slack is for relaxing the constraint to have feasible solutions when we can't divide in exact manner~.

## Mixed Stratified Sampling

In this case, we know that samples from a group can have different class labels.

**How to prepare K-folds having the group constraint and also having same class distribution among folds?**

This is a harder problem. There are more constraints to be satisfied. Now, rather than single capacity, we should check for a capacity for each class. Each bin/fold should satisfy the capacity constraint (**# of Ci (class i) / k**) for each class. We extend the splitting solution in the same way as well.

Notice that we can't handle each class separately as in non-mixed case since a group may have samples from different classes. 

## Approach

We approach to the problem as an optimization problem. We target a dummy objective and satisfy the constraints. We have capacity constraints as presented in the previous sections. We also add constraints to push each group to be exactly in single fold/bin.

We utilized or-tools for solving the problems. We present 2 classes; one for splitting and one for folding. The folding class is extended from sklearn's _BaseKFold. We may directly use it in GridSearchCV and in other supporting classes~.

We also present the functionality by examples and verify the solution.

## Bonus

We provide the VSCode Docker Environment which you can directly setup the project and try the examples. Also, such configuration will be handy for your other projects/services if you would like to develop on containers with VSCode.