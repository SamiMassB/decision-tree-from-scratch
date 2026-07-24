# Decision Tree Spam Classifier

Decision Tree (entropy / information gain) built from scratch

## Setup
```bash
pip install -r requirements.txt
```

## Running
```bash
python -m src.main --data data/daneshchat_logs.csv
```

## Sample Results On Sample Data

```
DecisionTreeClassifierScratch
  Accuracy:  0.6875
  Precision: 0.729373445644206
  Recall:    0.6875
  F1-score:  0.6542730111732789
```

## What I Built

I implemented a binary decision tree classifier completely from scratch in NumPy/Pandas, with no use of `sklearn`'s model classes. It includes:

- **Entropy-based splitting**: uses `H(y) = -Σ p_i log2(p_i)` as the impurity measure at each node.
- **Information gain criterion**: for every candidate split, computes `IG = H(parent) - Σ (|child| / |parent|) * H(child)` and keeps the split that maximizes it.
- **Exhaustive threshold search**: for each feature, tries every midpoint between consecutive sorted values as a candidate threshold, so splits aren't limited to a fixed grid.
- **Recursive tree building**: recursively partitions the data into left/right subtrees until a stopping condition is met.
- **Configurable stopping rules**: `max_depth`, `min_samples_split`, and `min_samples_leaf` all control how deep the tree grows and prevent overfitting on tiny leaf nodes.
- **Majority-vote leaves**: each leaf predicts the majority class of the samples that land there.
- **From-scratch prediction**: inference walks the tree node by node, comparing each row's feature value against the learned threshold.
- **Custom evaluation metrics**: accuracy, precision, recall, and F1-score are also computed manually rather than pulled from a library.

=======
# decision-tree-from-scratch
A binary decision tree classifier built from scratch in NumPy/Pandas (entropy, information gain, recursive splitting) applied to spam message detection — no sklearn model classes used.
