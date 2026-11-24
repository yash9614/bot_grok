# 05 · Model Optimization: Regularization (AtliQ Customer Churn)

**Notebook:** [`Regularization_Techniques_Solutions.ipynb`](Regularization_Techniques_Solutions.ipynb)
**Data:** [`AtliQ_Churn_Prediction_Codebasics_DL.csv`](AtliQ_Churn_Prediction_Codebasics_DL.csv) (5,000 customers, 6 behaviour features, target `Churned` 0/1)

## What's inside
**Revision problems**
1. **L1 regularization**: for `w = [0.5, −0.3, 0.8, −1.5]`, `Σ|w| = 3.1` and its gradient is `sign(w) = [1, −1, 1, −1]`.
2. **Batch normalization by hand**: `(X − mean) / std`, worked out column by column without `nn.BatchNorm`.

**Main task: churn prediction** (70/30 split, `StandardScaler`, BCELoss, SGD lr 0.01, 30 epochs, batch 32)

| Model | Change | Final val loss (saved run) |
|---|---|---|
| Base | `Linear 6→32→16→1` + ReLU + Sigmoid | 0.6129 |
| + Dropout | `nn.Dropout(0.4)` after each hidden layer | 0.6122 |
| + L2 | `optim.SGD(..., weight_decay=0.1)` | 0.6196 |
| + BatchNorm | `nn.BatchNorm1d(32)` / `(16)` | 0.6188 |

## Why regularize?
```
 Underfit            Just right            Overfit
 ───────            ───────────            ────────
 train loss high    train ≈ val, both low  train ↓↓  but  val ↑
 model too simple   generalises            memorised the training set
```
Regularization keeps a model on the "just right" side.

## The techniques
* **L1 (Lasso)**: adds `λ·Σ|w|` to the loss. Its gradient is a constant push (`±λ`) toward 0, so many weights become **exactly zero** (sparse, a kind of automatic feature selection).
* **L2 (Ridge / weight decay)**: adds `λ·Σw²`. The gradient `2λw` shrinks big weights the most and keeps them small but non-zero. In PyTorch it's just `weight_decay=` on the optimizer. With `0.1` the penalty was too strong here and both losses crept up (see the notebook's key insight). Smaller values like `1e-4`–`1e-3` are typical.
* **Dropout**: during training it randomly switches off p = 40% of neurons on each pass, so the network can't rely on any single neuron. `model.eval()` turns it off at test time. Expect a noisier training curve.
  ```
  train:  o  x  o  o  x  o     (x = dropped this batch)
  eval :  o  o  o  o  o  o     (all neurons, outputs scaled)
  ```
* **Batch Normalization**: re-centres and re-scales each layer's inputs per mini-batch (`(x−μ)/σ`, then a learnable scale and shift). That gives faster, more stable training. Here the training loss kept dropping while validation plateaued, which is a mild overfitting signal.

## Clean-up note
The Colab path `/content/AtliQ_Churn_Prediction_Codebasics_DL.csv` became the relative path. The file name typo `Model_Optimizations_…` was simplified.
