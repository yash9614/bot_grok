# 06 · Model Optimization: Hyperparameter Tuning (AtliQ Fashion)

**Notebook:** [`Hyperparameter_Tuning_Solutions.ipynb`](Hyperparameter_Tuning_Solutions.ipynb)
**Data:** FashionMNIST (auto-downloaded into `./data`, git-ignored). It uses a 10,000 train / 2,000 test subset via `random_split`.
**Extra install:** `pip install optuna`

## Parameters vs hyperparameters
* **Parameters** are what the model *learns* (weights, biases).
* **Hyperparameters** are what *you* choose before training: learning rate, batch size, number of neurons, epochs, and so on.

## Three search strategies (all implemented)
```
 Grid search                 Random search               Bayesian (Optuna)
 lr ↑  ● ● ●                 lr ↑   ●     ●              lr ↑      ○ → ● 
       ● ● ●                         ●  ●                         ○ ↗ ●●
       ● ● ●                       ●                             ○
       ───────► batch             ───────► batch                ───────► neurons
 try EVERY combo             try N random combos         learn from past trials,
 (exhaustive, costly)        (cheap, may miss best)      sample promising areas
```
| Method | Search space in notebook | Saved best |
|---|---|---|
| Grid | lr ∈ {0.001, 0.01, 0.1} × batch ∈ {32, 64} | lr 0.1, batch 64 |
| Random (5 trials) | lr ∈ {0.0001…0.1}, batch ∈ {16, 32, 64, 128} | lr 0.1, batch 128 |
| Optuna (10 trials) | `trial.suggest_float('lr', 1e-4, 0.1)`, `trial.suggest_int('neurons', 32, 256)` | lr ≈ 0.093, neurons 118 |

The final model is re-trained for 50 epochs with the best `lr` and `batch_size` and reaches **77.10% test accuracy**. A 3×3 grid of predictions is plotted at the end.

## Optuna in 4 lines
```python
def objective(trial):
    lr = trial.suggest_float('lr', 0.0001, 0.1)
    neurons = trial.suggest_int('neurons', 32, 256)
    ...train a few epochs...; return test_loss      # value to minimise
study = optuna.create_study(direction='minimize'); study.optimize(objective, n_trials=10)
```

## 🧑‍🏫 Teacher's notes
* Grid and random search here keep training the *same* model object, so later combinations get a head start. Re-initialising the model for each trial gives a fairer comparison (the Optuna objective already does this).
* Like Module 04, the network ends in `Softmax` before `CrossEntropyLoss`. Dropping the Softmax layer usually lowers the loss.

## Clean-up note
Renamed from `Model_Optimization_Hyperparameter_Tunning_Excerise_Solution.ipynb`, fixing the "Tunning"/"Excerise" typos.
