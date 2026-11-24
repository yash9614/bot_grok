# 03 · Neural Network Training: AI Agriculture Company

**Notebook:** [`Neural_Network_Training_Solutions.ipynb`](Neural_Network_Training_Solutions.ipynb)
**Data:** [`agriculture_dataset_codebasics_DL.csv`](agriculture_dataset_codebasics_DL.csv) (10 rows: Temperature, Rainfall, Soil pH, Nitrogen, Irrigation Hours, Fertilizer Usage, then the target **Crop Yield (tons/ha)**)

## Three tasks
| Task | What you build | Concept |
|---|---|---|
| 1 · Equipment cost | 1-weight linear model, target `cost = material²` | **Backprop by hand** and checking it with autograd (`torch.allclose`) |
| 2 · Production efficiency | `Linear(2,5) → ReLU → Linear(5,1) → Sigmoid`, SGD, 200 epochs | Train/validation loss curves |
| 3 · Crop yield | `Linear(6,10) → ReLU → Linear(10,1)` regression with MSE | **Batch vs Mini-batch vs Stochastic** gradient descent |

## Concept 1: backpropagation is just the chain rule
With input `x = 5`, weight `w = 0.5`, bias `b = 0` and target `25`:
```
 x ──► ŷ = x·w + b ──► L = (ŷ − y)²
        ŷ = 2.5          L = (2.5 − 25)² = 506.25

 dL/dŷ = 2(ŷ − y)        = −45
 dL/dw = dL/dŷ · dŷ/dw   = −45 · x = −225
 dL/db = dL/dŷ · dŷ/db   = −45 · 1 = −45
```
The notebook computes these by hand, calls `loss.backward()`, and confirms that `weight.grad = −225` and `bias.grad = −45`, so the two match.

## Concept 2: gradient descent update
```
   w_new = w_old − learning_rate × dL/dw
```
The learning rate is the step size. Too small and training crawls. Too big and the loss bounces or blows up (try the notebook's bonus: 500 epochs, lr = 0.05).

## Concept 3: three flavours of gradient descent
```
 Batch GD        [██████████ all rows ██████████] ─► 1 update / epoch    (smooth, slow)
 Mini-batch GD   [███ 16 ███][███ 16 ███][███ …]  ─► many updates/epoch  (the usual choice)
 Stochastic GD   [█][█][█][█][█][█][█][█][█][█]   ─► 1 update per row   (noisy, fast steps)
```
`train_model(optimizer, X, y, batch_size=None | 16 | 1)` implements all three, and the loss curves are plotted side by side. In the saved run the final validation MSE is **0.0153**.

## Concept 4: training vs validation loss
Plot both. If both go down, the model is learning. If training goes down but validation goes up, it's **overfitting** (Module 05 fixes that).

## 🧑‍🏫 Teacher's notes (original course code left as-is)
* **Task 2:** `scaler.fit_transform(X)` runs *after* the train/val split and its result isn't reused, so `X_train`/`X_val` stay **unscaled**. To really normalise, do `X_train = scaler.fit_transform(X_train); X_val = scaler.transform(X_val)`.
* **Task 2:** the markdown formula divides by 5, but the code divides by **50**. Dividing by 50 is what keeps the score between 0 and 1, which matters because the output is a Sigmoid.
* The tip image at the end of the notebook is embedded in the notebook itself.

## Clean-up note
The Colab path `/content/agriculture_dataset_codebasics_DL.csv` became the relative path `agriculture_dataset_codebasics_DL.csv`.
