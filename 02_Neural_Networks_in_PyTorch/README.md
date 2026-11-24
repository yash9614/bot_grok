# 02 · Neural Networks in PyTorch: AtliQ Wildlife Reserve Threat Detector

**Notebook:** [`Neural_Networks_PyTorch_Solutions.ipynb`](Neural_Networks_PyTorch_Solutions.ipynb)
**Data:** [`habitat_images_codebasics_DL.csv`](habitat_images_codebasics_DL.csv) (100 rows: Image Brightness, Movement Intensity, Number of Shapes Detected, Noise Level, Threat Label 0/1)

## What's inside
**Warm-up revision**
1. *AtliQ Warehouse*: an `nn.Sequential` network with 128 inputs, two hidden layers of 64 (ReLU) and 1 output.
2. *Sales data*: NumPy array to `torch.tensor(..., dtype=torch.float32)`.
3. *Customer classification*: raw logits plus `nn.CrossEntropyLoss` for 3 classes (loss = 1.0845).

**Main task: Habitat Threat Detector** (binary classification). It reached **97% test accuracy** in the saved run, with a confusion-matrix heatmap.

## The full deep-learning recipe (you'll reuse it in every module)
```
  CSV ─► pandas ─► min-max normalise ─► torch tensors ─► train/test split (70/30)
                                                              │
                                          TensorDataset + DataLoader(batch_size=16, shuffle)
                                                              │
                                                              ▼
   ┌──────────────── training loop (20 epochs) ────────────────┐
   │ for X_batch, y_batch in train_loader:                     │
   │     y_pred = model(X_batch)          # 1. forward pass    │
   │     loss   = BCELoss(y_pred, y_batch)# 2. measure error   │
   │     optimizer.zero_grad()            # 3. clear old grads │
   │     loss.backward()                  # 4. backprop        │
   │     optimizer.step()                 # 5. update weights  │
   └───────────────────────────────────────────────────────────┘
                                                              ▼
              model.eval() + torch.no_grad() ─► threshold 0.5 ─► accuracy + confusion matrix
```

## Key concepts
* **`nn.Module`**: subclass it, build layers in `__init__`, and describe the data flow in `forward()`. `ThreatDetector` wraps `Linear(4,8) → ReLU → Linear(8,1) → Sigmoid`.
* **`nn.Sequential`**: a quick way to stack layers in order.
* **Activation functions**: ReLU (`max(0,x)`) adds non-linearity in hidden layers. Sigmoid squashes the output to 0‒1, so it reads as a probability.
* **Loss functions**:
  * `nn.BCELoss` is binary cross-entropy. It expects probabilities, which is why there's a Sigmoid at the end.
  * `nn.CrossEntropyLoss` is for multi-class. It expects *raw logits* plus integer class labels.
* **`TensorDataset` / `DataLoader`**: pair features with labels and serve them in shuffled mini-batches.
* **Optimizer**: `optim.Adam(model.parameters(), lr=0.01)`.
* **Evaluation mode**: `model.eval()` and `torch.no_grad()` switch off training-only behaviour and gradient tracking.
* **Confusion matrix**: shows *which* mistakes the model makes (false alarms vs missed threats), not just how many.

## Clean-up note
`pd.read_csv("/content/…")` (a Google Colab path) became `pd.read_csv("habitat_images_codebasics_DL.csv")`. The CSV sits next to the notebook, so it runs both locally and in Colab once you upload the CSV.
