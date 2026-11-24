# CONCEPTS: Deep Learning with PyTorch, One-Page Revision

A compact recap of every idea used in the solved notebooks. Each section links to the module that goes deeper.

---

## 1. Tensors & autograd → [Module 01](01_PyTorch_Basics/README.md)
| Idea | Code | Remember |
|---|---|---|
| Create | `torch.tensor(data, dtype=torch.float32)` | like a NumPy array, but can run on a GPU |
| Device | `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` | move with `.to(device)` |
| Reduce | `t.sum(dim=0)`, `t.mean(dim=1)`, `t.max(dim=1)` | `dim=0` → per column, `dim=1` → per row |
| Multiply | `a * b` vs `torch.matmul(a, b)` | element-wise vs matrix product |
| Gradients | `x = torch.tensor(4., requires_grad=True); y.backward(); x.grad` | autograd applies the chain rule for you |

## 2. Building a network → [Module 02](02_Neural_Networks_in_PyTorch/README.md)
```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())
    def forward(self, x):
        return self.network(x)
```
`TensorDataset(X, y)` + `DataLoader(ds, batch_size=16, shuffle=True)` serve mini-batches.

## 3. The training loop (memorise this!)
```
 for epoch in range(E):
     for X, y in loader:
         pred = model(X)              ① forward
         loss = loss_fn(pred, y)      ② loss
         optimizer.zero_grad()        ③ reset grads
         loss.backward()              ④ backprop (chain rule)
         optimizer.step()             ⑤ update: w ← w − η·∂L/∂w
 model.eval(); with torch.no_grad(): …evaluate…
```

## 4. Picking the output layer and the loss
| Problem | Last layer | Loss |
|---|---|---|
| Regression (crop yield) | `Linear(…,1)` (no activation) | `nn.MSELoss` |
| Binary (threat, churn, spam) | `Linear(…,1)` + `Sigmoid` | `nn.BCELoss` (or no Sigmoid + `BCEWithLogitsLoss`) |
| Multi-class (FashionMNIST, flowers, tweets) | `Linear(…,C)` raw logits | `nn.CrossEntropyLoss` (softmax is built in) |

## 5. Gradient-descent flavours → [Module 03](03_Neural_Network_Training/README.md)
**Batch** (all rows per update) · **Mini-batch** (e.g. 16 rows) · **Stochastic** (1 row). Mini-batch is the everyday default.

## 6. Optimizers → [Module 04](04_Model_Optimization_Training_Algorithms/README.md)
```
 SGD       w -= η·g
 Momentum  v = βv + (1−β)g            ; w -= η·v
 RMSProp   s = βs + (1−β)g²           ; w -= η·g/√(s+ε)
 Adam      m = β1m+(1−β1)g, v = β2v+(1−β2)g², bias-correct, w -= η·m̂/(√v̂+ε)
```
`optim.SGD(params, lr, momentum)`, `optim.RMSprop(params, lr, alpha)`, `optim.Adam(params, lr, betas)`, `optim.AdamW` (for BERT).

## 7. Regularization → [Module 05](05_Model_Optimization_Regularization/README.md)
| Technique | How | Effect |
|---|---|---|
| L1 | `loss + λ·Σ abs(w)` | pushes weights to exactly 0 (sparse) |
| L2 / weight decay | `optim.SGD(..., weight_decay=λ)` | keeps weights small |
| Dropout | `nn.Dropout(p)` | randomly drops neurons while training |
| BatchNorm | `nn.BatchNorm1d(n)` / `BatchNorm2d` | normalises activations per batch, so training is faster and more stable |
| Data augmentation | `RandomHorizontalFlip`, `RandomCrop` | more varied training images (Module 07) |

Overfitting signal: training loss ↓ while validation loss ↑.

## 8. Hyperparameter tuning → [Module 06](06_Model_Optimization_Hyperparameter_Tuning/README.md)
**Grid** (every combination) · **Random** (N random picks) · **Bayesian / Optuna** (`trial.suggest_float`, `trial.suggest_int`, `study.optimize`). These tune learning rate, batch size, neurons, epochs and so on.

## 9. CNNs & transfer learning → [Module 07](07_CNNs_and_Transfer_Learning/README.md)
```
 image ─► [Conv2d ─► ReLU ─► MaxPool] x N ─► Flatten ─► Linear … ─► classes
 Output size of a conv: (W − K + 2P)/S + 1   e.g. (28 − 3 + 2)/1 + 1 = 28, then pool/2 → 14
```
Transfer learning: load `resnet18(weights=IMAGENET1K_V1)`, freeze it with `requires_grad=False`, replace `model.fc`, and train only the new head.

## 10. Transformers & BERT → [Module 08](08_Transformers_and_BERT/README.md)
* **Tokenizer** → `input_ids` + `attention_mask` (padding = 0).
* **Self-attention:** `softmax(QKᵀ/√d)·V`, so every token can attend to every other token.
* **BERT** = 12 stacked encoder layers, pre-trained on masked-word prediction. The `[CLS]` vector summarises the sentence.
* **Fine-tuning** (`BertForSequenceClassification`, AdamW lr 1e-5) vs **feature extraction** (frozen BERT plus a small head).

## 11. Evaluating
`accuracy = correct / total`, `sklearn.metrics.confusion_matrix` + `seaborn.heatmap`, and train/val loss curves with `matplotlib`. Use `model.eval()` and `torch.no_grad()` when evaluating.
