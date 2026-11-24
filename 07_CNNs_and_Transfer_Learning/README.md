# 07 · CNNs and Transfer Learning

| Notebook | Story | Data (auto-downloaded to `./data` / `data/`) |
|---|---|---|
| [`CNNs_FashionMNIST_Solutions.ipynb`](CNNs_FashionMNIST_Solutions.ipynb) | AtliQ "Fashion Classification System" | FashionMNIST (10k train / 3k test subset) |
| [`Transfer_Learning_Flowers102_Solutions.ipynb`](Transfer_Learning_Flowers102_Solutions.ipynb) | AtliQ "Smart Florist Assistant" | Oxford Flowers-102 (~345 MB download) |

> 💡 A GPU (for example Colab → Runtime → T4 GPU) is strongly recommended for the Flowers-102 notebook.

## Part A: a CNN from scratch (FashionMNIST)
```
 1x28x28 ─► Conv2d(1→32, 3x3, pad 1) ─► ReLU ─► MaxPool 2x2 ─► 32x14x14
          ─► Conv2d(32→64, 3x3, pad 1)─► ReLU ─► MaxPool 2x2 ─► 64x7x7
          ─► Flatten (3136) ─► Linear 128 ─► ReLU ─► Dropout 0.5 ─► Linear 10 ─► LogSoftmax
```
* **Convolution**: a small 3×3 filter slides over the image and lights up where it finds a pattern. Early layers find edges and later layers find textures and shapes. The same filter is reused everywhere, so there are far fewer weights than a dense layer.
* **Padding = 1** keeps the 28×28 size after a 3×3 conv.
* **Max pooling 2×2** keeps the strongest signal in each 2×2 window, halving height and width (28 → 14 → 7).
* **Data augmentation**: `RandomHorizontalFlip` + `RandomCrop(28, padding=4)` create slightly different copies of each image, which helps the model generalise.
* **Training**: Adam (lr 0.001, betas 0.9/0.99), CrossEntropyLoss, 10 epochs. **Test accuracy 80.23%** in the saved run.

## Part B: transfer learning (Flowers-102)
```
         ┌────────────── ResNet18 pretrained on ImageNet ───────────────┐
 image ─►│ conv blocks … (FROZEN: requires_grad = False)                │─► 512 features ─► NEW fc: Linear(512→102)
 224x224 └──────────────────────────────────────────────────────────────┘                    (only this is trained)
```
| Approach | Train acc. | Validation acc. |
|---|---|---|
| CNN from scratch (2 conv blocks, Dropout 0.5, 8 epochs) | 57.55 % | **19.61 %** (badly overfits, too little data) |
| ResNet18, frozen backbone + new `fc` (Adam lr 0.001, 6 epochs) | 96.08 % | **79.31 %** |

**Why it works:** a network trained on 1.2 M ImageNet photos already knows edges, colours, petals and textures. We keep that knowledge, freeze it, and only teach a new final layer to map those features to the 102 flower classes. That means less data, less compute, and much better accuracy.

Key code:
```python
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for p in model.parameters(): p.requires_grad = False      # freeze
model.fc = nn.Linear(model.fc.in_features, 102)           # new head
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)   # train only the head
```
Images are resized to 224×224 and normalised with ImageNet mean/std `[0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]`.

## 🧑‍🏫 Teacher's note
The FashionMNIST CNN ends in `LogSoftmax` and uses `CrossEntropyLoss`, which applies log-softmax again. The textbook pairing is either `LogSoftmax` + `nn.NLLLoss` or *no* final activation + `CrossEntropyLoss`.

## Clean-up note
The notebooks were renamed to say which dataset each one uses. The code is unchanged.
