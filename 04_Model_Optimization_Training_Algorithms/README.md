# 04 · Model Optimization: Training Algorithms (Optimizers)

| Notebook | What it covers |
|---|---|
| [`Part1_Optimizers_on_FashionMNIST_Solutions.ipynb`](Part1_Optimizers_on_FashionMNIST_Solutions.ipynb) | A fully connected FashionMNIST classifier trained with **SGD → SGD+Momentum → RMSProp → Adam (bonus)** |
| [`Part2_Optimizers_From_Scratch_Solutions.ipynb`](Part2_Optimizers_From_Scratch_Solutions.ipynb) | **Bonus exercise:** Gradient Descent, Momentum, RMSProp and Adam written by hand in plain Python, with the derivations |

**Data:** FashionMNIST is downloaded automatically by `torchvision` into `./data` (git-ignored). Part 1 uses a 12,000-image train subset and a 3,000-image test subset, with batch size 32.

## Part 1: the model
```
 28x28 image ─► Flatten(784) ─► Linear 128 ─► ReLU ─► Linear 64 ─► ReLU ─► Linear 10 ─► Softmax
                                                                                 (10 clothing classes)
```
| Optimizer (10 epochs each) | Settings | Saved test accuracy |
|---|---|---|
| SGD | lr 0.01, momentum 0 | 70.00 % |
| SGD + Momentum | lr 0.01, momentum 0.9 | 80.50 % |
| RMSProp | lr 0.001, alpha 0.9 | 80.07 % |
| Adam *(bonus, added)* | lr 0.001 | run it yourself |

## The four optimizers, teacher-style
Imagine rolling a ball down a hilly loss surface to reach the lowest valley.

```
 Plain GD     : step = −η·g                              (walks straight downhill, can zig-zag)
 Momentum     : v = β·v + (1−β)·g ;  w −= η·v           (builds up speed, smooths zig-zags)
 RMSProp      : s = β·s + (1−β)·g² ; w −= η·g/√(s+ε)    (each weight gets its own step size)
 Adam         : m = β1·m + (1−β1)·g        ← momentum
                v = β2·v + (1−β2)·g²       ← RMSProp
                m̂ = m/(1−β1ᵗ), v̂ = v/(1−β2ᵗ)   ← bias correction
                w −= η·m̂/(√v̂ + ε)
```

### Part 2 problems (all solved by hand)
| # | Problem | Function | Result in notebook |
|---|---|---|---|
| 1 | Gradient descent, demand forecasting | `L(w)=(w−4)²`, w₀=0, η=0.1, 10 steps | w → 3.5705 (heading to 4) |
| 2 | Momentum, supply-chain contour | `f(x,y)=x²+3y²`, start (2,2), η=0.1, β=0.9, 5 steps | (1.4986, 0.6351) |
| 3 | RMSProp, dynamic pricing | `f(w)=w²+5`, w₀=5, η=0.01, β=0.9, 15 steps | w → 4.7704 |
| 4 | Adam, restocking | `f(x)=x⁴−3x³+2`, x₀=3, η=0.01, β1=0.9, β2=0.09, 19 steps | x → 2.7937 |

The steps are small because η is small. With more iterations each one keeps sliding toward the minimum.

## 🧑‍🏫 Teacher's notes
* **Softmax + CrossEntropyLoss:** `nn.CrossEntropyLoss` already applies log-softmax internally. Adding a `Softmax` layer on top squashes the logits twice, which is why the loss plateaus around 1.6. Removing the final `Softmax` usually trains faster.
* **Reused model:** the same `model` instance continues training under each optimizer, so later optimizers start from already-trained weights. For a fair race, re-create `model = FashionNN()` before each one.

## Clean-up notes
* Original names `…Alogrithms_Excerise_1_solution` and `…Excerise_2_Solution 2` were renamed to the clean names above.
* The notebook ended with an unanswered **BONUS: Adam** prompt. An Adam cell, built from the same loop the notebook uses, was **added (not yet executed)**.
