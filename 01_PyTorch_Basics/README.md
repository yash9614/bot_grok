# 01 · PyTorch Basics: "Welcome to AI Town!"

**Notebook:** [`PyTorch_Basics_Solutions.ipynb`](PyTorch_Basics_Solutions.ipynb)
**Data:** none (all tensors are created inside the notebook)

You play an AI engineer in *AI Town* and solve 6 small tasks. Each one teaches a core PyTorch building block.

| Task | Story | Concept you practise |
|---|---|---|
| 1 | Inventory management | `torch.tensor`, `sum(dim=0)`, `mean(dim=1)` (needs a float dtype) |
| 2 | Vehicle flow on 2 roads | `torch.randint`, element-wise `+` |
| 3 | Fitness matrix | broadcasting a weight vector, `max(dim=…)`, `.t()` transpose |
| 4 | Chain rule in action | `requires_grad=True`, `y.backward()`, `x.grad` |
| 5 | Camera calibration | `*` (Hadamard) vs `torch.matmul`, `torch.det` |
| 6 | Neural network foundations | a single neuron `y = w·x + b` |

## Concepts, teacher-style

### 1. A tensor is a NumPy array that can live on a GPU and remember its math
```
 inventory_tensor (3 days x 5 items)
            item0 item1 item2 item3 item4
 day 0  [   12    15    10     0     5  ]   -> mean(dim=1) = average per DAY
 day 1  [   10     8     7     5     4  ]
 day 2  [   20    10    15     5     2  ]
            |                               
            v sum(dim=0) = total per ITEM  ->  [42, 33, 32, 10, 11]
```
* `dim=0` collapses the rows (you get one number per column).
* `dim=1` collapses the columns (one number per row).
* `mean()` only works on floating-point tensors, so use `.float().mean(...)` or `mean(..., dtype=torch.float32)`.

### 2. Broadcasting
`weights = [0.8, 1.2, 1.5]` (shape 3) times `fitness` (shape 3×3): PyTorch "stretches" the vector across every row, so each column (strength, stamina, flexibility) gets its own weight. You don't need a loop.

### 3. Autograd: PyTorch does calculus for you
For `y = 2x³ + 5x² − 3x + 7` at `x = 4`:
```
 x (requires_grad=True) ──► y = 2x³+5x²−3x+7 ──► y.backward()
                                                   │
          x.grad = dy/dx = 6x² + 10x − 3 = 133  ◄──┘
```
* PyTorch records every operation in a **computational graph** and walks it backwards with the chain rule.
* The graph is freed after `.backward()`. Pass `retain_graph=True` if you need to call backward on it again.

### 4. Element-wise vs matrix multiplication
```
 [1 2] * [5 6] = [ 5 12]        torch.matmul([1 2],[5 6]) = [19 22]
 [3 4]   [7 8]   [21 32]                     [3 4] [7 8]    [43 50]
   (Hadamard: same position)       (rows · columns)
```
`torch.det` needs a float, square matrix.

### 5. The neuron: `y = w·x + b`
This one line is the "atom" of every network in the later modules. `w` and `b` are created with `requires_grad=True` because those are the values training will learn.

## Clean-up note
Task 2 / step 3 originally ended with `print()` (an empty print), so the hourly total was never shown. It now prints `total_flow_hourly`. That cell's old output was cleared, so re-run it to see the result.
