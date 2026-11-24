# bot_grok · Deep Learning with PyTorch: Solved Exercises & Concept Notes

Solved notebooks, datasets and beginner-friendly concept guides for a hands-on **Deep Learning with PyTorch** track (Codebasics DL course exercises). It takes you from your first tensor all the way to fine-tuning BERT.

Every module folder contains:
* ✅ the **solved notebook(s)**, with outputs from a real run,
* 📄 the **dataset(s)** it needs, placed right next to it so `pd.read_csv("file.csv")` just works,
* 📘 a **README** that explains the concepts teacher-style, with diagrams.

For a one-page revision sheet, see **[CONCEPTS.md](CONCEPTS.md)**.

---

## 🗺️ Learning path
```
  ┌──────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
  │ 01 PyTorch Basics│──►│ 02 Neural Networks   │──►│ 03 NN Training       │
  │ tensors, autograd│   │ nn.Module, DataLoader│   │ backprop, GD flavours│
  └──────────────────┘   └──────────────────────┘   └──────────┬───────────┘
                                                               │
            ┌──────────────── Model Optimization ──────────────┼───────────────┐
            │  ┌──────────────────┐  ┌──────────────────┐  ┌───▼──────────────┐│
            │  │ 06 Hyperparameter│◄─│ 05 Regularization│◄─│ 04 Optimizers    ││
            │  │ grid/random/Optuna│ │ L1/L2/Dropout/BN │  │ SGD→Adam         ││
            │  └────────┬─────────┘  └──────────────────┘  └──────────────────┘│
            └───────────┼──────────────────────────────────────────────────────┘
                        ▼
  ┌──────────────────────────────┐   ┌──────────────────────────────────────┐
  │ 07 CNNs & Transfer Learning  │──►│ 08 Transformers & BERT               │
  │ Conv2d, pooling, ResNet18    │   │ self-attention, fine-tuning, spam    │
  └──────────────────────────────┘   └──────────────────────────────────────┘
```

## 📚 Modules
| # | Module | Solved notebook(s) | Dataset | Concept guide | Headline result (saved run) |
|---|---|---|---|---|---|
| 01 | PyTorch Basics | [PyTorch_Basics_Solutions](01_PyTorch_Basics/PyTorch_Basics_Solutions.ipynb) | none (generated) | [README](01_PyTorch_Basics/README.md) | 6 tensor/autograd tasks |
| 02 | Neural Networks in PyTorch | [Neural_Networks_PyTorch_Solutions](02_Neural_Networks_in_PyTorch/Neural_Networks_PyTorch_Solutions.ipynb) | [habitat_images](02_Neural_Networks_in_PyTorch/habitat_images_codebasics_DL.csv) | [README](02_Neural_Networks_in_PyTorch/README.md) | Threat detector: 97% accuracy |
| 03 | Neural Network Training | [Neural_Network_Training_Solutions](03_Neural_Network_Training/Neural_Network_Training_Solutions.ipynb) | [agriculture_dataset](03_Neural_Network_Training/agriculture_dataset_codebasics_DL.csv) | [README](03_Neural_Network_Training/README.md) | Manual vs autograd gradients match |
| 04 | Optimizers (Training Algorithms) | [Part 1: FashionMNIST](04_Model_Optimization_Training_Algorithms/Part1_Optimizers_on_FashionMNIST_Solutions.ipynb) · [Part 2: from scratch](04_Model_Optimization_Training_Algorithms/Part2_Optimizers_From_Scratch_Solutions.ipynb) | FashionMNIST (auto-download) | [README](04_Model_Optimization_Training_Algorithms/README.md) | SGD 70% → Momentum 80.5% |
| 05 | Regularization | [Regularization_Techniques_Solutions](05_Model_Optimization_Regularization/Regularization_Techniques_Solutions.ipynb) | [AtliQ_Churn](05_Model_Optimization_Regularization/AtliQ_Churn_Prediction_Codebasics_DL.csv) | [README](05_Model_Optimization_Regularization/README.md) | Base vs Dropout vs L2 vs BatchNorm |
| 06 | Hyperparameter Tuning | [Hyperparameter_Tuning_Solutions](06_Model_Optimization_Hyperparameter_Tuning/Hyperparameter_Tuning_Solutions.ipynb) | FashionMNIST (auto-download) | [README](06_Model_Optimization_Hyperparameter_Tuning/README.md) | Tuned model: 77.1% |
| 07 | CNNs & Transfer Learning | [CNNs_FashionMNIST](07_CNNs_and_Transfer_Learning/CNNs_FashionMNIST_Solutions.ipynb) · [Transfer_Learning_Flowers102](07_CNNs_and_Transfer_Learning/Transfer_Learning_Flowers102_Solutions.ipynb) | FashionMNIST, Flowers-102 (auto-download) | [README](07_CNNs_and_Transfer_Learning/README.md) | CNN 80.2% · ResNet18 79.3% vs 19.6% from scratch |
| 08 | Transformers & BERT | [Transformers_COVID_Tweets](08_Transformers_and_BERT/Transformers_COVID_Tweets_Solutions.ipynb) · [BERT_Spam_Classification](08_Transformers_and_BERT/BERT_Spam_Classification.ipynb) | [covid_twitter](08_Transformers_and_BERT/covid_twitter_dataset_codebasics_DL.csv) · [spam](08_Transformers_and_BERT/spam.csv) | [README](08_Transformers_and_BERT/README.md) | COVID 88.5% · Spam 97.4% |
| — | Extra material | [random_files/](random_files/) | — | [README](random_files/README.md) | OpenAI first-app notes, handwritten ML notes, ML crash course |

## 🚀 Setup
**Option A: Google Colab (easiest, free GPU)**
1. Open a notebook on GitHub, then use *Open in Colab* (or upload it at colab.research.google.com).
2. `Runtime → Change runtime type → T4 GPU` (recommended for modules 07 and 08).
3. Upload the module's `.csv` file(s) to the Colab file panel. The notebooks read them with a plain relative path.

**Option B: run locally**
```bash
git clone https://github.com/yash9614/bot_grok.git
cd bot_grok
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```
Always start Jupyter **from the module folder** (or open the notebook so its folder is the working directory) so the CSV paths resolve. FashionMNIST, Flowers-102 and the BERT weights download automatically on first run (into a git-ignored `data/` folder and the Hugging Face cache).

## 📁 Repository layout
```
bot_grok/
├── README.md · CONCEPTS.md · requirements.txt · .gitignore
├── 01_PyTorch_Basics/
├── 02_Neural_Networks_in_PyTorch/
├── 03_Neural_Network_Training/
├── 04_Model_Optimization_Training_Algorithms/
├── 05_Model_Optimization_Regularization/
├── 06_Model_Optimization_Hyperparameter_Tuning/
├── 07_CNNs_and_Transfer_Learning/
├── 08_Transformers_and_BERT/
└── random_files/
```

## 🙏 Dataset credits
* `*_codebasics_DL.csv` datasets (habitat images, agriculture, AtliQ churn, COVID tweets) come from the **Codebasics Deep Learning course** exercises.
* `spam.csv` is the **SMS Spam Collection v.1** by Tiago A. Almeida and José María Gómez Hidalgo ([UCI ML Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)). See [dataset_credits_readme.txt](08_Transformers_and_BERT/dataset_credits_readme.txt).
* **FashionMNIST** (Zalando Research) and **Oxford Flowers-102** (Visual Geometry Group, Oxford) are downloaded via `torchvision.datasets`.
* Pretrained models: **ResNet18** (torchvision, ImageNet) and **BERT** `bert-base-cased` / `bert-base-uncased` (Google, via Hugging Face).

## 🧹 Notes on this clean-up
* Only the **solved** notebooks are included. Duplicate copies were removed and typo'd file names were fixed (`Alogrithms`, `Excerise`, `Tunning`).
* Colab-only paths like `/content/file.csv` became relative paths, and each CSV sits next to its notebook.
* Small fixes: an empty `print()` in Module 01, two unused/broken imports in the spam notebook, and two **bonus** cells that were left unanswered (Adam in Module 04 and a confusion matrix in Module 08). The bonus cells are added but **not yet executed**.
* Colab widget metadata was stripped so the notebooks render on GitHub.
