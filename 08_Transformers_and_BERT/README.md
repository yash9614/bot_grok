# 08 · Transformers and BERT

| Notebook | Task | Data |
|---|---|---|
| [`Transformers_COVID_Tweets_Solutions.ipynb`](Transformers_COVID_Tweets_Solutions.ipynb) | Is a tweet COVID-19 related? (fine-tune `BertForSequenceClassification`, `bert-base-cased`) | [`covid_twitter_dataset_codebasics_DL.csv`](covid_twitter_dataset_codebasics_DL.csv) (5,287 tweets: `ID, text, target`) |
| [`BERT_Spam_Classification.ipynb`](BERT_Spam_Classification.ipynb) | SMS spam vs ham (frozen `bert-base-uncased` + small classifier head) | [`spam.csv`](spam.csv) (5,572 SMS: `Category, Message`). Credits: [`dataset_credits_readme.txt`](dataset_credits_readme.txt) |

> 💡 Run on a GPU. Both notebooks download BERT weights (~420 MB) from Hugging Face the first time.

## Transformers in one picture
```
 "Vaccines are now available"                       
        │ tokenizer (WordPiece)                          
        ▼                                                
 [CLS] vaccines are now available [SEP] [PAD] [PAD]   + attention_mask 1 1 1 1 1 1 0 0
        │ token + position embeddings
        ▼
 ┌───────── Transformer encoder layer  (x12 in BERT-base) ─────────┐
 │  Multi-head SELF-ATTENTION: every token looks at every other    │
 │  token and decides how much to "pay attention" to it:           │
 │        Attention(Q,K,V) = softmax(Q·Kᵀ / √d) · V                │
 │  ─► Add & LayerNorm ─► Feed-forward ─► Add & LayerNorm          │
 └─────────────────────────────────────────────────────────────────┘
        │
        ▼
 [CLS] vector (768 numbers = summary of the whole sentence) ─► classifier ─► COVID / not COVID
```
* **Self-attention** lets "available" connect to "vaccines" no matter how far apart they are, unlike RNNs that read word by word.
* **BERT** (Bidirectional Encoder Representations from Transformers) is a stack of encoder layers pre-trained on huge text corpora by predicting masked words. Then we **fine-tune** it on our small labelled dataset.
* **Tokenizer outputs:** `input_ids` (token numbers, `101 = [CLS]`, `102 = [SEP]`, `0 = [PAD]`) and `attention_mask` (1 = real token, 0 = padding, so it is ignored).
* **cased vs uncased:** `bert-base-cased` keeps capital letters (useful for names like "COVID"). `bert-base-uncased` lower-cases everything.

## Notebook 1: COVID tweets (full fine-tuning)
1. 70/30 `train_test_split`, then `AutoTokenizer.from_pretrained("bert-base-cased")` with padding and truncation.
2. A custom **`TokenData(Dataset)`** class returns `{input_ids, token_type_ids, attention_mask, labels}` per row, so `DataLoader` can batch it (batch size 30).
3. `BertForSequenceClassification` adds a fresh 2-class head on top of `[CLS]`. Optimizer `AdamW(lr=1e-5)` (a tiny lr so the pretrained knowledge isn't destroyed), `CrossEntropyLoss`, 3 epochs.
4. Training loss 0.43 → 0.22 → 0.14. **Test accuracy 88.53%**.
5. `predict_tweet()` runs inference on new tweets with softmax confidence.
6. *Bonus confusion matrix*: a cell was **added (not yet executed)** using the notebook's `all_preds` / `all_labels`.

## Notebook 2: SMS spam (feature extraction)
1. Map `ham → 0`, `spam → 1`. The data is imbalanced (4,825 ham vs 747 spam), so it's **balanced** by keeping all spam and sampling 1,000 ham.
2. 80/20 split. `BertTokenizer` (`bert-base-uncased`, `max_length=128`).
3. The model `SentimentClassifier` **freezes all BERT layers** and trains only `Linear(768→256) → ReLU → Dropout(0.3) → Linear(256→1) → Sigmoid` on the `[CLS]` token embedding (`last_hidden_state[:, 0, :]`). Adam lr 0.001, `BCELoss`, 2 epochs.
4. **Validation accuracy 97.4%**. `predict()` labels new messages such as *"This is your last chance to WIN cash…"*.

### Fine-tuning vs feature extraction
```
 Fine-tuning (notebook 1)          Feature extraction (notebook 2)
 BERT weights  ✎ updated           BERT weights  🔒 frozen
 head          ✎ updated           head          ✎ updated
 slower, usually more accurate     fast, cheap, great for small data
```

## Clean-up notes
* COVID notebook: the Colab path `/content/covid_twitter_dataset_codebasics_DL.csv` became the relative path.
* Spam notebook (`Download files/bert_spam_classification.ipynb`) was **already fully solved**. The only change was removing two unused imports: `AdamW` from `transformers` (no longer exists in recent versions and would crash the first cell) and `datasets.load_dataset`.
* Dataset credit: SMS Spam Collection v.1, Tiago A. Almeida and José María Gómez Hidalgo, [UCI ML Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).
