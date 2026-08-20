# Pretrained Model Challenge — Electronics Reviews

An independent build of the "Pretrained Model Challenge" brief: compare pretrained sentiment models
on review text, and classify reviews by topic, without any fine-tuning.

## What's here

- **`Pretrained_Model_Challenge_Electronics.ipynb`** — the full analysis: synthetic e-commerce
  electronics review dataset, three pretrained sentiment models evaluated with
  accuracy/F1/confusion matrices, error analysis, and a topic-classification task solved with
  sentence-embedding similarity instead of zero-shot NLI.
- **`app.py`** — a Streamlit app that runs the same models interactively: type in (or pick an example)
  review and get a live sentiment + topic prediction.
- **`requirements.txt`** — pinned dependencies for both the notebook and the app.

## Approach

**Sentiment (3 models, no fine-tuning):**
- `siebert/sentiment-roberta-large-english` — general-purpose binary POSITIVE/NEGATIVE
- `finiteautomata/bertweet-base-sentiment-analysis` — POS/NEG/NEU, tuned on informal/social text
- `j-hartmann/emotion-english-distilroberta-base` — 7-way emotion classifier, repurposed for polarity
  via a joy/surprise → positive, anger/disgust/fear/sadness/neutral → negative mapping

Each model returns labels in a different format. A single `normalize_label` function reads the label
text itself (prefix match + an emotion→polarity map) instead of a lookup table keyed by model ID, so
it isn't hardcoded to know which model produced which label.

**Topic classification:** instead of zero-shot NLI (`facebook/bart-large-mnli`-style), reviews and short
topic descriptions are both embedded with `sentence-transformers/all-MiniLM-L6-v2`, and each review is
assigned the topic with the closest embedding by cosine similarity. The gap between the top match and
the runner-up topic is tracked, so genuinely ambiguous misclassifications (two topics nearly tied) can
be told apart from cases where the method was just wrong.

## Running the notebook

Needs a Colab-style environment (GPU optional, but downloads ~2GB of model weights on first run).

```bash
pip install -r requirements.txt
```

Then run all cells top to bottom.

## Running the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

First launch downloads and caches the four models (~2GB total); subsequent launches are fast. Enter a
review or pick an example, then click **Analyze** to see the per-model sentiment breakdown, a majority
vote, and the predicted topic with its similarity gap to the runner-up.
