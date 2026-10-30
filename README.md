# Hate Speech Detection (NLP)

Baseline pipeline for detecting hate/offensive text: text cleaning (noise reduction) -> TF-IDF (1-2 grams) -> Logistic Regression.

## Run
```bash
pip install -r requirements.txt
python train.py                # uses data_sample.csv
python train.py your_data.csv  # CSV with columns: text,label (1 = hate/offensive, 0 = normal)
python predict.py "your sentence here"
```

`data_sample.csv` is a tiny demo set. For real results use public datasets (e.g., Davidson, HateXplain).

## Next steps
Extend with hybrid deep learning (CNN + BiLSTM / BERT embeddings) and compare against this baseline.
