import sys, joblib
from preprocess import clean_text

model = joblib.load("model.joblib")
text = " ".join(sys.argv[1:]) or "have a nice day"
label = model.predict([clean_text(text)])[0]
print("Hate/Offensive" if label == 1 else "Normal")
