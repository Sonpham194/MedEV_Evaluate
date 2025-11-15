import pandas as pd
import random
from sacrebleu.metrics import TER
from nltk.translate.meteor_score import meteor_score
from nltk import word_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import nltk

# Download required resources for METEOR
nltk.download("wordnet")
nltk.download("omw-1.4")

# --- Load dataset ---
csv_path = os.path.join(os.path.dirname(__file__), "medev_test_aligned.csv")
aligned_df = pd.read_csv(csv_path)
print(f"Loaded {len(aligned_df)} sentence pairs.")

# --- Model setup ---
model_name = "VietAI/envit5-translation"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# --- Random sampling ---
random.seed(42)
num_samples = 100
preds, refs = [], []

for i in range(num_samples):
    idx = random.randint(0, len(aligned_df) - 1)
    src = str(aligned_df["en"][idx])  # input: English
    tgt = str(aligned_df["vi"][idx])  # target: Vietnamese (reference)

    # Run translation
    inputs = tokenizer(src, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model.generate(**inputs, max_length=256, num_beams=5)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(tgt)
    print(pred)
    preds.append(pred)
    refs.append(tgt)

    print(f"→ Collected {i+1}/{num_samples} valid samples", end="\r")

print(f"\nFinished sampling {num_samples} medium-length sentences!")

# ----------------------------------------
# 1. METEOR Score
# ----------------------------------------
meteor_scores = [
    meteor_score([word_tokenize(ref)], word_tokenize(hyp))
    for ref, hyp in zip(refs, preds)
]
avg_meteor = sum(meteor_scores) / len(meteor_scores)
print(f"Average METEOR score: {avg_meteor:.4f}")

# ----------------------------------------
# 2. TER Score
# ----------------------------------------
ter = TER()
ter_score = ter.corpus_score(preds, [refs])
print(f"TER score: {ter_score.score:.2f}")
