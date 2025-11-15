import pandas as pd
import random
from sacrebleu import corpus_bleu
from sacrebleu.metrics import TER
from nltk.translate.meteor_score import meteor_score
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

# --- Load dataset ---
csv_path = os.path.join(os.path.dirname(__file__), "classified_output/Article_Translations.csv")
aligned_df = pd.read_csv(csv_path)
print(f"Loaded {len(aligned_df)} sentence pairs.")

# --- Model setup ---
model_name = "VietAI/envit5-translation"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# --- Random sampling with inline filtering ---
random.seed(42)
num_samples = 100
preds, refs = [], []
count = 0

while count < num_samples:
    # pick a random row index
    idx = random.randint(0, len(aligned_df) - 1)

    # Removed length check; accept any sampled sentence
    src = aligned_df["en"][idx]
    tgt = aligned_df["vi"][idx]

        # run translation
    inputs = tokenizer(src, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model.generate(**inputs, max_length=256, num_beams=5)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    preds.append(pred)
    refs.append(tgt)
    count += 1
    print(f"→ Collected {count}/{num_samples} valid samples", end="\r")

print(f"\nFinished sampling {num_samples} medium-length sentences!")

# --- Compute BLEU ---
bleu = corpus_bleu(preds, [refs])
print(f"\nBLEU Score: {bleu.score:.2f}")
