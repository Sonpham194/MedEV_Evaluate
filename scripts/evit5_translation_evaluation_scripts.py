import pandas as pd
import random
from sacrebleu import corpus_bleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

# --- Load dataset ---
csv_path = os.path.join(os.path.dirname(__file__), "medev_test_aligned.csv")
aligned_df = pd.read_csv(csv_path)
print(f"Loaded {len(aligned_df)} sentence pairs.")
print(f"This is the new test")

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
    tgt_text = str(aligned_df["en"][idx])

    # check length condition
    word_count = len(tgt_text.split())
    if 10 <= word_count <= 20:
        src = aligned_df["vi"][idx]
        tgt = aligned_df["en"][idx]

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
print(f"\nBLEU Score (10–20 words): {bleu.score:.2f}")
