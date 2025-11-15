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

# --- Load dataset ---
csv_path = os.path.join(os.path.dirname(__file__), "medev_test_aligned.csv")
aligned_df = pd.read_csv(csv_path)
print(f"Loaded {len(aligned_df)} sentence pairs.")

# --- Model setup ---
model_name = "vinai/vinai-translate-en2vi"
tokenizer_en2vi = AutoTokenizer.from_pretrained(model_name, src_lang="en_XX")
model_en2vi = AutoModelForSeq2SeqLM.from_pretrained(model_name)
def translate_en2vi(vn_text: str) -> str:
    input_ids = tokenizer_en2vi(vn_text, return_tensors="pt").input_ids
    output_ids = model_en2vi.generate(
        input_ids,
        decoder_start_token_id=tokenizer_en2vi.lang_code_to_id["en_XX"],
        num_return_sequences=1,
        num_beams=5,
        early_stopping=True
    )
    en_text = tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
    en_text = " ".join(en_text)
    return en_text
# --- Random sampling ---
random.seed(42)
num_samples = 100
preds, refs = [], []

for i in range(num_samples):
    idx = random.randint(0, len(aligned_df) - 1)
    src = str(aligned_df["en"][idx])  # input: English
    tgt = str(aligned_df["vi"][idx])  # target: Vietnamese (reference)

    # Run translation
    pred = translate_en2vi(src)
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
