import pandas as pd
import random
from sacrebleu import corpus_bleu
from sacrebleu.metrics import TER
from nltk.translate.meteor_score import meteor_score
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

# --- Load dataset ---
csv_path = os.path.join(os.path.dirname(__file__), "medev_test_aligned.csv")
aligned_df = pd.read_csv(csv_path)
print(f"Loaded {len(aligned_df)} sentence pairs.")

# --- Model setup ---
model_name = "vinai/vinai-translate-vi2en-v2"
tokenizer_en2vi = AutoTokenizer.from_pretrained(model_name, src_lang="vi_VN")
model_en2vi = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = model.to(device)
#--- Translation function ---
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
# --- Random sampling with inline filtering ---
random.seed(42)
num_samples = 100
preds, refs = [], []
count = 0

while count < num_samples:
    # pick a random row index
    idx = random.randint(0, len(aligned_df) - 1)
    tgt_text = str(aligned_df["vi"][idx])

    # check length condition
    word_count = len(tgt_text.split())
    if   50 <= word_count:
        src = aligned_df["vi"][idx]
        tgt = aligned_df["en"][idx]

        # run translation
        #inputs = tokenizer(src, return_tensors="pt", truncation=True, padding=True).to(device)
        #outputs = model.generate(**inputs, max_length=256, num_beams=5)
        pred = translate_en2vi(src)
        preds.append(pred)
        refs.append(tgt)
        count += 1
        print(f"→ Collected {count}/{num_samples} valid samples", end="\r")

print(f"\nFinished sampling {num_samples} medium-length sentences!")

# --- Compute BLEU ---
bleu = corpus_bleu(preds, [refs])
print(f"\nBLEU Score (over 50 words): {bleu.score:.2f}")
