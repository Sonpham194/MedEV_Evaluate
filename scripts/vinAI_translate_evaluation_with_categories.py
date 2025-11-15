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
model_name = "vinai/vinai-translate-en2vi"
#tokenizer_en2vi = AutoTokenizer.from_pretrained(model_name, src_lang="vi_VN")
#model_en2vi = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer_en2vi = AutoTokenizer.from_pretrained(model_name, src_lang ="en_XX")
model_en2vi = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = model.to(device)
#--- Translation function ---
def translate_en2vi(en_text: str) -> str:
    input_ids = tokenizer_en2vi(en_text, return_tensors="pt").input_ids
    output_ids = model_en2vi.generate(
        input_ids,
        decoder_start_token_id=tokenizer_en2vi.lang_code_to_id["en_XX"],
        num_return_sequences=1,
        num_beams=5,
        early_stopping=True
    )
    vn_text = tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
    vn_text = " ".join(vn_text)
    return vn_text
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
    #inputs = tokenizer(src, return_tensors="pt", truncation=True, padding=True).to(device)
    #outputs = model.generate(**inputs, max_length=256, num_beams=5)
    pred = translate_en2vi(src)
    preds.append(pred)
    refs.append(tgt)
    count += 1
    print(f"→ Collected {count}/{num_samples} valid samples", end="\r")

print(f"\nFinished sampling {num_samples} sentences!")

# --- Compute BLEU ---
bleu = corpus_bleu(preds, [refs])
print(f"\nBLEU Score: {bleu.score:.2f}")
