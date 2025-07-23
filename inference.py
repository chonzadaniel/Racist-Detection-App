# import os
# import yaml
# import torch
# import numpy as np
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from symspellpy import SymSpell
# from text_preprocessor import preprocess_tweet
#
# # ---------------------------- Environment Setup and HuggingFace API Key ---------------------------
# def load_api_keys():
#     with open("huggingface_credentials.yml", "r") as f:
#         huggingface_keys = yaml.safe_load(f)
#     hf_key = huggingface_keys.get("HUGGINGFACE_API_KEY", "")
#     if not hf_key:
#         raise ValueError("HUGGINGFACE_API_KEY missing in huggingface_credentials.yml")
#
#     os.environ["HUGGINGFACE_API_KEY"] = hf_key
#     os.environ["TOKENIZERS_PARALLELISM"] = "false"
#     os.environ["TRANSFORMERS_OFFLINE"] = "1"
#     os.environ["HF_TOKEN"] = hf_key
#     return hf_key
#
# HUGGINGFACE_API_KEY = load_api_keys()
#
# # ---------------------------- Constants ---------------------------
# OUTPUT_DIR = 'bertweet_train_dir'
# MODEL_DIR = os.path.join(OUTPUT_DIR, 'vinai-bertweet-base')
# # DICT_PATH = 'frequency_dictionary_en_82_765.txt'
# DICT_PATH = "/Users/emmanueldanielchonza/Documents/Sentiment_Analysis/frequency_dictionary_en_82_765.txt"
#
# LABEL2ID = {'non_racist': 0, 'racist': 1}
# ID2LABEL = {v: k for k, v in LABEL2ID.items()}
#
# # ---------------------------- Initialize SymSpell ---------------------------
# if not os.path.exists(DICT_PATH):
#     raise FileNotFoundError(f"{DICT_PATH} not found. Please download it and place it in the project root.")
#
# sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
# sym_spell.load_dictionary(DICT_PATH, term_index=0, count_index=1)
# print(f"SymSpell dictionary loaded from {DICT_PATH}.")
#
# # ---------------------------- Load tokenizer and model ---------------------------
# tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
# model.eval()
# print(f"Model loaded from {MODEL_DIR}.")
#
# # ---------------------------- Prediction function ---------------------------
# def predict(tweet: str):
#     """
#     Predicts the label and confidence for a single tweet.
#     Returns: (label_id, label_name, confidence)
#     """
#     # Here you can optionally also use sym_spell inside preprocess_tweet
#     cleaned_tweet = preprocess_tweet(tweet)
#
#     tokens = tokenizer(
#         cleaned_tweet,
#         truncation=True,
#         padding='max_length',
#         max_length=128,
#         return_tensors='pt'
#     )
#
#     with torch.no_grad():
#         outputs = model(**tokens)
#         logits = outputs.logits
#         probs = torch.nn.functional.softmax(logits, dim=1).numpy()[0]
#
#     pred = int(np.argmax(probs))
#     confidence = float(probs[pred])
#
#     return pred, ID2LABEL[pred], confidence

import os
import json
import yaml
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from symspellpy import SymSpell
from text_preprocessor import preprocess_tweet

# ---------------------------- Environment Setup and HuggingFace API Key ---------------------------
def load_api_keys():
    with open("huggingface_credentials.yml", "r") as f:
        huggingface_keys = yaml.safe_load(f)
    hf_key = huggingface_keys.get("HUGGINGFACE_API_KEY", "")
    if not hf_key:
        raise ValueError("HUGGINGFACE_API_KEY missing in huggingface_credentials.yml")

    os.environ["HUGGINGFACE_API_KEY"] = hf_key
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_TOKEN"] = hf_key
    return hf_key

HUGGINGFACE_API_KEY = load_api_keys()

# ---------------------------- Constants ---------------------------
OUTPUT_DIR = 'bertweet_train_dir2'
DICT_PATH = "/Users/emmanueldanielchonza/Documents/Sentiment_Analysis/frequency_dictionary_en_82_765.txt"

LABEL2ID = {'non_racist': 0, 'racist': 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ---------------------------- Initialize SymSpell ---------------------------
if not os.path.exists(DICT_PATH):
    raise FileNotFoundError(f"{DICT_PATH} not found. Please download it and place it in the project root.")

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary(DICT_PATH, term_index=0, count_index=1)
print(f"SymSpell dictionary loaded from {DICT_PATH}.")

# ---------------------------- Select best fold model ---------------------------
fold_metrics = {}

for fold_dir in os.listdir(OUTPUT_DIR):
    fold_path = os.path.join(OUTPUT_DIR, fold_dir)
    if os.path.isdir(fold_path):
        eval_path = os.path.join(fold_path, "eval_results.json")
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                metrics = json.load(f)
                f1 = metrics.get("eval_f1")
                if f1 is not None:
                    fold_metrics[fold_dir] = f1

if not fold_metrics:
    raise ValueError("No valid eval_results.json files found in OUTPUT_DIR.")

best_fold = max(fold_metrics, key=fold_metrics.get)
MODEL_DIR = os.path.join(OUTPUT_DIR, best_fold)

print(f"Best model selected: {best_fold} with F1 = {fold_metrics[best_fold]:.4f}")
print(f"Loading model from: {MODEL_DIR}")

# ---------------------------- Load tokenizer and model ---------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
print(f"Model loaded and ready for inference.")

# ---------------------------- Prediction function ---------------------------
def predict(tweet: str):
    """
    Predicts the label and confidence for a single tweet.
    Returns: (label_id, label_name, confidence)
    """
    # Optionally use sym_spell in preprocess_tweet
    cleaned_tweet = preprocess_tweet(tweet)

    tokens = tokenizer(
        cleaned_tweet,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )

    with torch.no_grad():
        outputs = model(**tokens)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).numpy()[0]

    pred = int(np.argmax(probs))
    confidence = float(probs[pred])

    return pred, ID2LABEL[pred], confidence