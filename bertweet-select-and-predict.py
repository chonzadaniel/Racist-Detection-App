# import os
# import json
# import pandas as pd
# from datasets import Dataset
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# import numpy as np
# from tqdm import tqdm
#
# # ---------------- CONFIG ----------------
# TRAIN_DIR = "bertweet_train_dir2"
# TEST_CSV = '/Users/emmanueldanielchonza/Documents/Sentiment_Analysis/data/df_test_cleaned_final.csv'
# SUBMISSION_CSV = os.path.join(TRAIN_DIR, "bertweet_predictions_submission_final.csv")
#
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# LABEL2ID = {'non_racist': 0, 'racist': 1}
# ID2LABEL = {0: 'non_racist', 1: 'racist'}
#
# # ---------------- STEP 1: Find best fold by eval F1 ----------------
# fold_metrics = {}
#
# for i in range(1, 6):
#     fold_name = f"fold_{i}"
#     metrics_path = os.path.join(TRAIN_DIR, fold_name, "eval_results.json")
#     if os.path.exists(metrics_path):
#         with open(metrics_path) as f:
#             metrics = json.load(f)
#             f1 = metrics.get("eval_f1")
#             if f1 is not None:
#                 fold_metrics[i] = f1
#
# if not fold_metrics:
#     raise ValueError("No eval_results.json with 'eval_f1' found in any fold.")
#
# # Print scores and select best fold
# print("\nFold F1 scores:")
# for fold, score in fold_metrics.items():
#     print(f"  fold_{fold}: F1 = {score:.4f}")
#
# best_fold = max(fold_metrics, key=fold_metrics.get)
# print(f"\nBest model: fold_{best_fold} with F1 = {fold_metrics[best_fold]:.4f}")
#
# best_model_dir = os.path.join(TRAIN_DIR, f"vinai-bertweet-base-fold{best_fold}")
#
# # ---------------- STEP 2: Load best model & tokenizer ----------------
# print(f"\nLoading model from: {best_model_dir}")
# tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
# model = AutoModelForSequenceClassification.from_pretrained(best_model_dir).to(DEVICE)
#
# # ---------------- STEP 3: Load and prepare test data ----------------
# df_test = pd.read_csv(TEST_CSV)
# df_test['clean_tweet'] = df_test['clean_tweet'].fillna("no text available")
#
# test_ds = Dataset.from_pandas(df_test[['id', 'clean_tweet']], preserve_index=False)
#
# def tokenize(batch):
#     return tokenizer(batch['clean_tweet'], truncation=True, padding='max_length', max_length=128)
#
# test_ds = test_ds.map(tokenize, batched=True)
# test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
#
# # ---------------- STEP 4: Inference ----------------
# model.eval()
# preds = []
#
# with torch.no_grad():
#     for item in tqdm(test_ds, desc="🔍 Predicting"):
#         input_ids = item['input_ids'].unsqueeze(0).to(DEVICE)
#         attention_mask = item['attention_mask'].unsqueeze(0).to(DEVICE)
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         logits = outputs.logits
#         pred = torch.argmax(logits, dim=1).cpu().item()
#         preds.append(pred)
#
# # ---------------- STEP 5: Save submission ----------------
# df_test['label'] = preds
# df_test['label_name'] = df_test['label'].map(ID2LABEL)
#
# submission = df_test[['id', 'label']]
# submission.to_csv(SUBMISSION_CSV, index=False)
#
# print(f"\nSubmission file saved to: {SUBMISSION_CSV}")
# print("\nSample predictions:")
# print(df_test[['id', 'label', 'label_name']].head())

# import os
# import pandas as pd
# from datasets import Dataset
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# import numpy as np
# from tqdm import tqdm
#
# # ---------------- CONFIG ----------------
# BEST_MODEL_DIR = "bertweet_train_dir2/vinai-bertweet-base-fold5"
# TEST_CSV = '/Users/emmanueldanielchonza/Documents/Sentiment_Analysis/data/df_test_cleaned_final.csv'
# SUBMISSION_CSV = "bertweet_train_dir2/bertweet_predictions_submission_final.csv"
#
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# LABEL2ID = {'non_racist': 0, 'racist': 1}
# ID2LABEL = {0: 'non_racist', 1: 'racist'}
#
# # ---------------- STEP 1: Load best model & tokenizer ----------------
# print(f"\nLoading model from: {BEST_MODEL_DIR}")
# tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL_DIR)
# model = AutoModelForSequenceClassification.from_pretrained(BEST_MODEL_DIR).to(DEVICE)
#
# # ---------------- STEP 2: Load and prepare test data ----------------
# df_test = pd.read_csv(TEST_CSV)
# df_test['clean_tweet'] = df_test['clean_tweet'].fillna("no text available")
#
# test_ds = Dataset.from_pandas(df_test[['id', 'clean_tweet']], preserve_index=False)
#
# def tokenize(batch):
#     return tokenizer(batch['clean_tweet'], truncation=True, padding='max_length', max_length=128)
#
# test_ds = test_ds.map(tokenize, batched=True)
# test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
#
# # ---------------- STEP 3: Inference ----------------
# model.eval()
# preds = []
#
# with torch.no_grad():
#     for item in tqdm(test_ds, desc="Predicting"):
#         input_ids = item['input_ids'].unsqueeze(0).to(DEVICE)
#         attention_mask = item['attention_mask'].unsqueeze(0).to(DEVICE)
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         logits = outputs.logits
#         pred = torch.argmax(logits, dim=1).cpu().item()
#         preds.append(pred)
#
# # ---------------- STEP 4: Save submission ----------------
# df_test['label'] = preds
# df_test['label_name'] = df_test['label'].map(ID2LABEL)
#
# submission = df_test[['id', 'label']]
# submission.to_csv(SUBMISSION_CSV, index=False)
#
# print(f"\nSubmission file saved to: {SUBMISSION_CSV}")
# print("\nSample predictions:")
# print(df_test[['id', 'label', 'label_name']].head())

# ------------------------------------------ Import Libraries ------------------------------------------
import os
import json
import pandas as pd
from datasets import Dataset
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

# ----------------------------------------- CONFIG -------------------------------------------------------
TRAIN_DIR = "bertweet_train_dir2"
TEST_CSV = '/Users/emmanueldanielchonza/Documents/Sentiment_Analysis/data/df_test_cleaned_final.csv'
SUBMISSION_CSV = os.path.join(TRAIN_DIR, "bertweet_predictions_submission_final.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL2ID = {'non_racist': 0, 'racist': 1}
ID2LABEL = {0: 'non_racist', 1: 'racist'}

# -------------------------------------- STEP 1: Find best fold by eval F1 -------------------------------
fold_metrics = {}

for i in range(1, 6):
    model_dir = os.path.join(TRAIN_DIR, f"vinai-bertweet-base-fold{i}")
    metrics_path = os.path.join(model_dir, "eval_results.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
            f1 = metrics.get("eval_f1")
            if f1 is not None:
                fold_metrics[i] = f1

if not fold_metrics:
    raise ValueError("No eval_results.json with 'eval_f1' found in any fold.")

# Print scores and select best fold
print("\nFold F1 scores:")
for fold, score in fold_metrics.items():
    print(f"  vinai-bertweet-base-fold{fold}: F1 = {score:.4f}")

best_fold = max(fold_metrics, key=fold_metrics.get)
print(f"\nBest model: vinai-bertweet-base-fold{best_fold} with F1 = {fold_metrics[best_fold]:.4f}")

best_model_dir = os.path.join(TRAIN_DIR, f"vinai-bertweet-base-fold{best_fold}")

# ----------------------------------------- STEP 2: Load best model & tokenizer --------------------------------
print(f"\nLoading model from: {best_model_dir}")
tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
model = AutoModelForSequenceClassification.from_pretrained(best_model_dir).to(DEVICE)

# ------------------------------------------- STEP 3: Load and prepare test data -------------------------------
df_test = pd.read_csv(TEST_CSV)
df_test['clean_tweet'] = df_test['clean_tweet'].fillna("no text available")

test_ds = Dataset.from_pandas(df_test[['id', 'clean_tweet']], preserve_index=False)

def tokenize(batch):
    return tokenizer(batch['clean_tweet'], truncation=True, padding='max_length', max_length=128)

test_ds = test_ds.map(tokenize, batched=True)
test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# ------------------------------------------- STEP 4: Inference ------------------------------------------
model.eval()
preds = []

with torch.no_grad():
    for item in tqdm(test_ds, desc="🔍 Predicting"):
        input_ids = item['input_ids'].unsqueeze(0).to(DEVICE)
        attention_mask = item['attention_mask'].unsqueeze(0).to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).cpu().item()
        preds.append(pred)

# ----------------------------------------- STEP 5: Save submission --------------------------------------
df_test['label'] = preds
df_test['label_name'] = df_test['label'].map(ID2LABEL)

# Optional: if test set has true labels
if 'label' in df_test.columns:
    from sklearn.metrics import classification_report
    print("\nClassification Report on Test Set:")
    print(classification_report(df_test['label'], preds, target_names=[ID2LABEL[0], ID2LABEL[1]]))

submission = df_test[['id', 'label']]
submission.to_csv(SUBMISSION_CSV, index=False)

print(f"\nSubmission file saved to: {SUBMISSION_CSV}")
print("\nSample predictions:")
print(df_test[['id', 'label', 'label_name']].head())


