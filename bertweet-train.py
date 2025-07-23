import os
import random
import yaml
import json
import numpy as np
import pandas as pd
import torch

from datasets import Dataset, DatasetDict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoConfig,
    TrainingArguments, Trainer
)
import evaluate
from torch.nn import CrossEntropyLoss

# ---------------------------------------- Paths & Parameters ------------------------------------
TRAIN_CSV = '/Users/emmanueldanielchonza/Documents/Sentiment_Analysis/data/train_cleaned_tweets_final.csv'
TEST_CSV = '/Users/emmanueldanielchonza/Documents/Sentiment_Analysis/data/df_test_cleaned_final.csv'
OUTPUT_DIR = 'bertweet_train_dir2'
MODEL_CKPT = "vinai/bertweet-base"

LABEL2ID = {'non_racist': 0, 'racist': 1}
ID2LABEL = {0: 'non_racist', 1: 'racist'}

BATCH_SIZE = 32
EPOCHS = 3
LR = 2e-5
SEED = 42
NUM_FOLDS = 5

# ---------------------------------------- Environment Setup -----------------------------------------
def load_api_keys():
    with open("huggingface_credentials.yml", "r") as f:
        keys = yaml.safe_load(f)
    hf_key = keys.get("HUGGINGFACE_API_KEY", "")
    if not hf_key:
        raise ValueError("HUGGINGFACE_API_KEY missing in huggingface_credentials.yml")
    os.environ["HF_TOKEN"] = hf_key
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

load_api_keys()

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ------------------------------------------------- Load Data -----------------------------------------
train_data = pd.read_csv(TRAIN_CSV, index_col=0).dropna()
df_test = pd.read_csv(TEST_CSV, index_col=0)
df_test['clean_tweet'] = df_test['clean_tweet'].fillna("no text available")

train_data['label_name'] = train_data['label_name'].replace({
    'non_racist_sexist': 'non_racist',
    'racist_sexiest': 'racist'
})

train_data['label'] = train_data['label_name'].map(LABEL2ID)

# ------------------------------------------------- Tokenizer -------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT, use_fast=True)

def tokenize(batch):
    texts = [str(t) if t is not None else "" for t in batch['clean_tweet']]
    return tokenizer(texts, truncation=True, padding='max_length', max_length=128)

# ----------------------------------------- Metrics -------------------------------------------------------
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    macro_f1 = f1_score(labels, preds, average='macro')
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "precision": precision.compute(predictions=preds, references=labels)["precision"],
        "recall": recall.compute(predictions=preds, references=labels)["recall"],
        "macro_f1": macro_f1,
        "f1": f1_metric.compute(predictions=preds, references=labels)["f1"]
    }

# ---------------------------------------- Cross-validation ----------------------------------------
kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(kf.split(train_data, train_data['label']), 1):
    print(f"\nFold {fold}/{NUM_FOLDS}")
    train_df = train_data.iloc[train_idx]
    val_df = train_data.iloc[val_idx]

    # ----------------------------------------- Oversample minority ----------------------------------------
    majority = train_df[train_df['label_name'] == 'non_racist']
    minority = train_df[train_df['label_name'] == 'racist']
    minority_oversampled = minority.sample(n=len(majority), replace=True, random_state=SEED)
    train_oversampled = pd.concat([majority, minority_oversampled]).sample(frac=1, random_state=SEED)

    train_oversampled['label'] = train_oversampled['label_name'].map(LABEL2ID)
    val_df['label'] = val_df['label_name'].map(LABEL2ID)

    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_oversampled[['clean_tweet', 'label']], preserve_index=False),
        'validation': Dataset.from_pandas(val_df[['clean_tweet', 'label']], preserve_index=False),
    })

    dataset = dataset.map(tokenize, batched=True)
    dataset['train'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    dataset['validation'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Model and weights
    config = AutoConfig.from_pretrained(
        MODEL_CKPT, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT, config=config, device_map="auto")

    # --------------------------------------- Class weights ----------------------------------------------
    class_counts = train_oversampled['label'].value_counts()
    # weights = torch.tensor([1.0, class_counts[0] / class_counts[1]]).to(model.device)
    weights = torch.tensor([1.0, class_counts[0] / class_counts[1]], dtype=torch.float32).to(model.device)


    def custom_compute_loss(model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    # ------------------------------------- Training arguments --------------------------------------------
    fold_output_dir = os.path.join(OUTPUT_DIR, f"fold_{fold}")
    training_args = TrainingArguments(
        output_dir=fold_output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=0.01,
        seed=SEED,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_dir=os.path.join(OUTPUT_DIR, f"logs/fold_{fold}"),
        report_to='none'
    )

    # ----------------------------------------- Prepare the trainer and Train the Model -------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # ----------------------------------- Save model & metrics --------------------------------------
    trainer.save_model(os.path.join(OUTPUT_DIR, f"vinai-bertweet-base-fold{fold}"))
    metrics = trainer.evaluate()
    with open(os.path.join(OUTPUT_DIR, f"vinai-bertweet-base-fold{fold}", "eval_results.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # ----------------------------------- Confusion matrix ------------------------------------------
    eval_output = trainer.predict(dataset['validation'])
    y_true = eval_output.label_ids
    y_pred = np.argmax(eval_output.predictions, axis=1)

    print(classification_report(y_true, y_pred, target_names=[ID2LABEL[0], ID2LABEL[1]]))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[ID2LABEL[0], ID2LABEL[1]],
                yticklabels=[ID2LABEL[0], ID2LABEL[1]])
    plt.title(f'Confusion Matrix Fold {fold}', fontsize=16)
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_fold_{fold}.png")
    plt.close()

print("\nAll folds completed.")

# ---------------------------- Note: Select Best Fold Separately ----------------------------
print("\nTo make predictions on test set, load the best fold model manually after reviewing `eval_results.json` in each fold directory.")