# Racist-Detection-App
This is a production-ready, end-to-end system developed to detect and classify racist tweets using advanced Natural Language Processing (NLP) techniques. Built on top of BERTweet (vinai/bertweet-base) and fine-tuned with a robust, k-fold cross-validation training pipeline, powered by streamlit UI!

# 📝 Racist/Sexist Tweet Detection App

## 1. Overview
This project implements a robust racist/sexist tweet detection system using the BERTweet (vinai/bertweet-base) model, fine-tuned on domain-specific labeled data.
It includes a training pipeline, preprocessing, inference module, and a Streamlit web application for real-time predictions.

## 2. Features
- Fine-tuning with 5-fold cross-validation and best-model selection
- Advanced tweet preprocessing pipeline (including spell correction)
- Offline-friendly reusable inference module
- User-friendly Streamlit web application

## 3. Model: vinai/bertweet-base
- A transformer-based language model pre-trained on 850M English tweets.
- Fine-tuned the model on a binary classification task:
  - non_racist
  - racist

## 4. Workflow
### 4.1 Training (bertweet_train.py)
- Loads cleaned and oversampled dataset.
- Performs 5-fold stratified cross-validation.
- Applies oversampling to balance the minority class.
- Preprocesses tweets using text_preprocessor.py.
- Fine-tunes vinai/bertweet-base using HuggingFace Trainer.
- Saves metrics and confusion matrix per fold.
- Selects best fold based on F1-score.

### 4.1 Best Model Consolidation
- Automatically finds the fold with the highest F1-score by reading eval_results.json in each fold directory.

### 4.3 Prediction (predict.py)
- Loads the best fold model automatically.
- Runs predictions on the test set or new data.
- Prints classification report and saves predictions to CSV.

### 4.4 Spell Correction (symspellpy_local.py)
- Prepares and saves the SymSpell dictionary locally for use during inference.

### 4.5 Inference (inference.py)
- Loads the best fold model.
- Preprocesses text (including spell correction, emojis, hashtags).
- Performs inference on single tweet input.
- Returns label_id, label_name, and confidence score.

### 4.6 Streamlit App (app.py)
- Real-time web interface.
- Server-side preprocessing and model inference.
- Displays prediction with confidence.

## 5. Evaluation
- Validation metrics per fold:
- Accuracy, Precision, Recall, F1-score
- Classification report
- Confusion matrix

## 6. Preprocessor: text_preprocessor.py
- Preprocessing steps:
  - Converts to UTF-8 
  - Removes irrelevant characters and stopwords 
  - Corrects spelling with SymSpell 
  - Extracts meaningful elements: emojis, hashtags, mentions 
  - Ensures consistency with training preprocessing

## 7. Web App: 🌐 Streamlit
- Run the web app for interactive predictions:
- bash
``
streamlit run app.py
``
- Features:
  - Single tweet input 
  - Clean and preprocess 
  - Predict and display label_name & confidence

## 8. Project Structure

racist-detection/
├── app.py                       # Streamlit UI
├── bertweet_train.py            # Training with 5-fold CV
├── predict.py                   # Batch predictions & evaluation
├── inference.py                 # Single-tweet inference logic
├── symspellpy_local.py          # Save SymSpell dictionary locally
├── text_preprocessor.py         # Text preprocessing logic
├── bertweet_train_dir2/         # Fine-tuned models & metrics
│   └── vinai-bertweet-base-foldX/
├── requirements.txt
├── README.md

## 9. Installation
- Install dependencies:
  - bash
  ``
pip install -r requirements.txt
  ``

## 10. Example Inference Output
- Sample prediction: (1, 'racist', 0.84)

## 11. Future Work
- Batch predictions with CSV upload
- More granular sentiment categories
- Explainable AI integration (e.g., SHAP)
- Deploy on HuggingFace Spaces or Docker

## 12. Acknowledgments
- vinai/bertweet-base by VinAI Research
- 🤗 HuggingFace Transformers & Datasets
- Streamlit
