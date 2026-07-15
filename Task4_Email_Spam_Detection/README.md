# Task 4 — Email Spam Detection with Machine Learning

**Internship:** OIBSIP
**Author:** Pragyan Kumar Beheruk
**File naming:** `Pragyan_Kumar_Beheruk_Task4`

## Objective
Build a Natural Language Processing (NLP) binary classifier that distinguishes spam messages from legitimate (ham) messages.

## Dataset
- **Source:** SMS Spam Collection Dataset — self-sourced from Kaggle / UCI Machine Learning Repository.
- **Size:** 5,572 labeled SMS messages (`ham` or `spam`).
- **File used:** `spam.csv`

## Tech Stack
- Python
- pandas, numpy
- scikit-learn (TF-IDF Vectorizer, Multinomial Naive Bayes, Logistic Regression)
- NLTK (stopword removal, stemming)
- Matplotlib, Seaborn (visualizations)
- WordCloud (bonus visualization)
- Jupyter Notebook

## Project Workflow
1. **Data loading & class distribution** — loaded the dataset, checked spam vs. ham counts and percentages.
2. **Text preprocessing pipeline** — lowercase conversion, URL/punctuation/digit removal, stopword removal, and stemming (Porter Stemmer).
3. **Feature extraction** — TF-IDF vectorization, with a markdown explanation of what TF-IDF measures.
4. **Train/test split** — 80/20 stratified split.
5. **Model training** — Multinomial Naive Bayes (industry standard for text) and Logistic Regression (alternative classifier).
6. **Evaluation** — accuracy, precision, recall, F1-score, and confusion matrices for both models.
7. **Discussion** — markdown cell explaining why recall is particularly important for spam detection.
8. **Bonus** — WordCloud visualizations of the most frequent words in spam vs. ham messages.

## Repository Structure
```
Task4_Email_Spam_Detection/
├── spam.csv
├── Pragyan_Kumar_Beheruk_Task4.ipynb
├── README_Task4.md
├── requirements_Task4.txt
└── outputs/
    ├── 01_Class_Distribution.png
    ├── 02_Confusion_Matrices.png
    ├── 03_Model_Comparison.png
    └── 04_WordClouds.png
```

## How to Run
1. Install dependencies:
   ```
   pip install -r requirements_Task4.txt
   ```
2. Launch Jupyter:
   ```
   jupyter notebook Pragyan_Kumar_Beheruk_Task4.ipynb
   ```
3. Run all cells in order (Cell → Run All).

## Results Summary
Both classifiers were evaluated on the held-out test set using accuracy, precision, recall, and F1-score. Recall was emphasized as the most critical metric for this task, since a missed spam message (false negative) carries greater real-world risk than an occasional false positive.

## Submission Checklist (per internship guidelines)
- [x] GitHub repository named `OIBSIP`
- [x] File naming format: `Pragyan_Kumar_Beheruk_Task4`
- [x] README with full documentation
- [x] Clean, commented Jupyter Notebook
- [x] Submitted via official submission form with repo link and demo (if applicable)
