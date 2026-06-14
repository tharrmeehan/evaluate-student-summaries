# Evaluate Student Summaries

A three-model ensemble for the [CommonLit — Evaluate Student Summaries](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries) Kaggle competition. The system predicts two continuous scores — content coverage and writing quality (wording) — for student-written summaries of grades 3–12 source texts.

Competition result: **975th out of 2,065 teams** (top 47%).

Team: Tharrmeehan Krishnathasan, Jonathan Carona, Josef Rittiner — AI Challenge, Hochschule Luzern (12 ECTS).

---

## Architecture

```
data/
  summaries_train.csv    (student_id, prompt_id, text, content, wording)
  prompts_train.csv      (prompt_id, prompt_title, prompt_question, prompt_text)
        │
        ├─► ROUGE-Based Neural Network
        │     ROUGE-1/2/L between summary and source text as input features
        │     Feed-forward network predicts content + wording
        │
        ├─► LightGBM Model
        │     Readability indices (Flesch, SMOG, Dale-Chall)
        │     Complexity and grade-level features
        │     LightGBM gradient boosting → content + wording scores
        │
        └─► DeBERTa v3 Large (Transformer)
              Fine-tuned on (prompt_text + summary) → regress content + wording
              Handles semantic alignment between summary and source
              │
              ▼
        Ensemble
              Weighted averaging of three model outputs → final predictions
```

Each model preprocesses inputs independently. The ensemble weights were tuned to balance the strengths of lexical overlap (ROUGE), feature-based readability (LightGBM), and deep contextual understanding (DeBERTa).

---

## Core Technical Stack

Python, PyTorch, Hugging Face Transformers (DeBERTa v3 Large), LightGBM, scikit-learn, ROUGE-score, pandas, Kaggle API

---

## Key Methodologies

- **ROUGE-based features as a neural input** — ROUGE-1, ROUGE-2, and ROUGE-L F1 scores quantify lexical overlap between a student summary and its source text. Using them as features to a neural network rather than as a direct score allows the model to learn a non-linear mapping from overlap to human-assigned content quality.

- **LightGBM on readability signals** — features such as Flesch Reading Ease, SMOG index, and Dale-Chall score capture linguistic complexity and grade-level alignment independently of semantic similarity. Gradient boosting on these structured features provides a strong, fast baseline that complements the transformer's dense representations.

- **DeBERTa v3 Large for semantic alignment** — fine-tuned on the concatenation of the prompt text and student summary, DeBERTa captures whether the student understood and addressed the prompt's specific question. This is information that neither ROUGE metrics nor readability indices encode.

- **Ensemble by weighted averaging** — rather than stacking with a meta-learner (which risks overfitting on a small validation set), the three model predictions are blended with fixed weights. This was found to be more stable given the competition's test-set distribution.

---

## Production Metrics & Validation

- Evaluated on Kaggle's held-out test set using MCRMSE (Mean Columnwise Root Mean Squared Error) across the content and wording targets.
- Baseline (DummyRegressor, median strategy) established early in the course for reference.
- Final ensemble placed **975 / 2,065** in the public leaderboard.
- Full methodology and ablation discussion available in [`EvalStudentSummaries_Report.pdf`](EvalStudentSummaries_Report.pdf).

---

## Local Replication

Prerequisites: Python 3.8+, CUDA recommended for DeBERTa fine-tuning.

```bash
git clone https://github.com/tharrmeehan/evaluate-student-summaries.git
cd evaluate-student-summaries

pip install pandas scikit-learn lightgbm torch transformers rouge-score jupyter

# Download competition data from Kaggle and place in data/
# https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/data

# Run individual models
jupyter notebook LightGBM-Model/run_lightgbm.ipynb
jupyter notebook ROUGE-Based-Model/run_rouge_new.ipynb
jupyter notebook Transformer_Deberta-V3-Large/run_transformer_new.ipynb

# Run the ensemble
jupyter notebook Ensemble/run_transformer_new.ipynb
```

---

## Project Structure

```
├── LightGBM-Model/
│   ├── AIOLightGBM.py
│   └── run_lightgbm.ipynb
├── ROUGE-Based-Model/
│   └── run_rouge_new.ipynb
├── Transformer_Deberta-V3-Large/
│   └── run_transformer_new.ipynb
├── Ensemble/
│   └── run_transformer_new.ipynb
├── data_preprocessing/
├── baseline.ipynb                    # DummyRegressor baseline
├── EvalStudentSummaries_Report.pdf   # Full course report
└── data/
```
