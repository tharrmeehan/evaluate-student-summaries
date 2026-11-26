# Evaluate Student Summaries

A three-model ensemble for the [CommonLit — Evaluate Student Summaries](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries) Kaggle competition. The system predicts two continuous scores — content coverage and writing quality (wording) — for student-written summaries of grades 3–12 source texts.

Historical competition result: **975th out of 2,065 teams** (top 47%), as recorded for the competition submission. This documents the reported result; it does not establish that the full pipeline is currently reproducible.

Team: Tharrmeehan Krishnathasan, Jonathan Carona, Josef Rittiner — AI Challenge, Hochschule Luzern (12 ECTS). Competition: [CommonLit — Evaluate Student Summaries](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries).

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

- **Ensemble by weighted averaging** — the documented design blends predictions from the three models with fixed weights. The repository does not establish that this blend remains reproducible in the current environment.

---

## Historical Result & Project Artifacts

- Historical Kaggle result: **975 / 2,065** on the public leaderboard.
- The repository includes the [course report](EvalStudentSummaries_Report.pdf) and [final team report](documentation/finale/Team4-Evaluate-Student-Sumaries-Report.pdf). Submission and workflow examples include [the Kaggle submission notebook](Tharrmeehan_Playground/Kaggle_Submission_Notebook.ipynb), [the final notebook](Tharrmeehan_Playground/Final_Notebook.ipynb), and the model notebooks below.
- The leaderboard result is historical, not a claim that the notebooks can currently reproduce the submission or result.

Competition data and redistribution: this repository contains competition-related datasets and artifacts. Do not assume Kaggle permits their redistribution; obtain data through the [official competition page](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/data) and check its current rules and terms before sharing or reusing any competition data.

---

## Local Replication

Prerequisites vary by notebook and include Python plus the libraries used by each model. The notebooks reference data in different repository subdirectories; inspect each notebook's setup and data-loading cells and provide the expected inputs before running it.

Competition data may be subject to access and redistribution restrictions. Download it from the [official Kaggle data page](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/data) only if eligible, and follow the current terms. Do not treat checked-in data copies as permission to redistribute.

These are research and competition artifacts, not a verified end-to-end reproduction. No full rerun is claimed.

```bash
jupyter notebook LightGBM-Model/run_lightgbm.ipynb
jupyter notebook ROUGE-Based-Model/run_rouge_new.ipynb
jupyter notebook Transformer_Deberta-V3-Large/run_transformer_new.ipynb
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
