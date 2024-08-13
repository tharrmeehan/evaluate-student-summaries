import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# preprocessing
from nltk.corpus import stopwords
from transformers import AutoTokenizer
from textstat import textstat
from rouge import Rouge

# model
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error

# hyperparameter tuning
import optuna


class AIO:
    def __init__(self, train, test):
        self.train = train
        self.test = test
        self.stopwords = stopwords.words('english')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.rouge = Rouge()

    def word_overlap_count(self, row):
        """intersection(prompt_text, text)"""

        def check_is_stop_word(word):
            return word in self.stopwords

        prompt_words = row["prompt_tokens"]
        summary_words = row["summary_tokens"]

        if self.stopwords:
            prompt_words = list(filter(check_is_stop_word, prompt_words))
            summary_words = list(filter(check_is_stop_word, summary_words))
        return len(set(prompt_words).intersection(set(summary_words)))

    def ngrams(self, token, n):
        ngrams = zip(*[token[i:] for i in range(n)])
        return [" ".join(ngram) for ngram in ngrams]

    def ngram_co_occurrence(self, row, n: int) -> int:
        # Tokenize the original text and summary into words
        original_tokens = row["prompt_tokens"]
        summary_tokens = row["summary_tokens"]

        # Generate n-grams for the original text and summary
        original_ngrams = set(self.ngrams(original_tokens, n))
        summary_ngrams = set(self.ngrams(summary_tokens, n))

        # Calculate the number of common n-grams
        common_ngrams = original_ngrams.intersection(summary_ngrams)

        return len(common_ngrams)

    def get_rouge_score(self, row):
        scores = self.rouge.get_scores(row["text"], row["prompt_text"])
        return scores[0]["rouge-1"]["f"]

    def preprocess(self, dataframe) -> pd.DataFrame:
        dataframe["summary_tokens"] = dataframe["text"].apply(
            lambda x: self.tokenizer.tokenize(x, truncation=True, padding=True, max_length=1024)
        )

        dataframe["prompt_tokens"] = dataframe["prompt_text"].apply(
            lambda x: self.tokenizer.tokenize(x, truncation=True, padding=True, max_length=1024)
        )
        dataframe["flesch_reading_ease"] = dataframe["text"].apply(
            textstat.flesch_reading_ease
        )

        dataframe["difficult_words"] = dataframe["text"].apply(textstat.difficult_words)
        dataframe["bigrams_overlap_count"] = dataframe.apply(
            self.ngram_co_occurrence, args=(2,), axis=1
        )
        dataframe["automated_readability_index"] = dataframe["text"].apply(
            textstat.automated_readability_index
        )
        dataframe["coleman_liau_index"] = dataframe["text"].apply(
            textstat.coleman_liau_index
        )
        dataframe["linsear_write_formula"] = dataframe["text"].apply(
            textstat.linsear_write_formula
        )
        dataframe["gunning_fog"] = dataframe["text"].apply(textstat.gunning_fog)
        dataframe["smog_index"] = dataframe["text"].apply(textstat.smog_index)
        dataframe["word_overlap_count"] = dataframe.apply(self.word_overlap_count, axis=1)
        dataframe["rouge"] = dataframe.apply(self.get_rouge_score, axis=1)
        return dataframe

    def compute_mcrmse(self, eval_pred):
        """
        Calculates mean columnwise root mean squared error
        https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/overview/evaluation
        """
        preds, labels = eval_pred

        col_rmse = np.sqrt(np.mean((preds - labels) ** 2, axis=0))
        mcrmse = np.mean(col_rmse)

        return {
            "content_rmse": col_rmse[0],
            "wording_rmse": col_rmse[1],
            "mcrmse": mcrmse,
        }

    def run(self):
        self.train = self.preprocess(self.train)
        self.test = self.preprocess(self.test)
        self.train = self.train.drop(
            [
                "student_id",
                "text",
                "prompt_question",
                "prompt_title",
                "prompt_text",
                "summary_tokens",
                "prompt_tokens",
            ],
            axis=1,
        )
        self.test = self.test.drop(
            [
                "text",
                "prompt_question",
                "prompt_title",
                "prompt_text",
                "summary_tokens",
                "prompt_tokens",
            ],
            axis=1,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            self.train.drop(["wording", "content", "prompt_id"], axis=1),
            self.train[["wording", "content"]],
            test_size=0.2,
            random_state=42,
            stratify=self.train["prompt_id"],
        )

        def objective(trial):
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-8, 1.0, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 1, 1000),
                "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                "max_depth": trial.suggest_int("max_depth", 1, 32),
                "min_child_samples": trial.suggest_int("min_child_samples", 1, 100),
                "subsample": trial.suggest_float("subsample", 0.1, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                "random_state": 42,
                "n_jobs": -1,
            }

            lgb_model = lgb.LGBMRegressor(**params)
            model = MultiOutputRegressor(lgb_model)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = mean_squared_error(y_val, y_pred, squared=False)
            return score

        study_lgb = optuna.create_study(direction="minimize")
        study_lgb.optimize(objective, n_trials=100)
        model = MultiOutputRegressor(lgb.LGBMRegressor(**study_lgb.best_params))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        eval_pred = (y_pred, y_val)
        self.compute_mcrmse(eval_pred)
        y_pred_final = model.predict(self.test.drop(["student_id", "prompt_id"], axis=1))
        y_pred_final_df = pd.DataFrame(y_pred_final, columns=["wording", "content"])
        return y_pred_final_df

