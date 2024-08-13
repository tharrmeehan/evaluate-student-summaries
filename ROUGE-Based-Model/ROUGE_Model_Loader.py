import os

from sklearn.metrics import mean_squared_error, r2_score
from torch import nn, flatten, Tensor
import torch
import pandas as pd
from ROUGE_Based_Model import ROUGEBasedModel
from tqdm import tqdm
from rouge_score import *

path = './models/rouge_based_models'

functions = [
    ('rouge1', lambda row: rouge_n(row.prompt_text, row.text, 1)),
    ('rouge2', lambda row: rouge_n(row.prompt_text, row.text, 2)),
    ('rougeL', lambda row: rouge_l(row.prompt_text, row.text)),
    ('rougeLsum', lambda row: rouge_lsum(row.prompt_text, row.text))
]


def _preprocess(merged_df):
    path = 'data/merged_rouge_preprocessed.csv'

    if os.path.exists(path):
        merged_df = pd.read_csv(path)
    else:
        for i, func_tuple in enumerate(functions):
            func_name = func_tuple[0]
            func = func_tuple[1]

            tqdm.pandas(desc=f'{i + 1}/{len(functions)}')
            merged_df[[f'{func_name}_precision', f'{func_name}_recall', f'{func_name}_fmeasure']] = merged_df[
                ['text', 'prompt_text']].progress_apply(func, axis=1, result_type='expand')

        merged_df.to_csv(path)
        print("Done")

    return merged_df


def _get_scores(row):
    scores = []
    for score, _ in functions:
        scores.append(row[f'{score}_precision'])
        scores.append(row[f'{score}_recall'])
        scores.append(row[f'{score}_fmeasure'])

    scores = torch.Tensor(scores)
    scores = flatten(scores)
    return scores


class ROUGEModelLoader():
    def __init__(self,
                 training_summaries: pd.DataFrame,
                 hidden_dim: int = 64,
                 target_score: str = 'content',
                 device: torch.device = 'cpu') -> None:

        assert target_score in ['both', 'content', 'wording'], "target_score must be 'both', 'content' or 'wording'"
        self.target_score = target_score
        output_dim = 2 if target_score == 'both' else 1

        self.summaries = training_summaries
        self.summaries_processed = None

        text_len_mean = training_summaries.text.apply(len).mean()
        text_len_std = training_summaries.text.apply(len).std()

        self.model = ROUGEBasedModel(hidden_dim, output_dim, text_len_mean, text_len_std, device)
        self.model.reset_parameters()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam
        self.optimizer = self.optimizer(list(self.model.parameters()))

        self.device = device

    def preprocess(self):
        self.summaries_processed = _preprocess(self.summaries)

    def train(self, epochs: int = 10, lr: float = 0.01, verbose=True):
        self.optimizer.lr = lr

        if self.summaries_processed is None:
            self.preprocess()
        summaries = self.summaries_processed

        for _ in tqdm(range(epochs), desc='Training', leave=False, disable=not verbose):

            train_df = summaries.sample(frac=1).reset_index(drop=True)

            y_true = []
            y_pred = []

            for index, summary in tqdm(train_df.iterrows(), total=len(train_df), leave=False, disable=not verbose):
                scores = _get_scores(summary)

                predictions, target = self._predict(summary, scores)
                loss = self.criterion(predictions, target)
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()

                y_true.append([*(float(x) for x in target)])
                y_pred.append([*(float(x) for x in predictions)])

        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        performance = {'RMSE': rmse, 'R2': r2, 'MSE': mse}

        if self.target_score == 'both':
            score_map = {
                1: 'content',
                2: 'wording'
            }
        else:
            score_map = {1: self.target_score}

        for i in range(len(y_true[0])):
            y_true_i = [y[i] for y in y_true]
            y_pred_i = [y[i] for y in y_pred]

            rmse = mean_squared_error(y_true_i, y_pred_i, squared=False)
            mse = mean_squared_error(y_true_i, y_pred_i)
            r2 = r2_score(y_true_i, y_pred_i)

            i = score_map[i + 1]

            performance |= {f'RMSE_{i}': rmse, f'R2_{i}': r2, f'MSE_{i}': mse}

        return performance

    def _predict(self, summary, scores=None):

        if self.target_score == 'content':
            target = Tensor([summary.content]).to(self.device)
        elif self.target_score == 'wording':
            target = Tensor([summary.wording]).to(self.device)
        else:
            target = Tensor([summary.content, summary.wording]).to(self.device)

        predictions = self.model(summary.prompt_text, summary.text, scores)
        return predictions, target


if __name__ == '__main__':
    merged_df = pd.read_csv('data/merged.csv')

    _preprocess(merged_df)

    target_score = 'both'
    hidden_dim = 64
    epochs = 10
    lr = 0.01

    model_loader = ROUGEModelLoader(merged_df, hidden_dim, target_score)

    performance = model_loader.train(epochs, lr)
    print()
    print(*performance.items(), sep='\n')

    model = model_loader.model

    torch.save(model.state_dict(), f'{path}/ROUGE_Based_Model_{target_score}_{hidden_dim}_{epochs}_{lr}.pt')
