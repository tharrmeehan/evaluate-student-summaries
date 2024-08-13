import os

import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch import nn, Tensor, cat, flatten

from rouge_score import *
from ROUGE_Score_Model import Model

from tqdm import tqdm

_PATH = '/kaggle/input/commonlit-evaluate-student-summaries'


def get_data(path=None):
    if path is None:
        path = _get_path()

    summaries_df = pd.read_csv(f'{path}/summaries_train.csv')
    prompts_df = pd.read_csv(f'{path}/prompts_train.csv')
    return summaries_df, prompts_df


def mcrmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must be the same.")
    rmse_values = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    mcrmse = np.mean(rmse_values)

    return mcrmse


def _get_path():
    if os.name == 'nt':
        return f'.{_PATH}'
    elif os.name == 'posix':
        return _PATH


def _preprocess(summaries, prompts):
    prompt_columns = ['prompt_text', 'prompt_title', 'prompt_question']

    merged_df = summaries.merge(prompts, 'inner', 'prompt_id')

    functions = [('rouge1', lambda row: rouge_n(row.prompt_text, row.text, 1)),
                 ('rouge2', lambda row: rouge_n(row.prompt_text, row.text, 2)),
                 ('rougeL', lambda row: rouge_l(row.prompt_text, row.text)),
                 ('rougeLsum', lambda row: rouge_lsum(row.prompt_text, row.text)),
                 # ('rouge_pos',  lambda row: rouge_pos(row.prompt_text, row.text))
                 ]
    for i, func_tuple in enumerate(functions):
        func_name = func_tuple[0]
        func = func_tuple[1]

        tqdm.pandas(desc=f'{i + 1}/{len(functions)}')
        merged_df[[f'{func_name}_precision', f'{func_name}_recall', f'{func_name}_fmeasure']] = merged_df[
            ['text', 'prompt_text']].progress_apply(func, axis=1, result_type='expand')

    print("Done")

    summaries = merged_df.drop(prompt_columns, axis=1)
    return summaries, prompts


def _get_scores(row):
    scores = []
    for score in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum',
                  # 'rouge_pos'
                  ]:
        scores.append(row[f'{score}_precision'])
        scores.append(row[f'{score}_recall'])
        scores.append(row[f'{score}_fmeasure'])

    scores = torch.Tensor(scores)
    scores = flatten(scores)
    return scores


def _get_prompt(summary, prompts_df):
    return prompts_df.loc[prompts_df.prompt_id == summary.prompt_id].iloc[0]


def _get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('\x1b[0;32mGPU is available.\x1b[0m')
    else:
        device = torch.device("cpu")
        print('\x1b[0;34mGPU not available. CPU used.\x1b[0m')
    return device


class ModelLoader:
    def __init__(self,
                 training_summaries: pd.DataFrame,
                 prompts: pd.DataFrame,
                 hidden_dim: int = 64,
                 target_score: str = 'content',
                 device: torch.device = None) -> None:

        assert target_score in ['both', 'content', 'wording'], "target_score must be 'both', 'content' or 'wording'"
        self.target_score = target_score
        output_dim = 2 if target_score == 'both' else 1

        self.summaries = training_summaries
        self.prompts = prompts
        self.summaries_processed = None
        self.prompts_processed = None

        text_len_mean = training_summaries.text.apply(len).mean()
        text_len_std = training_summaries.text.apply(len).std()

        if device is None:
            self.device = _get_device()
        else:
            self.device = device

        self.model = Model(hidden_dim, output_dim, text_len_mean, text_len_std).to(self.device)
        self.model.reset_parameters()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam
        self.optimizer = self.optimizer(list(self.model.parameters()))

    def train(self, epochs: int = 10, lr: float = 0.01):
        self.optimizer.lr = lr

        if self.summaries_processed is None:
            self.preprocess()
        summaries, prompts = self.summaries_processed, self.prompts_processed

        for epoch in tqdm(range(epochs), desc='Training', leave=False):
            epoch += 1

            train_df = summaries.sample(frac=1).reset_index(drop=True)

            y_true = []
            y_pred = []

            for index, summary in tqdm(train_df.iterrows(), total=len(train_df), leave=False):
                scores = _get_scores(summary)

                predictions, target = self._predict(summary, prompts, scores)
                loss = self.criterion(predictions, target)
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()

                y_true.append([*(float(x) for x in target)])
                y_pred.append([*(float(x) for x in predictions)])

        rmse = mcrmse(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        performance = {'RMSE': rmse, 'R2': r2, 'MSE': mse}

        for i in range(len(y_true[0])):
            y_true_i = [y[i] for y in y_true]
            y_pred_i = [y[i] for y in y_pred]

            rmse = mcrmse(y_true_i, y_pred_i)
            mse = mean_squared_error(y_true_i, y_pred_i)
            r2 = r2_score(y_true_i, y_pred_i)

            i += 1

            performance |= {f'RMSE_{i}': rmse, f'R2_{i}': r2, f'MSE_{i}': mse}

        return performance

    def test(self, test_summaries: pd.DataFrame, prompts: pd.DataFrame):

        y_true = []
        y_pred = []
        for index, summary in tqdm(test_summaries.iterrows(), total=len(test_summaries), desc='Testing', leave=False):
            predictions, target = self._predict(summary, prompts)

            y_true.append([*(float(x) for x in target)])
            y_pred.append([*(float(x) for x in predictions)])

        rmse = mcrmse(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        performance = {'RMSE': rmse, 'R2': r2, 'MSE': mse}

        for i in range(len(y_true[0])):
            y_true_i = [y[i] for y in y_true]
            y_pred_i = [y[i] for y in y_pred]

            rmse = mcrmse(y_true_i, y_pred_i)
            mse = mean_squared_error(y_true_i, y_pred_i)
            r2 = r2_score(y_true_i, y_pred_i)

            i += 1

            performance |= {f'RMSE_{i}': rmse, f'R2_{i}': r2, f'MSE_{i}': mse}

        return performance

    def preprocess(self):
        self.summaries_processed, self.prompts_processed = _preprocess(self.summaries, self.prompts)

    def load_preprocessed_summaries(self, path):
        self.summaries_processed = pd.read_csv(path)
        self.prompts_processed = self.prompts

    def _predict(self, summary, prompts_df, scores=None):
        prompt = _get_prompt(summary, prompts_df)

        if self.target_score == 'content':
            target = Tensor([summary.content]).to(self.device)
        elif self.target_score == 'wording':
            target = Tensor([summary.wording]).to(self.device)
        else:
            target = Tensor([summary.content, summary.wording]).to(self.device)

        predictions = self.model(prompt.prompt_text, summary.text, scores)
        return predictions, target

    def __call__(self, summary: str, prompt: str):
        return self.model.predict(summary, prompt)


class _ModelLoaderPreprocessed(ModelLoader):
    def __init__(self,
                 training_summaries_processed: pd.DataFrame,
                 prompts_processed: pd.DataFrame,
                 hidden_dim: int = 64,
                 target_score: str = 'both',
                 device: torch.device = None) -> None:
        super().__init__(
            training_summaries_processed,
            prompts_processed,
            hidden_dim,
            target_score,
            device
        )

        self.summaries_processed = training_summaries_processed
        self.prompts_processed = prompts_processed


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split

    summaries_df, prompts_df = get_data('../kaggle/input/commonlit-evaluate-student-summaries')

    summaries_df, prompts_df = _preprocess(summaries_df, prompts_df)
    summaries_df.to_csv("summaries_df_processed.csv")

    train_df, test_df = train_test_split(summaries_df,
                                         test_size=0.2,
                                         stratify=summaries_df["prompt_id"],
                                         random_state=42)

    loader = ModelLoader(train_df, prompts_df, target_score='content')
    train_scores = loader.train(epochs=10)
    test_scores = loader.test(test_df, prompts_df)

    model = loader.model  # Final, trained Model

    print(f"Training Scores ({loader.target_score})")
    for score, value in train_scores.items():
        print(f'{score}:\t{value}')
    print()
    print(f"Validation Scores ({loader.target_score})")
    for score, value in test_scores.items():
        print(f'{score}:\t{value}')
    print()

    torch.save(model.state_dict(), f"models/ROUGE/ROUGE-Example.pt")
