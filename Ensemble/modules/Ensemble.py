import torch
from torch import nn
from tqdm import tqdm
import pandas as pd


def log(*args, sep=' ', end='\n', file_name='log.txt', flush=False, verbose=True):
    with open(file_name, 'a') as file:
        print(*args, sep=sep, end=end, file=file, flush=flush)
    print('logged\n' * verbose, end='')


def log_dict(dict, file_name='log.txt', verbose=True):
    with open(file_name, 'a') as file:
        for key, value in dict.items():
            print(key, "\t", value, file=file)
    print('logged\n' * verbose, end='')


class Ensemble(nn.Module):
    def __init__(self, transformer_model, transformer_tokenizer, rouge_model, lgbm_model, max_length=1024,
                 device='cpu'):
        super().__init__()

        self.max_length = max_length
        self.device = device

        self.transfomer_tokenizer = transformer_tokenizer
        self.transformer = transformer_model.to(device)
        self.rouge_model = rouge_model.to(device)
        self.lgbm_model = lgbm_model  # .to(device)

        self.layers = nn.ModuleList([
            nn.Linear(3 * 2, 2)
        ]).to(self.device)

    def forward(self, summary_text, prompt_title, prompt_question, prompt_text, transformer_prediction=None,
                rouge_prediction=None, lgbm_prediction=None):
        if transformer_prediction is None:
            transformer_output = self.forward_transformer(summary_text, prompt_question, prompt_text)
        else:
            transformer_output = transformer_prediction.to(self.device)

        if rouge_prediction is None:
            rouge_output = self.forward_rouge(summary_text, prompt_text)
        else:
            rouge_output = rouge_prediction.to(self.device)

        if lgbm_prediction is None:
            lgbm_output = self.forward_lgbm(summary_text, prompt_text)
        else:
            lgbm_output = lgbm_prediction.to(self.device)

        output = torch.stack([
            transformer_output.to(self.device),
            rouge_output.to(self.device),
            lgbm_output.to(self.device)
        ])

        # print(output)
        output = output.T.flatten()
        # print(output)

        return self._network_forward(output)

    def _network_forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer(output)
        return output

    def predict(self, data: pd.DataFrame, ensemble=True, verbose=True):
        columns = [
            "student_id",
            "prompt_id",
            "prompt_question",
            "prompt_title"
        ]
        for column in columns:
            assert column in data.columns, f'{column} must be present in data.'

        p_bar = tqdm(disable=not verbose, total=3 + ensemble)
        p_bar.update(0)
        p_bar.set_description_str(desc="LGBM", refresh=True)

        data.reset_index(drop=True, inplace=True)
        lgbm_data = data.drop(columns, axis=1)
        if 'content' in lgbm_data.columns:
            lgbm_data = lgbm_data.drop(['content', 'wording'], axis=1)

        lgbm_predictions = self._lgbm_predict(lgbm_data)
        lgbm_predictions = pd.DataFrame(lgbm_predictions, columns=["lgbm_wording", "lgbm_content"])
        predictions = pd.concat([data, lgbm_predictions],
                                axis=1)

        p_bar.update(1)
        p_bar.set_description_str(desc="Transformer", refresh=True)

        tqdm.pandas(desc="Transformer", leave=False, disable=not verbose)
        predictions[['transformer_content', 'transformer_wording']] = predictions.progress_apply(
            self._apply_transformer, axis=1, result_type='expand'
        )
        p_bar.update(1)
        p_bar.set_description_str(desc="ROUGE", refresh=True)

        tqdm.pandas(desc="ROUGE", leave=False, disable=not verbose)
        predictions[['rouge_content', 'rouge_wording']] = predictions.progress_apply(
            self._apply_rouge, axis=1, result_type='expand'
        )
        p_bar.update(1)

        if ensemble:
            p_bar.set_description_str(desc="Ensemble", refresh=True)
            tqdm.pandas(desc="Ensemble", leave=False, disable=not verbose)
            predictions[['ensemble_content', 'ensemble_wording']] = predictions.progress_apply(
                self._apply_ensemble, axis=1, result_type='expand'
            )
            p_bar.update(1)
        p_bar.set_description_str(desc="", refresh=True)

        return predictions

    def _apply_transformer(self, row):
        return self.forward_transformer(row.text, row.prompt_question, row.prompt_text).tolist()

    def _apply_rouge(self, row):
        return self.forward_rouge(row.text, row.prompt_text).tolist()

    def _apply_ensemble(self, row):
        input_tensor = torch.Tensor([
            row.transformer_content,
            row.rouge_content,
            row.lgbm_content,
            row.transformer_wording,
            row.rouge_wording,
            row.lgbm_wording
        ]).to(self.device)
        return self._network_forward(input_tensor).tolist()

    def forward_transformer(self, summary_text, prompt_question, prompt_text):
        output = self.transformer(**self.tokenize_encode(summary_text, prompt_question, prompt_text).to(self.device))
        return output.logits.reshape(-1)

    def tokenize_encode(self, text, prompt_question, prompt_text):
        sep_token = self.transfomer_tokenizer.sep_token
        prompt = f'Evaluate the content and wording score of this summary: {sep_token} {text} {sep_token} The summary must answer the following prompt: {prompt_question} {sep_token} The prompt is related towards the following original text: {prompt_text}'

        encoded = self.transfomer_tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors='pt'
        )

        return encoded

    def forward_rouge(self, summary_text, prompt_text):
        return self.rouge_model(prompt_text, summary_text)

    def forward_lgbm(self, summary_text, prompt_text):
        y_pred_final = self._lgbm_predict(pd.DataFrame({
            'text': [summary_text],
            'prompt_text': [prompt_text]
        }))[0]
        return torch.Tensor([y_pred_final[1], y_pred_final[0]])

    def _lgbm_predict(self, data):
        data = self.lgbm_model.preprocess(data)
        data = data.drop(
            [
                "text",
                "prompt_text",
                "summary_tokens",
                "prompt_tokens",
            ],
            axis=1,
        )
        return self.lgbm_model.model.predict(data)


class EnsembleNN(Ensemble):
    def __init__(self, transformer_model, transformer_tokenizer, rouge_model, lgbm_model, max_length=1024,
                 hidden_layers=1, hidden_dim=64, device='cpu'):
        super().__init__(transformer_model, transformer_tokenizer, rouge_model, lgbm_model, max_length, device)

        self.layers = nn.ModuleList(
            [nn.Linear(3 * 2, hidden_dim)] +
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers)] +
            [nn.Linear(hidden_dim, 2)]
        ).to(self.device)


if __name__ == '__main__':
    import pandas as pd
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from ROUGE_Model_Loader import ROUGEModelLoader
    from AIOLightGBM import AIO
    import os
    import joblib
    from datetime import datetime
    from sklearn.model_selection import train_test_split

    log("\n", "=" * 20, end="")
    log(datetime.now().strftime(" Started: %d/%m/%Y %H:%M:%S "), end="")
    log("=" * 20, "\n")

    # Data
    print("Loading Data...", end="\r")
    merged_train = pd.read_csv('../data/merged.csv')
    merged_train, merged_val = train_test_split(merged_train,
                                                test_size=0.2,
                                                stratify=merged_train["prompt_id"],
                                                random_state=42)
    # merged_train = merged_train.head(50)
    # merged_val = merged_val.head(50)
    merged_test = pd.read_csv('../data/merged_test.csv')
    print("Loading Data - ok")

    # Transformer
    print("Loading Transformer...", end="\r")
    # Replace the path for the transformer and tokenizer you want to run
    TRANSFORMER_PATH = '../models/deberta-v3-base/deberta-v3-base/checkpoint-4012'
    TOKENIZER_PATH = 'microsoft/deberta-v3-base'
    # Adjust max length to fitted model
    MAX_LENGTH = 1024

    transformer = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_PATH, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    print("Loading Transformer - ok")

    # Rouge
    print("Loading ROUGE model...", end="\r")

    target_score = 'both'
    hidden_dim = 64
    epochs = 10
    lr = 0.01

    ROUGE_MODEL_PATH = f'./models/rouge_based_models/ROUGE_Based_Model_{target_score}_{hidden_dim}_{epochs}_{lr}.pt'

    model_loader = ROUGEModelLoader(merged_train, hidden_dim, target_score)
    if os.path.exists(ROUGE_MODEL_PATH):
        rouge_model = model_loader.model
        rouge_model.load_state_dict(torch.load(ROUGE_MODEL_PATH))
    else:
        print("Loading ROUGE model...")
        print("Specified ROUGE Based Model doesn't exist. Training it now.")
        print(model_loader.train(epochs, lr))
        rouge_model = model_loader.model

    print("Loading ROUGE model - ok")

    # LGBM
    model_path = '../models/lgbm_models/lgbm_model.joblib'  # Adjust the file type if needed

    if os.path.exists(model_path):
        print("Loading LGBM model...", end='\r')
        lgbm_model = joblib.load(model_path)
        print("Loading LGBM model - ok")
    else:
        print("Loading LGBM model...", end='\r')
        lgbm_model = AIO(merged_train, merged_test)
        lgbm_model.run()
        # lgbm_model = lgbm.model
        print("Loading LGBM model - ok")

        # Save the model
        print("Saving LGBM model...", end='\r')
        joblib.dump(lgbm_model, model_path)
        print("Saving LGBM model - ok")

    # Ensemble

    hidden_layers = 1
    hidden_dim = 64

    model = EnsembleNN(
        transformer,
        tokenizer,
        rouge_model,
        lgbm_model,
        # Adjust max length to fitted model
        MAX_LENGTH,
        hidden_layers=hidden_layers,
        hidden_dim=hidden_dim,
        device='cuda'
    )

    log("model.predict example")
    log(model.predict(merged_test, False))

    log('model.predict(merged_train, ensemble=False)')
    prediction_df = model.predict(merged_train, ensemble=False)  # <===
    log(prediction_df.head())

    prediction_df.to_csv('data/merged_df_for_ensemble.csv', index=False)

    from sklearn.metrics import mean_squared_error, r2_score
    from numpy import array

    y_true = prediction_df[['content', 'wording']]
    y_true = array(y_true)

    for model_name in ['transformer', 'rouge', 'lgbm']:
        y_pred = prediction_df[[f'{model_name}_content', f'{model_name}_wording']]
        y_pred = array(y_pred)

        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        performance = {'y_pred': rmse, 'R2': r2, 'MSE': mse}
        log(model_name, verbose=False)
        log_dict(performance, verbose=False)
        log()

    log('testing: model.predictions = prediction_df')
    log(model(merged_train.text[0],
              merged_train.prompt_title[0],
              merged_train.prompt_question[0],
              merged_train.prompt_text[0]
              ))

    log(verbose=False)
    log('-' * 10, end='', verbose=False)
    log(" Training ", end='')
    log('-' * 10)
    log(verbose=False)

    epochs = 10
    lr = 0.01

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam
    optimizer = optimizer(list(model.parameters()))
    optimizer.lr = lr

    for epoch in tqdm(range(epochs), desc='Training'):
        train_df = prediction_df.sample(frac=1).reset_index(drop=True)

        y_true = []
        y_pred = []

        for index, summary in tqdm(train_df.iterrows(), total=len(train_df), leave=False):
            target = torch.Tensor([summary.content, summary.wording]).to('cuda')

            transformer_preds = torch.Tensor([summary.transformer_content, summary.transformer_wording])
            rouge_preds = torch.Tensor([summary.rouge_content, summary.rouge_wording])
            lgbm_preds = torch.Tensor([summary.lgbm_content, summary.lgbm_wording])

            predictions = model(summary.text, summary.prompt_title, summary.prompt_question, summary.prompt_text,
                                transformer_preds,
                                rouge_preds,
                                lgbm_preds
                                )
            loss = criterion(predictions, target)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            y_true.append([*(float(x) for x in target)])
            y_pred.append([*(float(x) for x in predictions)])

        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        performance = {'RMSE': rmse, 'R2': r2, 'MSE': mse}

        score_map = {
            1: 'content',
            2: 'wording'
        }

        log("\nEpoch", epoch + 1, verbose=False)
        log_dict(performance, verbose=False)

    performance = {}
    for i in range(len(y_true[0])):
        y_true_i = [y[i] for y in y_true]
        y_pred_i = [y[i] for y in y_pred]

        rmse = mean_squared_error(y_true_i, y_pred_i, squared=False)
        mse = mean_squared_error(y_true_i, y_pred_i)
        r2 = r2_score(y_true_i, y_pred_i)

        i = score_map[i + 1]
        performance |= {f'RMSE_{i}': rmse, f'R2_{i}': r2, f'MSE_{i}': mse}

        log_dict(performance, verbose=False)

    val_predicted_df = model.predict(merged_val)

    y_true = []
    y_pred = []

    for i, row in val_predicted_df.iterrows():
        y_true.append([row.content, row.wording])
        y_pred.append([row.ensemble_content, row.ensemble_wording])

    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    performance = {'RMSE': rmse, 'R2': r2, 'MSE': mse}

    score_map = {
        1: 'content',
        2: 'wording'
    }

    for i in range(len(y_true[0])):
        y_true_i = [y[i] for y in y_true]
        y_pred_i = [y[i] for y in y_pred]

        rmse = mean_squared_error(y_true_i, y_pred_i, squared=False)
        mse = mean_squared_error(y_true_i, y_pred_i)
        r2 = r2_score(y_true_i, y_pred_i)

        i = score_map[i + 1]

        performance |= {f'RMSE_{i}': rmse, f'R2_{i}': r2, f'MSE_{i}': mse}

    log("\nValidation:")
    log_dict(performance, verbose=False)

    log()
    log("=" * 20, end="")
    log(datetime.now().strftime(" Finished: %d/%m/%Y %H:%M:%S "), end="")
    log("=" * 20)
    log()

    torch.save(model.state_dict(), f'models/ensembles/ensemble_{hidden_layers}_{hidden_dim}_{epochs}_{lr}.pt')
