from torch import nn, Tensor, cat, flatten
from rouge_score import *


class Model(nn.Module):
    def __init__(self, hidden_dim, output_dim, summary_len_mean, summary_len_std):
        super().__init__()

        self.summary_len_mean = summary_len_mean
        self.summary_len_std = summary_len_std

        input_dim = 4 * 3 + 1
        # Input Dimensions:
        # 4 ROUGE-Scores:
        # ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum
        # 3 values per ROUGE-Score:
        # recall, precision, f-score
        # 1 normalized length of the summary

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.non_lin = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

        self.device = 'cpu'
        self.to(self.device)

    def forward(self, prompt_text, summary_text, scores=None):
        if scores is None:
            scores = self._calculate_rouge_scores(prompt_text, summary_text)
            scores = flatten(scores)
        summary_len_norm = self._summary_len_norm(summary_text)

        result = cat((scores, summary_len_norm))
        result = self.layer1(result)
        result = self.non_lin(result)
        result = self.layer2(result)

        return result

    def _calculate_rouge_scores(self, prompt_text, summary_text):
        rouge_1 = rouge_n(prompt_text, summary_text, 1)
        rouge_2 = rouge_n(prompt_text, summary_text, 2)
        rouge_L = rouge_l(prompt_text, summary_text)
        rouge_Lsum = rouge_lsum(prompt_text, summary_text)

        scores = [
            (rouge_1.precision, rouge_1.recall, rouge_1.fmeasure),
            (rouge_2.precision, rouge_2.recall, rouge_2.fmeasure),
            (rouge_L.precision, rouge_L.recall, rouge_L.fmeasure),
            (rouge_Lsum.precision, rouge_Lsum.recall, rouge_Lsum.fmeasure)
        ]

        return Tensor(scores).to(self.device)

    def _summary_len_norm(self, summary_text):
        zscore = (len(summary_text) - self.summary_len_mean) / self.summary_len_std
        return Tensor((zscore,)).to(self.device)

    def to(self, device, *args, **kwargs):
        super().to(device, *args, **kwargs)
        self.device = device
        return self


if __name__ == "__main__":
    from utils import *

    hidden_dim = 64
    output_dim = 1

    device, path = setup()
    summaries_df, prompts_df = get_data('../kaggle/input/commonlit-evaluate-student-summaries')

    text_len_mean = summaries_df.text.apply(len).mean()
    text_len_std = summaries_df.text.apply(len).std()

    model = Model(hidden_dim, output_dim, text_len_mean, text_len_std).to(device)

    print(model)
