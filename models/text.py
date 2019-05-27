import torch
import torch.nn as nn

from sru import SRU

from models.transformer import TransformerModel, load_openai_pretrained_model, DEFAULT_CONFIG

from common.config import PATH

__all__ = ["LSTMEmbedding", "TransformerEmbedding"]


class AText(nn.Module):
    def __init__(self):
        super().__init__()

    def get_lengths(self, sequence):
        sequence = sequence.contiguous().view(sequence.size(0), sequence.size(1), -1)
        mx, _ = torch.max(sequence, dim=2, keepdim=True)
        mask = torch.sign(mx)
        lengths = torch.argmax(mask, dim=1, keepdim=True) + 1
        lengths = lengths.view(-1)
        return lengths

    def select_last(self, sequence, length):
        (batch_size, max_length, out_size) = sequence.shape
        index = torch.arange(0, batch_size) * max_length
        index = index.to(length.device)
        index += length - 1
        flat = sequence.contiguous().view(-1, out_size)
        return flat[index]

    def set_requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = requires_grad


class LSTMEmbedding(AText):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.25, bidirectional=False):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )

    def forward(self, x, lengths=None):
        if lengths is None:
            lengths = self.get_lengths(x)
        x = x.permute(1, 0, 2)
        x, hn = self.lstm(x)
        x = x.permute(1, 0, 2)
        x = self.select_last(x, lengths)
        return x

class SRUEmbedding(AText):
    def __init__(self, input_size, output_size, num_layers=4, dropout=0.25):
        super().__init__()

        self.sru = SRU( input_size, output_size, num_layers=num_layers,
                        dropout=dropout, rnn_dropout=dropout,
                        use_tanh=True, has_skip_term=True,
                        v1=True, rescale=False)

    def forward(self, x, lengths=None):
        if lengths is None:
            lengths = self.get_lengths(x)
        x = x.permute(1, 0, 2)
        x, hn = self.sru(x)
        x = x.permute(1, 0, 2)
        x = self.select_last(x, lengths)
        return x


class TransformerEmbedding(AText):
    def __init__(self, emb_size=2400, pretrained=True, fc_dropout=0.5, transformer_freeze=True):
        super().__init__()

        self.transformer = TransformerModel(DEFAULT_CONFIG)
        load_openai_pretrained_model(
            self.transformer, 
            path=PATH["MODELS"]["TRANSFORMER_PRETRAINED"]["PATH"], 
            path_names=PATH["MODELS"]["TRANSFORMER_PRETRAINED"]["PATH_NAMES"])

        self.fc = nn.Linear(DEFAULT_CONFIG.n_embd, emb_size, bias=True)
        self.fc_dropout = nn.Dropout(p=fc_dropout)

        self.set_transformer_requires_grad(not transformer_freeze)

    def forward(self, x, lengths=None):
        if lengths is None:
            lengths = self.get_lengths(x)

        x = self.transformer(x)
        x = self.select_last(x, lengths)

        x = self.fc(self.fc_dropout(x))

        return x

    def set_transformer_requires_grad(self, requires_grad):
        for param in self.transformer.parameters():
            param.requires_grad = requires_grad
