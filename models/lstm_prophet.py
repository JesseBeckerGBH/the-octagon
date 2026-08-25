"""
LSTM Prophet — sequential form/momentum model over each fighter's fight
history.

STATUS: stub. The architecture below is real (a small bidirectional LSTM
over per-fight feature sequences), but it has nothing to train on yet:
ingestion doesn't currently emit the ordered per-fighter sequence data this
needs (round-by-round strike/TD stats), only won/lost outcomes. Wiring
this up is blocked on extending ingestion/ufc_scraper.py to parse fight
detail pages, not on model code.

Excluded from the live council blend (see config/settings.yaml) until then.
"""

import polars as pl

from models.base import Prophet


class LSTMProphet(Prophet):
    name = "lstm"

    def __init__(self, hidden_size: int = 32, num_layers: int = 1):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self._model = None

    def _build(self, input_size: int):
        import torch.nn as nn

        class SequenceLSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size, hidden_size, num_layers,
                    batch_first=True, bidirectional=True,
                )
                self.head = nn.Linear(hidden_size * 2, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                last = out[:, -1, :]
                return self.head(last)

        return SequenceLSTM(input_size, self.hidden_size, self.num_layers)

    def fit(self, features: pl.DataFrame, labels: pl.Series) -> "LSTMProphet":
        raise NotImplementedError(
            "LSTMProphet needs per-fighter sequence data (round-by-round stats) "
            "that ingestion/ufc_scraper.py doesn't parse yet. See README roadmap."
        )

    def predict_proba(self, features: pl.DataFrame) -> list[float]:
        raise NotImplementedError("LSTMProphet is not trained — see fit().")
