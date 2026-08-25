"""
Walk-forward backtesting engine.

STATUS: not yet implemented. Planned approach (see README roadmap):
train the council on fights up to date T, evaluate on fights in
(T, T + step], slide T forward, and aggregate Brier/ROI across all
windows. This is what turns "we validated on a held-out set" into
"we validated the way we'll actually operate" (never training on
future information relative to a given prediction).
"""


def walk_forward_backtest(*args, **kwargs):
    raise NotImplementedError(
        "backtesting.engine.walk_forward_backtest: not implemented yet. "
        "Needs real historical fight outcomes from the scraper before "
        "this is meaningful — see ingestion/ufc_scraper.py."
    )
