#!/usr/bin/env python3
"""
THE OCTAGON — paid odds/sharp-money ingestion.

STATUS: not yet implemented against a live provider. This defines the
interface the Sharp Money prophet and the validator's CLV calculation
expect, so it can be wired up to The Odds API or SportsData.io without
changing anything downstream.

To implement: call the provider on a schedule (APScheduler), normalize
each snapshot into the shape below, and INSERT into odds_history.
"""

from dataclasses import dataclass


@dataclass
class OddsSnapshot:
    fight_id: str
    timestamp: str
    book: str
    moneyline_a: float
    moneyline_b: float


def fetch_odds_snapshot(fight_id: str) -> list[OddsSnapshot]:
    """TODO: implement against ODDS_API_KEY / SPORTSDATA_IO_API_KEY.

    Raises NotImplementedError until a provider is wired up — deliberately
    loud rather than silently returning fake data.
    """
    raise NotImplementedError(
        "odds_ingestion.fetch_odds_snapshot: no provider configured yet. "
        "See README roadmap — this blocks the Sharp Money prophet."
    )
