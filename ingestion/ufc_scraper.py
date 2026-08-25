#!/usr/bin/env python3
"""
THE OCTAGON — UFC historical results scraper.

Pulls event + fight results from ufcstats.com. This is the free/historical
half of the data story; ingestion/odds_ingestion.py covers the paid,
real-time half (line movement, sharp money).

Scraping vs. paid APIs, and why we use both:
  - ufcstats.com is free and has rich per-round strike/TD stats, but the
    markup can change without notice and there's no SLA.
  - Paid odds APIs (The Odds API, SportsData.io) are structured and
    real-time but cost money and are rate-limited.
  Historical depth comes from scraping; live sharp-money signal comes from
  the paid feed. Neither alone is enough for the Sharp Money prophet.
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import duckdb
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from scripts.create_schema import init_schema

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "http://ufcstats.com"
DATA_RAW = Path("data/raw")
DB_PATH = Path("data/processed/octagon.duckdb")
USER_AGENT = "TheOctagon/2.0 (JBAnalytics LLC)"


def get_soup(url: str, retries: int = 3) -> BeautifulSoup:
    headers = {"User-Agent": USER_AGENT}
    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=20)
            r.raise_for_status()
            return BeautifulSoup(r.text, "lxml")
        except Exception as e:  # noqa: BLE001 - deliberately broad, we retry
            last_exc = e
            logger.warning("Attempt %d failed for %s: %s", attempt + 1, url, e)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {url}") from last_exc


def scrape_event_fights(event_url: str) -> list[dict]:
    soup = get_soup(event_url)
    fights = []
    rows = soup.select("tr.b-fight-details__table-row")
    for row in rows:
        fighters = row.select("a.b-link_style_black")
        if len(fighters) < 2:
            continue
        fights.append(
            {
                "fight_id": f"{event_url.rstrip('/').split('/')[-1]}_{len(fights)}",
                "fighter_a": fighters[0].get_text(strip=True),
                "fighter_b": fighters[1].get_text(strip=True),
                # TODO: parse winner/method/round/time — ufcstats renders these
                # per-fight rather than on the event list page; see the fight
                # detail page for the full stat breakdown.
            }
        )
    return fights


def list_completed_events(limit: int | None = None) -> list[dict]:
    url = f"{BASE_URL}/statistics/events/completed?page=all"
    soup = get_soup(url)
    rows = soup.select("tr.b-statistics__table-row")[1:]
    if limit:
        rows = rows[:limit]

    events = []
    for row in rows:
        link = row.find("a")
        if not link:
            continue
        cols = row.find_all("td")
        location = cols[-1].get_text(strip=True) if cols else ""
        event_url = link["href"]
        events.append(
            {
                "event_id": event_url.rstrip("/").split("/")[-1],
                "name": link.get_text(strip=True),
                "location": location,
                "url": event_url,
                "scraped_at": datetime.utcnow().isoformat(),
            }
        )
    return events


def run(limit: int = 30) -> None:
    """Scrape up to `limit` most-recent completed events into DuckDB.

    limit exists so a fresh checkout can smoke-test the pipeline in minutes
    instead of the 15-40 min a full historical pull takes. Run with a large
    limit (or add a --full flag override) for the real historical backfill.
    """
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH))
    init_schema(con)

    logger.info("Fetching completed-events index...")
    events = list_completed_events(limit=limit)

    for ev in tqdm(events, desc="Scraping events"):
        try:
            fights = scrape_event_fights(ev["url"])

            raw_file = DATA_RAW / f"{ev['event_id']}.json"
            raw_file.write_text(json.dumps({"event": ev, "fights": fights}, indent=2))

            con.execute(
                "INSERT OR IGNORE INTO events VALUES (?, ?, ?, ?, ?)",
                (ev["event_id"], ev["name"], "", ev["location"], datetime.utcnow()),
            )
            for f in fights:
                con.execute(
                    "INSERT OR IGNORE INTO fights "
                    "(fight_id, event_id, fighter_a, fighter_b, winner, method, round, time) "
                    "VALUES (?, ?, ?, ?, NULL, NULL, NULL, NULL)",
                    (f["fight_id"], ev["event_id"], f["fighter_a"], f["fighter_b"]),
                )

            time.sleep(1.2)  # be a polite scraper
        except Exception as e:  # noqa: BLE001
            logger.error("Failed on %s: %s", ev["name"], e)

    con.close()
    logger.info("Scrape complete. %d events processed.", len(events))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30, help="Number of recent events to scrape")
    args = parser.parse_args()
    run(limit=args.limit)
