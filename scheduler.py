"""
scheduler.py
------------
Always-on scheduler for Railway/Render cloud deployment.
Runs daily_picks.py every morning at 10:00 AM ET automatically.

This is the entry point for the cloud server.
Railway/Render will run this file 24/7.

Environment variables (set in Railway dashboard):
  TELEGRAM_TOKEN   — your bot token
  TELEGRAM_CHAT_ID — your chat ID
  BANKROLL         — starting bankroll (default 1000)
  MIN_EDGE         — minimum edge threshold (default 0.06)
  SEND_TIME        — time to send picks in ET (default "10:00")
"""

import os
import sys
import json
import time
import schedule
import logging
from pathlib import Path
from datetime import datetime

# Set up logging so Railway shows output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

CONFIG_FILE = Path("config.json")


def load_config_from_env() -> dict:
    """
    Load config from environment variables (Railway dashboard)
    falling back to config.json for local development.
    """
    config = {
        "telegram_token": os.environ.get("TELEGRAM_TOKEN", ""),
        "chat_id":        os.environ.get("TELEGRAM_CHAT_ID", ""),
        "bankroll":       float(os.environ.get("BANKROLL", 1000)),
        "min_edge":       float(os.environ.get("MIN_EDGE", 0.06)),
        "send_time":      os.environ.get("SEND_TIME", "10:00"),
    }

    # Fall back to config.json for local use
    if CONFIG_FILE.exists():
        saved = json.load(open(CONFIG_FILE))
        for k, v in saved.items():
            if not config.get(k):   # env vars take priority
                config[k] = v

    return config


def run_daily_picks():
    """Run the daily picks and send to Telegram."""
    log.info("Running daily picks...")

    config = load_config_from_env()

    # Write config.json so daily_picks.py can read it
    json.dump(config, open(CONFIG_FILE, "w"), indent=2)

    try:
        from daily_picks import (
            fetch_todays_games,
            run_k_predictions,
            build_message,
            send_telegram,
        )

        log.info("Fetching today's games...")
        games = fetch_todays_games()

        if games.empty:
            msg = (f"⚾ MLB Props — {datetime.now().strftime('%B %d, %Y')}\n\n"
                   f"No games today.")
            send_telegram(msg, config["telegram_token"], config["chat_id"])
            log.info("No games today — sent notification")
            return

        log.info(f"Found {len(games)} games")

        log.info("Running K predictions...")
        predictions = run_k_predictions(games)
        log.info(f"Generated {len(predictions)} predictions")

        message = build_message(games, predictions, config)
        send_telegram(message, config["telegram_token"], config["chat_id"])
        log.info("Daily picks sent successfully ✅")

    except Exception as e:
        log.error(f"Daily picks failed: {e}")
        # Send error notification so you know something went wrong
        try:
            from daily_picks import send_telegram
            send_telegram(
                f"⚠️ MLB Bot error: {str(e)[:200]}",
                config["telegram_token"],
                config["chat_id"]
            )
        except:
            pass


def run_heartbeat():
    """Send a weekly status check so you know the server is alive."""
    config = load_config_from_env()
    try:
        from daily_picks import send_telegram
        msg = (f"💚 MLB Bot is running\n"
               f"Server time: {datetime.now().strftime('%B %d %I:%M %p')}\n"
               f"Next picks: tomorrow at {config.get('send_time','10:00')} ET")
        send_telegram(msg, config["telegram_token"], config["chat_id"])
        log.info("Heartbeat sent")
    except Exception as e:
        log.error(f"Heartbeat failed: {e}")


def main():
    config = load_config_from_env()
    send_time = config.get("send_time", "10:00")

    log.info("=" * 50)
    log.info("MLB Props Bot starting up")
    log.info(f"Daily picks scheduled for {send_time} ET")
    log.info(f"Telegram configured: {'yes' if config.get('telegram_token') else 'NO - check env vars'}")
    log.info("=" * 50)

    # Schedule daily picks
    schedule.every().day.at(send_time).do(run_daily_picks)

    # Weekly heartbeat every Monday at 9am
    schedule.every().monday.at("09:00").do(run_heartbeat)

    # Run immediately on startup so you know it's working
    log.info("Running initial picks on startup...")
    run_daily_picks()

    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)   # check every minute


if __name__ == "__main__":
    main()
