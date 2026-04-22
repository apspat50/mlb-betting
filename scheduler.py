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


def run_batter_picks():
    """Run batter props (H, TB, HR) at noon when lineups are posted."""
    log.info("Running batter picks...")
    config = load_config_from_env()
    try:
        from daily_batter_picks import (
            fetch_todays_games,
            run_batter_predictions,
            build_message,
            save_batter_predictions,
            send_telegram,
        )

        games = fetch_todays_games()
        if games.empty:
            log.info("No games today — skipping batter picks")
            return

        predictions = run_batter_predictions(games)
        if not predictions:
            log.info("No batter predictions generated")
            return

        date_str = datetime.now().strftime("%Y-%m-%d")
        save_batter_predictions(predictions, date_str)

        message = build_message(predictions, games)
        send_telegram(message, config["telegram_token"], config["chat_id"])
        log.info("Batter picks sent successfully ✅")

    except Exception as e:
        log.error(f"Batter picks failed: {e}")


def run_daily_results():
    """Fetch actual K results and compare to today's predictions."""
    log.info("Running daily results check...")
    config = load_config_from_env()
    try:
        from daily_results import (
            load_predictions,
            fetch_all_results,
            compare_predictions,
            build_results_message,
            save_results,
        )
        from daily_picks import send_telegram

        date_str = datetime.now().strftime("%Y-%m-%d")
        preds_data    = load_predictions(date_str)
        pitcher_preds = preds_data.get("pitchers", [])

        if not pitcher_preds:
            log.info("No predictions found for today — skipping results")
            return

        actuals = fetch_all_results(date_str)
        if not actuals:
            log.info("No completed games yet — skipping results")
            return

        compared = compare_predictions(pitcher_preds, actuals)
        save_results(compared, date_str)
        message = build_results_message(compared, date_str)
        send_telegram(message, config["telegram_token"], config["chat_id"])
        log.info("Daily results sent successfully ✅")

    except Exception as e:
        log.error(f"Daily results failed: {e}")


def run_weekly_summary():
    """Send weekly K prediction recap every Sunday night."""
    log.info("Running weekly summary...")
    config = load_config_from_env()
    try:
        from weekly_summary import (
            load_week_predictions,
            fill_missing_actuals,
            score_rows,
            build_weekly_message,
        )
        from daily_picks import send_telegram

        rows = load_week_predictions(days=7)
        if not rows:
            log.info("No prediction files found for the week")
            return

        rows  = fill_missing_actuals(rows)
        stats = score_rows(rows)
        message = build_weekly_message(rows, stats, days=7)
        send_telegram(message, config["telegram_token"], config["chat_id"])
        log.info("Weekly summary sent successfully ✅")

    except Exception as e:
        log.error(f"Weekly summary failed: {e}")


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

    # Batter props at noon ET (lineups posted by then)
    schedule.every().day.at("12:00").do(run_batter_picks)

    # Daily results check at 11:30 PM ET (most games finished by then)
    schedule.every().day.at("23:30").do(run_daily_results)

    # Weekly summary every Sunday at 11:00 PM ET
    schedule.every().sunday.at("23:00").do(run_weekly_summary)

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
