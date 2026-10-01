#!/usr/bin/env python3
"""Hermesume (Hermes + Yume/夢): sleep-time memory consolidation for Hermes Agent.

Runs NREM -> REM -> Dream Log, like a sleeping brain.

Usage:
    python3 hermesume.py              # full dream cycle
    python3 hermesume.py --dry-run    # plan only: nothing written to Hermes
    python3 hermesume.py --nrem-only  # extract facts, don't touch memory files
"""

import argparse
import json
import logging
import sys
import time

from config import HERMES_HOME
from dream_log import write_dream_log
from hermes_memory import TARGETS, load_limits, read_entries
from meta import Meta
from nrem import run_nrem
from rem import run_rem
from sessions import archive_episode_files, save_cursor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("hermesume")


def main():
    parser = argparse.ArgumentParser(description="Hermesume -- Hermes Agent memory consolidation")
    parser.add_argument("--nrem-only", action="store_true",
                        help="Only replay sessions and extract facts (no memory writes)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan the full cycle and write a dream log, but leave "
                             "Hermes memory, the session cursor and metadata untouched")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    dry_run = args.dry_run or args.nrem_only
    if dry_run:
        log.warning("=== DRY RUN -- Hermes memory will not be modified ===")

    log.info("=" * 60)
    log.info("Hermesume falling asleep... (HERMES_HOME=%s)", HERMES_HOME)
    log.info("=" * 60)
    start = time.time()
    now = time.time()
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))

    limits = load_limits()
    snapshot = {t: read_entries(t) for t in TARGETS if limits[t]["enabled"]}
    if not snapshot:
        log.warning("Both MEMORY.md and USER.md are disabled in Hermes config; nothing to do")
        return
    meta = Meta()

    log.info("")
    log.info("Phase 1: NREM (sessions -> facts)")
    log.info("-" * 40)
    nrem_result = run_nrem(snapshot, meta, now)

    rem_result = {"targets": {}}
    if not args.nrem_only:
        log.info("")
        log.info("Phase 2: REM (integration + homeostasis)")
        log.info("-" * 40)
        rem_result = run_rem(nrem_result, snapshot, limits, meta, now,
                             dry_run=dry_run, stamp=stamp)

    if not dry_run:
        meta.prune({t: read_entries(t) for t in snapshot})
        meta.save()
        if nrem_result["cursor"] is not None:
            save_cursor(nrem_result["cursor"])
        archive_episode_files(nrem_result["episode_files"])

    log.info("")
    log.info("Phase 3: Dream Log")
    log.info("-" * 40)
    log_path = write_dream_log(nrem_result, rem_result, dry_run=dry_run)

    elapsed = time.time() - start
    log.info("")
    log.info("=" * 60)
    log.info("Woke up. (%.1fs)  Dream log: %s", elapsed, log_path)
    log.info("=" * 60)

    nrem_summary = {k: v for k, v in nrem_result.items() if k != "facts"}
    nrem_summary["facts"] = [{k: v for k, v in f.items() if k != "vector"}
                             for f in nrem_result["facts"]]
    print(json.dumps({
        "dry_run": dry_run,
        "nrem": nrem_summary,
        "rem": rem_result,
        "dream_log": log_path,
        "elapsed_seconds": round(elapsed, 1),
    }, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error("Hermesume failed: %s: %s", type(e).__name__, e)
        try:
            from alerts import send_alert
            send_alert(e)
        except Exception:
            pass  # alert delivery failure should never mask the real error
        sys.exit(1)
