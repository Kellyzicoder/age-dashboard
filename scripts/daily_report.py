"""Send the 5pm attendance email. Run by .github/workflows/daily-report.yml (or by hand).

Safe to run many times: it only sends once per day (checked in the email_log table), and only inside
the evening window unless FORCE=1. Needs env vars DATABASE_URL, SMTP_USER, SMTP_PASSWORD.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import datetime as dt  # noqa: E402

import attendance as A  # noqa: E402
import report as R  # noqa: E402

WINDOW = (dt.time(16, 40), dt.time(19, 0))  # NZ time; first run inside the window sends


def main() -> int:
    now = dt.datetime.now(A.TZ)
    force = os.environ.get("FORCE") == "1"
    if not force and not (WINDOW[0] <= now.time() < WINDOW[1]):
        print(f"{now:%H:%M} NZ is outside the send window — nothing to do.")
        return 0
    missing = [k for k in ("DATABASE_URL", "SMTP_USER", "SMTP_PASSWORD") if not os.environ.get(k)]
    if missing:
        print("Missing secrets: " + ", ".join(missing))
        return 1
    store = A.SqlStore(os.environ["DATABASE_URL"])
    if not force and store.sent_on(now.date().isoformat(), "daily"):
        print(f"Today's report was already sent — skipping.")
        return 0
    out = R.send(store, os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"], kind="daily" if not force else "manual")
    print(f"Sent “{out['subject']}” to {', '.join(out['to'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
