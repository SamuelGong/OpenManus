#!/usr/bin/env python3
import re
import datetime
from pathlib import Path

# --- CONFIG ---
SRC_ROOT   = Path("logs")
DST_ROOT   = Path("new-logs")
# threshold: keep records at or after this moment
THRESHOLD  = datetime.datetime(2025, 4, 27, 14, 10, 0)
# regex to detect the start of a record, capturing the timestamp
RE_RECORD_START = re.compile(
    r"^\[([A-Z]+)\]"                      # [LEVEL]
    r"\[([0-9]{4}-[0-9]{2}-[0-9]{2} "     # [YYYY-MM-DD 
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3})\]"  # HH:mm:ss.SSS]
    r"\[\d+\]"                            # [process]
    r"\[[^\]]+:\d+\]:\s?"                 # [file.py:line]:
)

def filter_log_file(src_path: Path, dst_path: Path) -> None:
    """
    Read src_path, parse by log-record boundaries, and write only
    records >= THRESHOLD to dst_path.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with src_path.open("r", encoding="utf-8") as fin, \
         dst_path.open("w", encoding="utf-8") as fout:

        current_record = []
        current_ts     = None

        for line in fin:
            m = RE_RECORD_START.match(line)
            if m:
                # flush previous record
                if current_record and current_ts >= THRESHOLD:
                    fout.writelines(current_record)
                # start new record
                current_record = [line]
                ts_str = m.group(2)
                current_ts = datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
            else:
                # continuation of current record
                if current_record is not None:
                    current_record.append(line)

        # flush last record
        if current_record and current_ts >= THRESHOLD:
            fout.writelines(current_record)

def main():
    for src_log in SRC_ROOT.rglob("*.log"):
        rel = src_log.relative_to(SRC_ROOT)
        dst_log = DST_ROOT / rel
        filter_log_file(src_log, dst_log)

    print(f"Done. Filtered logs written under '{DST_ROOT}/'")

if __name__ == "__main__":
    main()
