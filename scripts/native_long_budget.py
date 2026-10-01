"""Shared native-campaign GPU-hour accounting and explicit caps."""

import json
import math
import os
import tempfile


def write_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = stream.name
        try:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def budget_cap(plan):
    value = float(plan.get("lifetime_cap_gpu_hours", 192.))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("GPU-hour cap must be positive and finite")
    return value


def ledger_events(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def merge_events(groups):
    events = {}
    for rows in groups:
        for row in rows:
            hours = row.get("gpu_hours")
            if not isinstance(hours, (int, float)) or not math.isfinite(hours) or hours < 0:
                raise ValueError("GPU hours must be nonnegative and finite")
            key = row["event_id"]
            if key in events and events[key] != row:
                raise ValueError(f"conflicting ledger event: {key}")
            events[key] = row
    return [events[key] for key in sorted(events)]


def remaining_budget(root, plan):
    used = sum(r["gpu_hours"] for r in merge_events([ledger_events(root / "COST_LEDGER.jsonl")]))
    return budget_cap(plan) - plan["prior"]["gpu_hours"] - used
