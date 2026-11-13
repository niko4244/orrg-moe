"""17 independent isoparse cases used as the boltons per-case oracle.

Each case is a name plus a callable body executed with isoparse/dt_to_timestamp
already imported. Results are emitted as JSON so every failure is attributable.
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.getcwd())

CASES = [
    ("basic_epoch", "assert isoparse('1970-01-01T00:00:00') == datetime(1970, 1, 1)"),
    ("frac_3_digits_851", "assert isoparse('1970-01-01T00:00:00.851').microsecond == 851000"),
    ("frac_leading_zero_051", "assert isoparse('2020-01-02T03:04:05.051').microsecond == 51000"),
    ("frac_6_digits_000001", "assert isoparse('2020-01-02T03:04:05.000001').microsecond == 1"),
    ("frac_overprecise_truncates", "assert isoparse('2020-01-01T00:00:00.123456789').microsecond == 123456"),
    ("frac_1_digit_5", "assert isoparse('2020-01-02T03:04:05.5').microsecond == 500000"),
    ("frac_2_digits_05", "assert isoparse('2020-01-02T03:04:05.05').microsecond == 50000"),
    ("frac_4_digits_0001", "assert isoparse('2020-01-02T03:04:05.0001').microsecond == 100"),
    ("isoformat_milliseconds", "dt=datetime(2020,1,2,3,4,5,851000); assert isoparse(dt.isoformat(timespec='milliseconds')) == dt"),
    ("isoformat_microseconds", "dt=datetime(2020,1,2,3,4,5,851000); assert isoparse(dt.isoformat(timespec='microseconds')) == dt"),
    ("isoformat_default", "dt=datetime(2020,1,2,3,4,5,851000); assert isoparse(dt.isoformat()) == dt"),
    ("tz_zulu", "assert isoparse('2020-01-02T03:04:05Z') == datetime(2020,1,2,3,4,5,tzinfo=timezone.utc)"),
    ("tz_offset_plus_0530", "assert isoparse('2020-01-02T03:04:05+05:30').utcoffset() == timedelta(hours=5,minutes=30)"),
    ("tz_offset_with_fraction", "assert isoparse('2020-01-02T03:04:05.250-0800').utcoffset() == timedelta(hours=-8)"),
    ("tz_aware_isoformat_roundtrip", "dt=datetime(2020,1,2,3,4,5,6,tzinfo=timezone(timedelta(hours=-3,minutes=-30))); assert isoparse(dt.isoformat()) == dt"),
    ("tz_invalid_offset_raises", "import pytest\nwith pytest.raises(ValueError):\n    isoparse('2020-01-02T03:04:05+25:00')"),
    ("dt_to_timestamp_fraction", "assert abs(dt_to_timestamp(isoparse('1970-01-01T00:00:00.851')) - 0.851) < 1e-9"),
]

PRELUDE = ("from datetime import datetime, timedelta, timezone\n"
           "from boltons.timeutils import isoparse, dt_to_timestamp\n")


def main():
    out = []
    for name, body in CASES:
        ns = {}
        try:
            exec(PRELUDE, ns)
            exec(body, ns)
            out.append({"case": name, "status": "PASSED", "detail": ""})
        except Exception as exc:  # noqa: BLE001 - record exact failure
            out.append({"case": name, "status": "FAILED",
                        "detail": f"{type(exc).__name__}: {exc}"[:500]})
    print(json.dumps(out))


if __name__ == "__main__":
    main()
