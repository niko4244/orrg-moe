from datetime import datetime, timedelta, timezone
from boltons.timeutils import isoparse, dt_to_timestamp

def test_00_basic_epoch():
    assert isoparse('1970-01-01T00:00:00') == datetime(1970, 1, 1)

def test_01_frac_3_digits_851():
    assert isoparse('1970-01-01T00:00:00.851').microsecond == 851000

def test_02_frac_leading_zero_051():
    assert isoparse('2020-01-02T03:04:05.051').microsecond == 51000

def test_03_frac_6_digits_000001():
    assert isoparse('2020-01-02T03:04:05.000001').microsecond == 1

def test_04_frac_overprecise_truncates():
    assert isoparse('2020-01-01T00:00:00.123456789').microsecond == 123456

def test_05_frac_1_digit_5():
    assert isoparse('2020-01-02T03:04:05.5').microsecond == 500000

def test_06_frac_2_digits_05():
    assert isoparse('2020-01-02T03:04:05.05').microsecond == 50000

def test_07_frac_4_digits_0001():
    assert isoparse('2020-01-02T03:04:05.0001').microsecond == 100

def test_08_isoformat_milliseconds():
    dt=datetime(2020,1,2,3,4,5,851000); assert isoparse(dt.isoformat(timespec='milliseconds')) == dt

def test_09_isoformat_microseconds():
    dt=datetime(2020,1,2,3,4,5,851000); assert isoparse(dt.isoformat(timespec='microseconds')) == dt

def test_10_isoformat_default():
    dt=datetime(2020,1,2,3,4,5,851000); assert isoparse(dt.isoformat()) == dt

def test_11_tz_zulu():
    assert isoparse('2020-01-02T03:04:05Z') == datetime(2020,1,2,3,4,5,tzinfo=timezone.utc)

def test_12_tz_offset_plus_0530():
    assert isoparse('2020-01-02T03:04:05+05:30').utcoffset() == timedelta(hours=5,minutes=30)

def test_13_tz_offset_with_fraction():
    assert isoparse('2020-01-02T03:04:05.250-0800').utcoffset() == timedelta(hours=-8)

def test_14_tz_aware_isoformat_roundtrip():
    dt=datetime(2020,1,2,3,4,5,6,tzinfo=timezone(timedelta(hours=-3,minutes=-30))); assert isoparse(dt.isoformat()) == dt

def test_15_tz_invalid_offset_raises():
    import pytest
    with pytest.raises(ValueError):
        isoparse('2020-01-02T03:04:05+25:00')

def test_16_dt_to_timestamp_fraction():
    assert abs(dt_to_timestamp(isoparse('1970-01-01T00:00:00.851')) - 0.851) < 1e-9
