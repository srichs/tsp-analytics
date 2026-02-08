from datetime import date

from pandas.testing import assert_frame_equal

from tsp import TspAnalytics


def test_current_price_changes_accepts_as_of_date(tsp_price: TspAnalytics) -> None:
    target_date = date(2024, 1, 3)

    changes = tsp_price.get_current_price_changes(as_of=target_date)
    expected = tsp_price.get_price_changes_as_of(as_of=target_date)
    assert_frame_equal(changes, expected)

    changes_long = tsp_price.get_current_price_changes_long(as_of=target_date)
    expected_long = tsp_price.get_price_changes_as_of_long(as_of=target_date)
    assert_frame_equal(changes_long, expected_long)

    changes_dict = tsp_price.get_current_price_changes_dict(as_of=target_date)
    expected_dict = tsp_price.get_price_changes_as_of_dict(as_of=target_date)
    assert changes_dict == expected_dict


def test_current_price_changes_per_fund_accepts_as_of_date(
    tsp_price: TspAnalytics,
) -> None:
    target_date = date(2024, 1, 3)

    per_fund = tsp_price.get_current_price_changes_per_fund(as_of=target_date)
    expected = tsp_price.get_price_changes_as_of_per_fund(as_of=target_date)
    assert_frame_equal(per_fund, expected)

    per_fund_long = tsp_price.get_current_price_changes_per_fund_long(as_of=target_date)
    expected_long = tsp_price.get_price_changes_as_of_per_fund_long(as_of=target_date)
    assert_frame_equal(per_fund_long, expected_long)

    per_fund_dict = tsp_price.get_current_price_changes_per_fund_dict(as_of=target_date)
    expected_dict = tsp_price.get_price_changes_as_of_per_fund_dict(as_of=target_date)
    assert per_fund_dict == expected_dict


def test_current_price_snapshot_accepts_as_of_date(tsp_price: TspAnalytics) -> None:
    target_date = date(2024, 1, 3)

    snapshot = tsp_price.get_current_price_snapshot(as_of=target_date)
    expected = tsp_price.get_price_snapshot_as_of(as_of=target_date)
    assert_frame_equal(snapshot, expected)

    snapshot_long = tsp_price.get_current_price_snapshot_long(as_of=target_date)
    expected_long = tsp_price.get_price_snapshot_as_of_long(as_of=target_date)
    assert_frame_equal(snapshot_long, expected_long)

    snapshot_dict = tsp_price.get_current_price_snapshot_dict(as_of=target_date)
    expected_dict = tsp_price.get_price_snapshot_as_of_dict(as_of=target_date)
    assert snapshot_dict == expected_dict


def test_current_price_report_long_accepts_as_of_date(tsp_price: TspAnalytics) -> None:
    target_date = date(2024, 1, 3)

    report_long = tsp_price.get_current_price_report_long(as_of=target_date)
    expected = tsp_price.get_current_price_report(as_of=target_date).reset_index()
    assert_frame_equal(report_long, expected)

    latest_long = tsp_price.get_current_price_report_long()
    expected_latest = tsp_price.get_latest_price_report_long()
    assert_frame_equal(latest_long, expected_latest)
