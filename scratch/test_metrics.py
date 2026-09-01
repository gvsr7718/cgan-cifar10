"""Validation tests for evaluation metrics utilities."""

import os
import sys
import tempfile

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

from src.evaluation.metrics import (
    MetricTracker,
    export_metrics_to_json,
    generate_summary_report,
)


def run_tests():
    print("=" * 70)
    print("  Metrics Utilities — Validation Tests")
    print("=" * 70)

    all_passed = True

    # ---------------------------------------------------------------
    # Test MetricTracker
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST: MetricTracker")
    print("-" * 70)

    tracker = MetricTracker()

    tracker.update(
        {
            "d_loss": 1.2,
            "g_loss": 0.8,
        }
    )

    tracker.update(
        {
            "d_loss": 1.0,
            "g_loss": 0.9,
        }
    )

    history = tracker.get_history()

    history_ok = (
        history["d_loss"] == [1.2, 1.0]
        and history["g_loss"] == [0.8, 0.9]
    )

    print(
        f"  Metric history recorded: "
        f"{'PASS' if history_ok else 'FAIL'}"
    )

    latest_ok = (
        tracker.latest("d_loss") == 1.0
        and tracker.latest("g_loss") == 0.9
    )

    print(
        f"  Latest values correct: "
        f"{'PASS' if latest_ok else 'FAIL'}"
    )

    epoch_ok = tracker.epoch_count() == 2

    print(
        f"  Epoch count correct: "
        f"{'PASS' if epoch_ok else 'FAIL'}"
    )

    if not (history_ok and latest_ok and epoch_ok):
        all_passed = False

    # ---------------------------------------------------------------
    # Test JSON export and summary report
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST: File Export")
    print("-" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        json_path = os.path.join(temp_dir, "metrics.json")
        report_path = os.path.join(temp_dir, "summary.txt")

        export_metrics_to_json(history, json_path)
        generate_summary_report(history, report_path)

        json_ok = os.path.isfile(json_path)
        report_ok = os.path.isfile(report_path)

        print(
            f"  JSON file created: "
            f"{'PASS' if json_ok else 'FAIL'}"
        )

        print(
            f"  Summary report created: "
            f"{'PASS' if report_ok else 'FAIL'}"
        )

        if not (json_ok and report_ok):
            all_passed = False

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)

    if all_passed:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")

    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)