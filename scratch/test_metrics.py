"""Validation tests for evaluation metrics utilities."""

import os
import sys
import json
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
    # Test MetricTracker functionality
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST: MetricTracker Operations")
    print("-" * 70)

    tracker = MetricTracker()
    
    empty_ok = len(tracker.get_history()) == 0
    print(f"  Starts empty: {'PASS' if empty_ok else 'FAIL'}")
    if not empty_ok:
        all_passed = False

    # Test update()
    tracker.update({"d_loss": 1.5, "g_loss": 0.5})
    tracker.update({"d_loss": 1.2, "g_loss": 0.8})
    
    history = tracker.get_history()
    update_ok = history["d_loss"] == [1.5, 1.2] and history["g_loss"] == [0.5, 0.8]
    print(f"  update() records multiple metrics: {'PASS' if update_ok else 'FAIL'}")
    if not update_ok:
        all_passed = False

    # Test add()
    tracker.add("fid", 150.0)
    tracker.add("fid", 120.0)
    
    add_ok = tracker.get_history()["fid"] == [150.0, 120.0]
    print(f"  add() records single metric: {'PASS' if add_ok else 'FAIL'}")
    if not add_ok:
        all_passed = False

    # Test latest()
    latest_ok = tracker.latest("d_loss") == 1.2 and tracker.latest("fid") == 120.0
    print(f"  latest() returns most recent: {'PASS' if latest_ok else 'FAIL'}")
    if not latest_ok:
        all_passed = False

    # Test epoch_count()
    epoch_ok = tracker.epoch_count() == 2
    print(f"  epoch_count() is correct: {'PASS' if epoch_ok else 'FAIL'}")
    if not epoch_ok:
        all_passed = False

    # Test reset()
    tracker.reset()
    reset_ok = len(tracker.get_history()) == 0 and tracker.epoch_count() == 0
    print(f"  reset() clears history: {'PASS' if reset_ok else 'FAIL'}")
    if not reset_ok:
        all_passed = False

    # ---------------------------------------------------------------
    # Test file exports
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST: File Exports")
    print("-" * 70)

    # Repopulate tracker for export tests
    tracker.update({"fid": 100.0, "acc": 0.90})
    tracker.update({"fid": 80.0, "acc": 0.95})
    history = tracker.get_history()

    with tempfile.TemporaryDirectory() as temp_dir:
        json_path = os.path.join(temp_dir, "metrics.json")
        report_path = os.path.join(temp_dir, "summary.txt")

        # Test JSON Export
        export_metrics_to_json(history, json_path)
        json_exists = os.path.isfile(json_path)
        
        json_valid = False
        if json_exists:
            with open(json_path, "r") as f:
                loaded_data = json.load(f)
                json_valid = loaded_data == history

        print(f"  export_metrics_to_json() creates valid JSON: {'PASS' if json_valid else 'FAIL'}")
        if not json_valid:
            all_passed = False

        # Test Summary Report
        generate_summary_report(history, report_path)
        report_exists = os.path.isfile(report_path)
        
        report_valid = False
        if report_exists:
            with open(report_path, "r") as f:
                content = f.read()
                # Check for critical expected elements
                has_metric_names = "fid" in content and "acc" in content
                has_epoch_counts = "Epochs recorded: 2" in content
                has_latest = "Latest: 80.0" in content or "Latest: 0.95" in content
                has_min = "Minimum: 80.0" in content or "Minimum: 0.9" in content
                has_max = "Maximum: 100.0" in content or "Maximum: 0.95" in content
                report_valid = has_metric_names and has_epoch_counts and has_latest and has_min and has_max

        print(f"  generate_summary_report() format is correct: {'PASS' if report_valid else 'FAIL'}")
        if not report_valid:
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