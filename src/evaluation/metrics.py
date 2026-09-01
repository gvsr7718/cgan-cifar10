"""Metric aggregation, history logging, and summary reporting.

This module provides utilities for tracking cGAN training and evaluation
metrics across epochs and exporting the results to JSON or a human-readable
summary report.
"""

import json
import os
from typing import Dict, List, Optional


class MetricTracker:
    """Track training and evaluation metrics across epochs."""

    def __init__(self) -> None:
        self.history: Dict[str, List[float]] = {}

    def update(self, metrics: Dict[str, float]) -> None:
        """Add a set of metrics for the current epoch.

        Args:
            metrics: Dictionary mapping metric names to numeric values.
        """
        for name, value in metrics.items():
            if name not in self.history:
                self.history[name] = []

            self.history[name].append(float(value))

    def add(self, name: str, value: float) -> None:
        """Add a single metric value."""
        if name not in self.history:
            self.history[name] = []

        self.history[name].append(float(value))

    def get_history(self) -> Dict[str, List[float]]:
        """Return the complete metric history."""
        return self.history

    def latest(self, name: str) -> Optional[float]:
        """Return the most recent value for a metric."""
        values = self.history.get(name)

        if not values:
            return None

        return values[-1]

    def epoch_count(self) -> int:
        """Return the number of recorded epochs."""
        if not self.history:
            return 0

        return max(len(values) for values in self.history.values())

    def reset(self) -> None:
        """Clear all recorded metrics."""
        self.history.clear()


def export_metrics_to_json(
    metrics_dict: Dict,
    output_path: str,
) -> None:
    """Export metrics to a JSON file.

    Args:
        metrics_dict: Metrics dictionary to save.
        output_path: Destination JSON path.
    """
    directory = os.path.dirname(output_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metrics_dict, file, indent=2)


def generate_summary_report(
    metrics_history: Dict[str, List[float]],
    output_path: str,
) -> None:
    """Generate a human-readable metrics summary report.

    Args:
        metrics_history: Dictionary containing metric histories.
        output_path: Destination text file.
    """
    directory = os.path.dirname(output_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    lines = [
        "cGAN Evaluation Summary",
        "=" * 50,
        "",
    ]

    if not metrics_history:
        lines.append("No metrics recorded.")
    else:
        for name, values in metrics_history.items():
            if not values:
                continue

            latest_value = values[-1]
            best_value = min(values)
            worst_value = max(values)

            lines.extend(
                [
                    f"Metric: {name}",
                    f"  Epochs recorded: {len(values)}",
                    f"  Latest: {latest_value:.6f}",
                    f"  Minimum: {best_value:.6f}",
                    f"  Maximum: {worst_value:.6f}",
                    "",
                ]
            )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))