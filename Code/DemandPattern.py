# Analyze Demand Patterns

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import wntr

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def analyze_demand_patterns(inp_file: str | os.PathLike, output_dir: str | os.PathLike = "networks/demand_scenarios") -> Dict[str, Dict[str, float]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Reading network file: %s", inp_file)
    wn_original = wntr.network.WaterNetworkModel(str(inp_file))

    pattern_stats: Dict[str, Dict[str, float]] = {}
    for pattern_name, pattern in wn_original.patterns():
        multipliers = np.asarray(pattern.multipliers, dtype=float)
        pattern_stats[pattern_name] = {
            "min": float(np.min(multipliers)),
            "max": float(np.max(multipliers)),
            "mean": float(np.mean(multipliers)),
        }

    if pattern_stats:
        pattern_df = pd.DataFrame(pattern_stats).T
        logger.info("Pattern statistics:\n%s", pattern_df)

    scenarios = ["min", "max", "mean"]

    for scenario in scenarios:
        modified_inp = output_path / f"C-Town_{scenario}_demand.inp"
        logger.info("Creating %s demand scenario: %s", scenario, modified_inp)

        with open(inp_file, "r") as f:
            content = f.readlines()

        pattern_section = False
        modified_content = []

        for line in content:
            if line.strip() == "[PATTERNS]":
                pattern_section = True
                modified_content.append(line)
                continue

            if pattern_section:
                if line.strip().startswith("["):
                    pattern_section = False
                    modified_content.append(line)
                    continue

                parts = line.strip().split()
                if parts and parts[0] in pattern_stats:
                    pattern_name = parts[0]
                    target_value = pattern_stats[pattern_name][scenario]
                    num_values = len(parts) - 1
                    new_line = f"{pattern_name}\t" + "\t".join([f"{target_value:.6f}" for _ in range(num_values)]) + "\n"
                    modified_content.append(new_line)
                else:
                    modified_content.append(line)
            else:
                modified_content.append(line)

        with open(modified_inp, "w") as f:
            f.writelines(modified_content)

    logger.info("Finished writing demand scenarios to %s", output_path)
    return pattern_stats


if __name__ == "__main__":
    patterns_stats = analyze_demand_patterns(
        inp_file="networks/C-Town.inp",
        output_dir="networks/demand_scenarios",
    )