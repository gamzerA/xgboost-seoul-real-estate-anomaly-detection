"""Small regression tests for the advanced temporal validation pipeline."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).parents[1] / "src" / "train_robust_pipeline.py"
SPEC = importlib.util.spec_from_file_location("train_robust_pipeline", MODULE_PATH)
assert SPEC and SPEC.loader
PIPELINE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PIPELINE
SPEC.loader.exec_module(PIPELINE)


def synthetic_frame(rows_per_year: int = 30) -> pd.DataFrame:
    rows = []
    for year in (2022, 2023, 2024, 2025):
        for index in range(rows_per_year):
            area = 45.0 + index
            rows.append(
                {
                    "시군구": "서울특별시 강남구",
                    "자치구": "강남구" if index % 2 else "마포구",
                    "법정동": f"동{index % 5}",
                    "단지명": f"단지{index % 9}" if year < 2025 else f"신규{index % 9}",
                    "전용면적": area,
                    "층": index % 20 + 1,
                    "건축년도": 2000 + index % 20,
                    "건물나이": year - (2000 + index % 20),
                    "계약연도": year,
                    "계약월": index % 12 + 1,
                    "거래금액": 30_000 + area * 1_000 + (year - 2022) * 2_000,
                }
            )
    frame = pd.DataFrame(rows).reset_index(drop=True)
    frame.insert(0, "record_id", np.arange(len(frame), dtype=np.int64))
    return frame


class RobustPipelineTest(unittest.TestCase):
    def test_latest_year_is_reserved_from_model_selection(self) -> None:
        frame = synthetic_frame()
        folds, final_year = PIPELINE.make_temporal_folds(frame)
        self.assertEqual(final_year, 2025)
        self.assertEqual([fold.validation_year for fold in folds], [2023, 2024])
        for fold in folds:
            self.assertTrue(
                (frame.loc[fold.train_ids, "계약연도"] < fold.validation_year).all()
            )

    def test_unseen_future_categories_are_supported(self) -> None:
        frame = synthetic_frame()
        train = frame[frame["계약연도"] < 2025]
        future = frame[frame["계약연도"] == 2025]
        estimator = PIPELINE.build_estimator("median_baseline", seed=42, n_jobs=1)
        estimator.fit(train[PIPELINE.FEATURES], train["거래금액"])
        prediction = estimator.predict(future[PIPELINE.FEATURES])
        self.assertEqual(len(prediction), len(future))
        self.assertTrue(np.isfinite(prediction).all())

    def test_robust_scores_are_finite(self) -> None:
        frame = synthetic_frame()
        scored = frame.copy()
        scored["validation_year"] = scored["계약연도"]
        scored["prediction"] = scored["거래금액"] * np.linspace(0.8, 1.2, len(scored))
        result = PIPELINE.add_prediction_errors(scored)
        self.assertTrue(np.isfinite(result["강건성이상점수"]).all())

    def test_development_sampling_rebuilds_unique_ids(self) -> None:
        sampled = PIPELINE.stratified_sample_by_year(
            synthetic_frame(), sample_per_year=10, seed=42
        )
        self.assertEqual(len(sampled), 40)
        self.assertTrue(sampled["record_id"].is_unique)


if __name__ == "__main__":
    unittest.main()
