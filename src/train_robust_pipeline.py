"""Advanced, leakage-resistant real-estate modelling and anomaly screening.

This module keeps the conference baseline in ``train_xgboost.py`` intact and
adds a stricter research workflow:

1. Compare median and linear baselines, Random Forest, histogram gradient
   boosting and XGBoost using expanding-window validation.
2. Select a model using only years before the final holdout year.
3. Evaluate the selected model once on the untouched final year.
4. Measure seed, subgroup and bootstrap uncertainty.
5. Rank transactions with genuinely out-of-time residuals.

The anomaly score is a screening statistic, not evidence that a transaction
is illegal, fraudulent or incorrectly reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from matplotlib.ticker import FuncFormatter
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder
from xgboost import XGBRegressor


REQUIRED_COLUMNS = {
    "시군구": "시군구",
    "법정동": "법정동",
    "단지명": "단지명",
    "전용면적(㎡)": "전용면적",
    "계약년월": "계약년월",
    "거래금액(만원)": "거래금액",
    "층": "층",
    "건축년도": "건축년도",
}

NUMERIC_FEATURES = [
    "전용면적",
    "층",
    "건축년도",
    "건물나이",
    "계약연도",
    "계약월",
]
CATEGORICAL_FEATURES = ["자치구", "법정동", "단지명"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MODEL_NAMES = (
    "median_baseline",
    "ridge",
    "random_forest",
    "hist_gradient_boosting",
    "xgboost",
)
DEFAULT_SEEDS = (13, 42, 77)


@dataclass(frozen=True)
class TemporalFold:
    validation_year: int
    train_ids: np.ndarray
    validation_ids: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "모델 비교부터 시간 기반 강건성 검증까지 수행하는 발전 버전"
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="서울실거래가_*.csv가 있는 디렉터리",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/advanced/generated"),
        help="발전 버전 결과 저장 디렉터리",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
        help="시간 기반 모델 선정에서 비교할 후보",
    )
    parser.add_argument("--top-n", type=int, default=922)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--primary-seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=300)
    parser.add_argument("--min-group-size", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--sample-per-year",
        type=int,
        default=None,
        help=(
            "개발용 층화 표본 크기. 논문 결과 생성 시에는 사용하지 마세요."
        ),
    )
    return parser.parse_args()


def read_csv_file(path: Path) -> pd.DataFrame:
    """Read common MOLIT CSV export variants without silently losing headers."""

    attempts = (
        {"encoding": "cp949", "skiprows": 15},
        {"encoding": "cp949"},
        {"encoding": "utf-8-sig", "skiprows": 15},
        {"encoding": "utf-8-sig"},
    )
    last_error: Exception | None = None
    for options in attempts:
        try:
            frame = pd.read_csv(path, **options)
            if set(REQUIRED_COLUMNS).issubset(frame.columns):
                return frame
        except Exception as error:  # pragma: no cover - depends on input encoding
            last_error = error
    raise ValueError(f"지원하지 않는 CSV 형식입니다: {path}") from last_error


def load_and_prepare(data_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    """Load raw files and retain raw categorical values for fold-safe encoding."""

    files = sorted(data_dir.glob("서울실거래가_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"{data_dir}에서 서울실거래가_*.csv 파일을 찾을 수 없습니다."
        )

    raw = pd.concat([read_csv_file(path) for path in files], ignore_index=True)
    missing = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    frame = raw[list(REQUIRED_COLUMNS)].rename(columns=REQUIRED_COLUMNS).copy()
    frame = frame[frame["시군구"].astype(str).str.contains("서울", na=False)].copy()
    frame["거래금액"] = pd.to_numeric(
        frame["거래금액"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    for column in ["전용면적", "층", "건축년도", "계약년월"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["계약연도"] = (frame["계약년월"] // 100).astype("Int64")
    frame["계약월"] = (frame["계약년월"] % 100).astype("Int64")
    split_location = frame["시군구"].astype(str).str.split()
    frame["자치구"] = split_location.str[1]
    frame["법정동"] = frame["법정동"].astype("string").str.strip()
    frame["단지명"] = frame["단지명"].astype("string").str.strip()
    frame["건물나이"] = frame["계약연도"] - frame["건축년도"]

    required = ["거래금액", *FEATURES]
    frame = frame.dropna(subset=required)
    frame = frame[
        (frame["거래금액"] > 0)
        & (frame["전용면적"] > 0)
        & (frame["건물나이"] >= 0)
        & frame["계약월"].between(1, 12)
    ].copy()
    frame["계약연도"] = frame["계약연도"].astype(int)
    frame["계약월"] = frame["계약월"].astype(int)
    frame = frame.sort_values(["계약연도", "계약월"]).reset_index(drop=True)
    frame.insert(0, "record_id", np.arange(len(frame), dtype=np.int64))
    return frame, files


def stratified_sample_by_year(
    frame: pd.DataFrame, sample_per_year: int | None, seed: int
) -> pd.DataFrame:
    if sample_per_year is None:
        return frame
    pieces = [
        group.sample(n=min(len(group), sample_per_year), random_state=seed)
        for _, group in frame.groupby("계약연도", sort=True)
    ]
    sampled = pd.concat(pieces, ignore_index=True)
    sampled = sampled.sort_values(["계약연도", "계약월"]).reset_index(drop=True)
    sampled["record_id"] = np.arange(len(sampled), dtype=np.int64)
    return sampled


def make_preprocessor(seed: int) -> ColumnTransformer:
    """Fit category mappings inside each temporal fold and handle unseen values."""

    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    version_match = re.match(r"(\d+)\.(\d+)", sklearn.__version__)
    sklearn_version = (
        tuple(int(part) for part in version_match.groups())
        if version_match
        else (1, 4)
    )
    if sklearn_version >= (1, 9):
        target_encoder = TargetEncoder(
            target_type="continuous",
            smooth="auto",
            cv=KFold(n_splits=5, shuffle=True, random_state=seed),
        )
    else:
        target_encoder = TargetEncoder(
            target_type="continuous",
            smooth="auto",
            cv=5,
            shuffle=True,
            random_state=seed,
        )
    categorical = Pipeline(
        [
            (
                "encoder",
                target_encoder,
            )
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def build_estimator(name: str, seed: int, n_jobs: int) -> Pipeline:
    """Create comparable candidates with a shared, fold-safe preprocessor."""

    if name == "median_baseline":
        regressor = DummyRegressor(strategy="median")
    elif name == "ridge":
        regressor = Ridge(alpha=10.0)
    elif name == "random_forest":
        regressor = RandomForestRegressor(
            n_estimators=180,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=seed,
            n_jobs=n_jobs,
        )
    elif name == "hist_gradient_boosting":
        regressor = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=seed,
        )
    elif name == "xgboost":
        regressor = XGBRegressor(
            n_estimators=450,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            n_jobs=n_jobs,
        )
    else:  # pragma: no cover - argparse and tests constrain this value
        raise ValueError(f"알 수 없는 모델입니다: {name}")
    steps: list[tuple[str, object]] = [("preprocess", make_preprocessor(seed))]
    if name == "ridge":
        steps.append(("scale", StandardScaler()))
    steps.append(("model", regressor))
    return Pipeline(steps)


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    residual = actual - predicted
    return {
        "r2": float(r2_score(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "median_ae": float(np.median(np.abs(residual))),
        "wape": float(np.abs(residual).sum() / np.abs(actual).sum()),
    }


def make_temporal_folds(frame: pd.DataFrame) -> tuple[list[TemporalFold], int]:
    """Reserve the latest year and build expanding folds before it."""

    years = sorted(int(year) for year in frame["계약연도"].unique())
    if len(years) < 3:
        raise ValueError(
            "최소 3개 연도가 필요합니다: 초기 학습연도, 모델 선정연도, "
            "최종 평가연도."
        )
    final_year = years[-1]
    folds: list[TemporalFold] = []
    for validation_year in years[1:-1]:
        train_ids = frame.index[frame["계약연도"] < validation_year].to_numpy()
        validation_ids = frame.index[
            frame["계약연도"] == validation_year
        ].to_numpy()
        if len(train_ids) and len(validation_ids):
            folds.append(TemporalFold(validation_year, train_ids, validation_ids))
    if not folds:
        raise ValueError(
            "최종 연도 이전에 모델 선정용 시간 폴드를 만들 수 없습니다."
        )
    return folds, final_year


def compare_models(
    frame: pd.DataFrame,
    folds: list[TemporalFold],
    model_names: list[str],
    seed: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, int], pd.DataFrame]]:
    rows: list[dict[str, float | int | str]] = []
    prediction_cache: dict[tuple[str, int], pd.DataFrame] = {}
    for model_name in model_names:
        for fold in folds:
            train = frame.loc[fold.train_ids]
            validation = frame.loc[fold.validation_ids]
            estimator = build_estimator(model_name, seed, n_jobs)
            estimator.fit(train[FEATURES], train["거래금액"])
            prediction = estimator.predict(validation[FEATURES])
            metric = regression_metrics(validation["거래금액"], prediction)
            rows.append(
                {
                    "model": model_name,
                    "validation_year": fold.validation_year,
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    **metric,
                }
            )
            prediction_cache[(model_name, fold.validation_year)] = pd.DataFrame(
                {
                    "record_id": validation["record_id"].to_numpy(),
                    "validation_year": fold.validation_year,
                    "prediction": prediction,
                }
            )

    fold_metrics = pd.DataFrame(rows)
    comparison = (
        fold_metrics.groupby("model", as_index=False)
        .agg(
            folds=("validation_year", "count"),
            mean_r2=("r2", "mean"),
            std_r2=("r2", "std"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            worst_year_rmse=("rmse", "max"),
            mean_wape=("wape", "mean"),
        )
        .sort_values(["mean_rmse", "mean_mae", "model"])
        .reset_index(drop=True)
    )
    comparison.insert(0, "selection_rank", np.arange(1, len(comparison) + 1))
    return fold_metrics, comparison, prediction_cache


def evaluate_final_holdout(
    frame: pd.DataFrame,
    final_year: int,
    selected_model: str,
    seed: int,
    n_jobs: int,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    train = frame[frame["계약연도"] < final_year]
    test = frame[frame["계약연도"] == final_year]
    estimator = build_estimator(selected_model, seed, n_jobs)
    estimator.fit(train[FEATURES], train["거래금액"])
    prediction = estimator.predict(test[FEATURES])
    selected_metrics = {
        "model": selected_model,
        "role": "selected_model",
        "train_end_year": final_year - 1,
        "test_year": final_year,
        "train_rows": len(train),
        "test_rows": len(test),
        **regression_metrics(test["거래금액"], prediction),
    }

    baseline = build_estimator("median_baseline", seed, n_jobs)
    baseline.fit(train[FEATURES], train["거래금액"])
    baseline_prediction = baseline.predict(test[FEATURES])
    baseline_metrics = {
        "model": "median_baseline",
        "role": "reference_only_not_for_reselection",
        "train_end_year": final_year - 1,
        "test_year": final_year,
        "train_rows": len(train),
        "test_rows": len(test),
        **regression_metrics(test["거래금액"], baseline_prediction),
    }
    predictions = pd.DataFrame(
        {
            "record_id": test["record_id"].to_numpy(),
            "validation_year": final_year,
            "prediction": prediction,
        }
    )
    return estimator, pd.DataFrame([selected_metrics, baseline_metrics]), predictions


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    """Non-parametric 95% intervals for final-year metrics."""

    if iterations <= 0:
        return pd.DataFrame(columns=["metric", "estimate", "ci_2.5", "ci_97.5"])
    rng = np.random.default_rng(seed)
    n_rows = len(y_true)
    values: dict[str, list[float]] = {name: [] for name in ("r2", "mae", "rmse")}
    for _ in range(iterations):
        sample = rng.integers(0, n_rows, size=n_rows)
        metric = regression_metrics(y_true[sample], y_pred[sample])
        for name in values:
            values[name].append(metric[name])
    point = regression_metrics(y_true, y_pred)
    return pd.DataFrame(
        [
            {
                "metric": name,
                "estimate": point[name],
                "ci_2.5": float(np.percentile(samples, 2.5)),
                "ci_97.5": float(np.percentile(samples, 97.5)),
                "bootstrap_iterations": iterations,
            }
            for name, samples in values.items()
        ]
    )


def add_prediction_errors(scored: pd.DataFrame) -> pd.DataFrame:
    scored = scored.copy()
    scored["예측가격"] = np.maximum(scored["prediction"], 1.0)
    scored["잔차"] = scored["거래금액"] - scored["예측가격"]
    scored["절대잔차"] = scored["잔차"].abs()
    scored["오차율"] = scored["절대잔차"] / scored["거래금액"].clip(lower=1.0)
    scored["로그잔차"] = np.log(scored["거래금액"].clip(lower=1.0)) - np.log(
        scored["예측가격"]
    )

    def robust_z(group: pd.Series) -> pd.Series:
        median = group.median()
        mad = (group - median).abs().median()
        scale = max(1.4826 * mad, 1e-9)
        return ((group - median) / scale).abs()

    scored["강건성이상점수"] = scored.groupby("validation_year")["로그잔차"].transform(
        robust_z
    )
    return scored


def build_out_of_time_scores(
    frame: pd.DataFrame,
    selected_model: str,
    folds: list[TemporalFold],
    cache: dict[tuple[str, int], pd.DataFrame],
    final_prediction: pd.DataFrame,
) -> pd.DataFrame:
    pieces = [cache[(selected_model, fold.validation_year)] for fold in folds]
    pieces.append(final_prediction)
    predictions = pd.concat(pieces, ignore_index=True)
    scored = predictions.merge(frame, on="record_id", how="left", validate="one_to_one")
    return add_prediction_errors(scored)


def subgroup_metrics(scored: pd.DataFrame, min_group_size: int) -> pd.DataFrame:
    """Report final-year errors for geographic, area and target-price slices."""

    test = scored.copy()
    test["전용면적구간"] = pd.cut(
        test["전용면적"],
        bins=[0, 40, 60, 85, 135, np.inf],
        labels=["40㎡ 이하", "40–60㎡", "60–85㎡", "85–135㎡", "135㎡ 초과"],
    )
    try:
        test["가격구간"] = pd.qcut(
            test["거래금액"],
            q=4,
            labels=["Q1", "Q2", "Q3", "Q4"],
            duplicates="drop",
        )
    except ValueError:
        test["가격구간"] = "전체"

    rows: list[dict[str, float | int | str]] = []
    for group_type, column in (
        ("자치구", "자치구"),
        ("전용면적구간", "전용면적구간"),
        ("가격구간", "가격구간"),
    ):
        for group_value, group in test.groupby(column, observed=True):
            if len(group) < min_group_size:
                continue
            rows.append(
                {
                    "group_type": group_type,
                    "group_value": str(group_value),
                    "rows": len(group),
                    **regression_metrics(group["거래금액"], group["예측가격"]),
                }
            )
    return pd.DataFrame(rows)


def score_for_ranking(scored: pd.DataFrame) -> pd.Series:
    log_residual = np.log(scored["거래금액"].clip(lower=1.0)) - np.log(
        scored["예측가격"].clip(lower=1.0)
    )
    median = log_residual.median()
    mad = (log_residual - median).abs().median()
    return ((log_residual - median) / max(1.4826 * mad, 1e-9)).abs()


def seed_sensitivity(
    frame: pd.DataFrame,
    final_year: int,
    model_name: str,
    seeds: list[int],
    primary_seed: int,
    top_n: int,
    n_jobs: int,
    primary_prediction: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame[frame["계약연도"] < final_year]
    test = frame[frame["계약연도"] == final_year]
    predictions: dict[int, np.ndarray] = {
        primary_seed: primary_prediction["prediction"].to_numpy()
    }
    rows: list[dict[str, float | int | str]] = []
    ordered_seeds = list(dict.fromkeys([primary_seed, *seeds]))
    for seed in ordered_seeds:
        if seed not in predictions:
            estimator = build_estimator(model_name, seed, n_jobs)
            estimator.fit(train[FEATURES], train["거래금액"])
            predictions[seed] = estimator.predict(test[FEATURES])
        rows.append(
            {
                "model": model_name,
                "seed": seed,
                "test_year": final_year,
                **regression_metrics(test["거래금액"], predictions[seed]),
            }
        )

    candidate_count = min(max(top_n, 1), len(test))
    top_sets: dict[int, set[int]] = {}
    for seed, prediction in predictions.items():
        ranked = test[["record_id", "거래금액"]].copy()
        ranked["예측가격"] = np.maximum(prediction, 1.0)
        ranked["score"] = score_for_ranking(ranked)
        top_sets[seed] = set(
            ranked.nlargest(candidate_count, "score")["record_id"].astype(int)
        )
    reference = top_sets[primary_seed]
    stability = []
    for seed, candidates in top_sets.items():
        union = reference | candidates
        stability.append(
            {
                "reference_seed": primary_seed,
                "comparison_seed": seed,
                "top_n": candidate_count,
                "intersection": len(reference & candidates),
                "jaccard": len(reference & candidates) / len(union) if union else 1.0,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(stability)


def threshold_sensitivity(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for percentage in (0.1, 0.5, 1.0, 2.0):
        count = max(1, int(np.ceil(len(scored) * percentage / 100)))
        selected = scored.nlargest(count, "강건성이상점수")
        rows.append(
            {
                "top_percentage": percentage,
                "candidate_count": count,
                "minimum_score": float(selected["강건성이상점수"].min()),
                "median_absolute_residual": float(selected["절대잔차"].median()),
                "median_error_rate": float(selected["오차율"].median()),
            }
        )
    return pd.DataFrame(rows)


def feature_importance_table(
    estimator: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int,
) -> pd.DataFrame:
    """Use model-agnostic holdout permutation importance for fair reporting."""

    result = permutation_importance(
        estimator,
        X_test,
        y_test,
        scoring="neg_root_mean_squared_error",
        n_repeats=3,
        random_state=seed,
        n_jobs=1,
        max_samples=min(10_000, len(X_test)),
    )
    return pd.DataFrame(
        {
            "feature": FEATURES,
            "importance_mean_rmse_increase": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean_rmse_increase", ascending=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_korean_font() -> None:
    installed = {font.name for font in fm.fontManager.ttflist}
    for font_name in ("AppleGothic", "Malgun Gothic", "NanumGothic"):
        if font_name in installed:
            plt.rc("font", family=font_name)
            break
    plt.rcParams["axes.unicode_minus"] = False


def price_in_eok(value: float, _position: int) -> str:
    return f"{value / 10_000:.1f}"


def save_figures(
    comparison: pd.DataFrame,
    final_scored: pd.DataFrame,
    output_dir: Path,
) -> None:
    configure_korean_font()
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    plot = comparison.sort_values("mean_rmse")
    plt.figure(figsize=(9, 5))
    plt.barh(plot["model"], plot["mean_rmse"], color="#2F6FA3")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(price_in_eok))
    plt.xlabel("시간 검증 평균 RMSE (억원)")
    plt.title("모델 선정 결과: 낮을수록 우수")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_dir / "model_comparison_temporal_rmse.png", dpi=250)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.scatter(
        final_scored["거래금액"],
        final_scored["예측가격"],
        s=12,
        alpha=0.25,
        color="#4C78A8",
    )
    lower = min(final_scored["거래금액"].min(), final_scored["예측가격"].min())
    upper = max(final_scored["거래금액"].max(), final_scored["예측가격"].max())
    plt.plot([lower, upper], [lower, upper], "--", color="#E45756")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(price_in_eok))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(price_in_eok))
    plt.xlabel("실제 가격 (억원)")
    plt.ylabel("예측 가격 (억원)")
    plt.title("최종 연도 Out-of-Time 평가")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_dir / "final_holdout_actual_vs_predicted.png", dpi=250)
    plt.close()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def run(
    data_dir: Path,
    output_dir: Path,
    model_names: list[str],
    top_n: int,
    seeds: list[int],
    primary_seed: int,
    bootstrap_iterations: int,
    min_group_size: int,
    n_jobs: int,
    sample_per_year: int | None = None,
) -> dict[str, object]:
    frame, input_files = load_and_prepare(data_dir)
    frame = stratified_sample_by_year(frame, sample_per_year, primary_seed)
    folds, final_year = make_temporal_folds(frame)

    fold_metrics, comparison, cache = compare_models(
        frame, folds, model_names, primary_seed, n_jobs
    )
    selected_model = str(comparison.iloc[0]["model"])
    final_estimator, final_metrics, final_prediction = evaluate_final_holdout(
        frame, final_year, selected_model, primary_seed, n_jobs
    )

    out_of_time = build_out_of_time_scores(
        frame, selected_model, folds, cache, final_prediction
    )
    final_scored = out_of_time[out_of_time["validation_year"] == final_year].copy()
    bootstrap = bootstrap_confidence_intervals(
        final_scored["거래금액"].to_numpy(),
        final_scored["예측가격"].to_numpy(),
        bootstrap_iterations,
        primary_seed,
    )
    groups = subgroup_metrics(final_scored, min_group_size)
    seed_metrics, seed_stability = seed_sensitivity(
        frame,
        final_year,
        selected_model,
        seeds,
        primary_seed,
        top_n,
        n_jobs,
        final_prediction,
    )
    thresholds = threshold_sensitivity(out_of_time)
    importance = feature_importance_table(
        final_estimator,
        final_scored[FEATURES],
        final_scored["거래금액"],
        primary_seed,
    )

    candidate_count = min(max(top_n, 1), len(out_of_time))
    candidates = out_of_time.nlargest(candidate_count, "강건성이상점수").copy()
    candidates.insert(0, "강건성이상순위", np.arange(1, len(candidates) + 1))
    candidate_columns = [
        "강건성이상순위",
        "record_id",
        "시군구",
        "자치구",
        "법정동",
        "단지명",
        "전용면적",
        "층",
        "건축년도",
        "계약연도",
        "계약월",
        "거래금액",
        "예측가격",
        "잔차",
        "절대잔차",
        "오차율",
        "강건성이상점수",
        "validation_year",
    ]

    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    write_csv(fold_metrics, table_dir / "temporal_fold_metrics.csv")
    write_csv(comparison, table_dir / "model_comparison.csv")
    write_csv(final_metrics, table_dir / "final_holdout_metrics.csv")
    write_csv(bootstrap, table_dir / "bootstrap_confidence_intervals.csv")
    write_csv(groups, table_dir / "subgroup_metrics.csv")
    write_csv(seed_metrics, table_dir / "seed_sensitivity.csv")
    write_csv(seed_stability, table_dir / "seed_candidate_stability.csv")
    write_csv(thresholds, table_dir / "threshold_sensitivity.csv")
    write_csv(importance, table_dir / "permutation_importance.csv")
    write_csv(candidates[candidate_columns], table_dir / "anomaly_candidates.csv")
    write_csv(
        out_of_time[
            [
                "record_id",
                "validation_year",
                "거래금액",
                "예측가격",
                "잔차",
                "절대잔차",
                "오차율",
                "강건성이상점수",
            ]
        ],
        table_dir / "out_of_time_predictions.csv",
    )
    save_figures(comparison, final_scored, output_dir)

    selected_final = final_metrics[final_metrics["role"] == "selected_model"].iloc[0]
    summary: dict[str, object] = {
        "status": "completed",
        "methodology": {
            "selection": "expanding-window validation before final holdout",
            "selection_metric": "mean RMSE",
            "final_holdout_year": final_year,
            "anomaly_prediction": "out-of-time prediction; no row predicts itself",
            "anomaly_score": "year-normalized robust z-score of log-price residual",
            "caution": "screening candidate only; not a fraud determination",
        },
        "data": {
            "transactions": len(frame),
            "years": sorted(int(year) for year in frame["계약연도"].unique()),
            "development_sample_per_year": sample_per_year,
            "input_files": [
                {"path": str(path), "sha256": sha256(path)} for path in input_files
            ],
        },
        "model_selection": {
            "candidates": model_names,
            "selected_model": selected_model,
            "selection_mean_rmse": float(comparison.iloc[0]["mean_rmse"]),
        },
        "final_holdout": {
            key: float(selected_final[key]) for key in ("r2", "mae", "rmse", "wape")
        },
        "reproducibility": {
            "primary_seed": primary_seed,
            "sensitivity_seeds": list(dict.fromkeys([primary_seed, *seeds])),
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"발전 버전 결과를 {output_dir}에 저장했습니다.")
    return summary


if __name__ == "__main__":
    args = parse_args()
    run(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_names=args.models,
        top_n=args.top_n,
        seeds=args.seeds,
        primary_seed=args.primary_seed,
        bootstrap_iterations=args.bootstrap_iterations,
        min_group_size=args.min_group_size,
        n_jobs=args.n_jobs,
        sample_per_year=args.sample_per_year,
    )
