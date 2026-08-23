"""Train an XGBoost price model and rank residual-based review candidates.

The candidate ranking is a screening aid. It does not determine that a
transaction is illegal or fraudulent.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
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

FEATURES = [
    "전용면적",
    "층",
    "건축년도",
    "건물나이",
    "계약연도",
    "계약월",
    "자치구_enc",
    "법정동_enc",
    "단지명_enc",
]

FEATURE_LABELS = {
    "전용면적": "전용면적",
    "층": "층",
    "건축년도": "건축년도",
    "건물나이": "건물나이",
    "계약연도": "계약연도",
    "계약월": "계약월",
    "자치구_enc": "자치구",
    "법정동_enc": "법정동",
    "단지명_enc": "단지명",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="서울시 아파트 가격 예측 및 잔차 기반 이상 거래 후보 탐지"
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
        default=Path("results/generated"),
        help="표와 그래프를 저장할 디렉터리",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=922,
        help="절대잔차 기준으로 저장할 검토 후보 수",
    )
    return parser.parse_args()


def read_csv_file(path: Path) -> pd.DataFrame:
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
        except Exception as error:  # Try the next known export format.
            last_error = error
    raise ValueError(f"지원하지 않는 CSV 형식입니다: {path}") from last_error


def load_and_prepare(data_dir: Path) -> pd.DataFrame:
    files = [Path(path) for path in sorted(glob.glob(str(data_dir / "서울실거래가_*.csv")))]
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

    frame["계약연도"] = frame["계약년월"] // 100
    frame["계약월"] = frame["계약년월"] % 100
    frame["자치구"] = frame["시군구"].astype(str).str.split().str[1]
    frame["법정동"] = frame["법정동"].astype(str).str.strip()
    frame["단지명"] = frame["단지명"].astype(str).str.strip()
    frame["건물나이"] = frame["계약연도"] - frame["건축년도"]

    use_columns = [
        "거래금액",
        "전용면적",
        "층",
        "건축년도",
        "건물나이",
        "계약연도",
        "계약월",
        "자치구",
        "법정동",
        "단지명",
    ]
    frame = frame.dropna(subset=use_columns)
    frame = frame[
        (frame["거래금액"] > 0)
        & (frame["전용면적"] > 0)
        & (frame["건물나이"] >= 0)
        & frame["계약연도"].between(2022, 2025)
    ].copy()

    for column in ["자치구", "법정동", "단지명"]:
        encoder = LabelEncoder()
        frame[f"{column}_enc"] = encoder.fit_transform(frame[column].astype(str))

    return frame


def build_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )


def configure_korean_font() -> None:
    candidates = ("Malgun Gothic", "AppleGothic", "NanumGothic")
    installed = {font.name for font in fm.fontManager.ttflist}
    for font_name in candidates:
        if font_name in installed:
            plt.rc("font", family=font_name)
            break
    plt.rcParams["axes.unicode_minus"] = False


def price_in_eok(value: float, _position: int) -> str:
    return f"{value / 10_000:.1f}"


def save_figures(
    frame: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    importance: pd.DataFrame,
    output_dir: Path,
) -> None:
    configure_korean_font()
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.35, s=30, color="#4C78A8")
    lower = min(y_test.min(), y_pred.min())
    upper = max(y_test.max(), y_pred.max())
    plt.plot([lower, upper], [lower, upper], color="#E45756", linestyle="--")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(price_in_eok))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(price_in_eok))
    plt.title("실제 거래가격과 예측가격 비교")
    plt.xlabel("실제 거래가격 (억원)")
    plt.ylabel("예측 거래가격 (억원)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_dir / "actual_vs_predicted.png", dpi=300)
    plt.close()

    plot_frame = importance.sort_values("중요도")
    plt.figure(figsize=(8, 6))
    plt.barh(plot_frame["변수"], plot_frame["중요도"], color="#2F6FA3")
    plt.title("변수 중요도")
    plt.xlabel("중요도")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_dir / "feature_importance.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(frame["잔차"], bins=60, color="#72B7B2", edgecolor="white")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(price_in_eok))
    plt.title("잔차 분포")
    plt.xlabel("잔차 (억원)")
    plt.ylabel("거래 건수")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_dir / "residual_distribution.png", dpi=300)
    plt.close()


def run(data_dir: Path, output_dir: Path, top_n: int) -> None:
    frame = load_and_prepare(data_dir)
    X = frame[FEATURES]
    y = frame["거래금액"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "transactions": int(len(frame)),
        "test_r2": float(r2_score(y_test, y_pred)),
        "test_mae": float(mean_absolute_error(y_test, y_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(build_model(), X, y, cv=cv, scoring="r2")
    metrics["cv_r2_mean"] = float(cv_r2.mean())
    metrics["cv_r2_std"] = float(cv_r2.std())

    frame["예측가격"] = model.predict(X)
    frame["잔차"] = frame["거래금액"] - frame["예측가격"]
    frame["절대잔차"] = frame["잔차"].abs()
    frame["오차율"] = frame["절대잔차"] / frame["거래금액"]
    candidates = frame.nlargest(top_n, "절대잔차").copy()

    importance = pd.DataFrame(
        {
            "변수": [FEATURE_LABELS[name] for name in FEATURES],
            "중요도": model.feature_importances_,
        }
    ).sort_values("중요도", ascending=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "model_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    importance.to_csv(
        table_dir / "feature_importance.csv", index=False, encoding="utf-8-sig"
    )

    result_columns = [
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
    ]
    candidates[result_columns].to_csv(
        table_dir / "anomaly_candidates.csv", index=False, encoding="utf-8-sig"
    )
    save_figures(frame, y_test, y_pred, importance, output_dir)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"검토 후보 {len(candidates):,}건과 결과물을 {output_dir}에 저장했습니다.")
    print("주의: 검토 후보는 불법·부정 거래를 확정하는 결과가 아닙니다.")


if __name__ == "__main__":
    args = parse_args()
    run(args.data_dir, args.output_dir, args.top_n)
