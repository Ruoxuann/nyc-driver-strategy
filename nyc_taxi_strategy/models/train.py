"""Train wait-time and fare prediction models and save to disk."""

import logging
import pickle
from pathlib import Path


logger = logging.getLogger(__name__)


def train_models(cfg) -> tuple:
    from nyc_taxi_strategy.data.etl import query_demand
    from nyc_taxi_strategy.features.transformers import (
        CyclicTimeEncoder, LagFeatureBuilder, RollingStatsBuilder,
        HolidayFeature, FeaturePipeline,
    )
    from nyc_taxi_strategy.models.predictors import create_model

    df = query_demand(cfg.data.db_path)
    logger.info("Loaded %d zone-hour rows", len(df))

    df["wait_time_s"] = 3600.0 / df["trip_count"].clip(lower=1)

    feat_cfg = cfg.features
    transformers = [CyclicTimeEncoder()]
    if feat_cfg.lag_hours:
        transformers.append(LagFeatureBuilder(lag_hours=feat_cfg.lag_hours))
    if feat_cfg.rolling_windows:
        transformers.append(RollingStatsBuilder(windows=feat_cfg.rolling_windows))
    if feat_cfg.use_holidays:
        transformers.append(HolidayFeature())

    pipeline = FeaturePipeline(transformers)
    df_feat = pipeline.fit_transform(df).dropna()

    exclude = {"pickup_zone", "hour_start", "trip_count", "avg_fare",
               "median_fare", "avg_duration_s", "avg_distance", "wait_time_s"}
    feature_cols = [c for c in df_feat.columns if c not in exclude]
    X = df_feat[feature_cols]

    wt_cfg = cfg.model.wait_time
    wait_model = create_model(wt_cfg.algorithm, **wt_cfg.params)
    wait_model.fit(X, df_feat["wait_time_s"])
    m = wait_model.evaluate(X, df_feat["wait_time_s"])
    logger.info("Wait model (%s) — MAE=%.1fs  RMSE=%.1fs", wt_cfg.algorithm, m["mae"], m["rmse"])

    f_cfg = cfg.model.fare
    fare_model = create_model(f_cfg.algorithm, **f_cfg.params)
    fare_model.fit(X, df_feat["avg_fare"])
    m = fare_model.evaluate(X, df_feat["avg_fare"])
    logger.info("Fare model (%s) — MAE=$%.2f  RMSE=$%.2f", f_cfg.algorithm, m["mae"], m["rmse"])

    return wait_model, fare_model, feature_cols


def save_models(wait_model, fare_model, feature_cols, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "wait_model.pkl", "wb") as f:
        pickle.dump((wait_model, feature_cols), f)
    with open(output_dir / "fare_model.pkl", "wb") as f:
        pickle.dump((fare_model, feature_cols), f)
    logger.info("Models saved to %s", output_dir)


def load_models(model_dir: str | Path):
    model_dir = Path(model_dir)
    with open(model_dir / "wait_model.pkl", "rb") as f:
        wait_model, feature_cols = pickle.load(f)
    with open(model_dir / "fare_model.pkl", "rb") as f:
        fare_model, _ = pickle.load(f)
    return wait_model, fare_model, feature_cols


if __name__ == "__main__":
    import argparse
    from nyc_taxi_strategy.utils.config import load_config

    parser = argparse.ArgumentParser(description="Train demand and fare models")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    wait_model, fare_model, feature_cols = train_models(cfg)
    save_models(wait_model, fare_model, feature_cols, Path(cfg.evaluation.output_dir) / "models")
