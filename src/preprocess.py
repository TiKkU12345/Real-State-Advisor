"""Data cleaning, feature engineering and target creation."""
import json

import numpy as np
import pandas as pd

from src import config as C


def load_raw(path=None):
    return pd.read_csv(path or C.RAW_PATH)


def clean(raw):
    """Handle duplicates, missing values, text noise and impossible floor values."""
    report = {"rows_raw": int(len(raw))}
    df = raw.copy()

    feature_cols = [c for c in df.columns if c != "ID"]
    before = len(df)
    df = df.drop_duplicates(subset=feature_cols).reset_index(drop=True)
    report["duplicates_removed"] = int(before - len(df))

    report["missing_values_before"] = int(df.isna().sum().sum())
    num_cols = [c for c in df.columns if c != "ID" and pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c != "ID" and c not in num_cols]
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0]).str.strip()
    report["missing_values_after"] = int(df.isna().sum().sum())

    # Floor_No cannot exceed Total_Floors. Swap the pair where it does.
    bad = df["Floor_No"] > df["Total_Floors"]
    report["floor_greater_than_total_fixed"] = int(bad.sum())
    lo = np.minimum(df["Floor_No"], df["Total_Floors"])
    hi = np.maximum(df["Floor_No"], df["Total_Floors"])
    df["Floor_No"], df["Total_Floors"] = lo, hi

    df["Age_of_Property"] = C.CURRENT_YEAR - df["Year_Built"]
    df["Price_per_SqFt_Rs"] = df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]

    # Outliers are flagged (IQR rule), not dropped: the values are valid listings.
    for col, flag in [("Price_per_SqFt_Rs", "Outlier_PPS"), ("Size_in_SqFt", "Outlier_Size"),
                      ("Price_in_Lakhs", "Outlier_Price")]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df[flag] = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).astype(int)
        report[f"{flag}_count"] = int(df[flag].sum())
    return df, report


def engineer_features(df):
    """Input-only features (no price information), safe to compute in the app."""
    df = df.copy()
    df["Floor_Ratio"] = df["Floor_No"] / df["Total_Floors"].clip(lower=1)
    df["Size_per_BHK"] = df["Size_in_SqFt"] / df["BHK"].clip(lower=1)
    df["School_Density_Score"] = df["Nearby_Schools"] * 10.0
    df["Hospital_Score"] = df["Nearby_Hospitals"] * 10.0
    df["Transport_Score"] = df["Public_Transport_Accessibility"].map(C.TRANSPORT_MAP)
    df["Infrastructure_Score"] = (0.3 * df["School_Density_Score"]
                                  + 0.3 * df["Hospital_Score"]
                                  + 0.4 * df["Transport_Score"] / 3 * 100)
    for a in C.AMENITIES:
        df[f"Has_{a}"] = df["Amenities"].fillna("").str.contains(a).astype(int)
    df["Amenity_Count"] = df[[f"Has_{a}" for a in C.AMENITIES]].sum(axis=1)
    df["Parking_Flag"] = (df["Parking_Space"] == "Yes").astype(int)
    df["Security_Flag"] = (df["Security"] == "Yes").astype(int)
    if "Price_in_Lakhs" in df and "Price_per_SqFt_Rs" not in df:
        df["Price_per_SqFt_Rs"] = df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]
    return df


def city_tier(city):
    if city in C.TIER1:
        return "tier1"
    if city in C.TIER2:
        return "tier2"
    return "tier3"


def growth_rate_pct(df):
    """Assumed annual appreciation (%): city tier base + property/location adjustments."""
    base = df["City"].map(city_tier).map(C.BASE_GROWTH)
    r = (base
         + df["Property_Type"].map(C.TYPE_GROWTH_ADJ).fillna(0)
         + df["Public_Transport_Accessibility"].map(C.TRANSPORT_GROWTH_ADJ).fillna(0)
         + np.where(df["Availability_Status"] == "Under_Construction", C.UNDER_CONSTRUCTION_ADJ, 0)
         - df["Age_of_Property"] / 35 * 0.6
         + (df["Infrastructure_Score"] - 60) / 40 * 0.5)
    return r.clip(*C.GROWTH_CLIP)


def build_reference(df):
    ref = {
        "city_median_pps": df.groupby("City")["Price_per_SqFt_Rs"].median().round(2).to_dict(),
        "city_median_price": df.groupby("City")["Price_in_Lakhs"].median().round(2).to_dict(),
        "state_cities": {s: sorted(g["City"].unique().tolist()) for s, g in df.groupby("State")},
        "city_localities": {c: sorted(g["Locality"].unique().tolist(), key=lambda x: int(x.split("_")[-1]))
                            for c, g in df.groupby("City")},
        "global_median_pps": float(df["Price_per_SqFt_Rs"].median()),
        "ranges": {"size": [int(df.Size_in_SqFt.min()), int(df.Size_in_SqFt.max())],
                   "price": [float(df.Price_in_Lakhs.min()), float(df.Price_in_Lakhs.max())],
                   "year": [int(df.Year_Built.min()), int(df.Year_Built.max())]},
    }
    for col in ["Property_Type", "Furnished_Status", "Facing", "Owner_Type",
                "Availability_Status", "Public_Transport_Accessibility"]:
        ref[col] = sorted(df[col].unique().tolist())
    return ref


def score_investment(df, ref):
    """Multi-factor rule label. Returns df with per-factor points and Good_Investment."""
    df = df.copy()
    city_med = df["City"].map(ref["city_median_pps"]).fillna(ref["global_median_pps"])
    df["Growth_Rate_Pct"] = growth_rate_pct(df)
    df["Pt_Value"] = (df["Price_per_SqFt_Rs"] <= city_med).astype(int)
    df["Pt_Family"] = (df["BHK"] >= C.BHK_MIN).astype(int)
    df["Pt_Ready"] = (df["Availability_Status"] == "Ready_to_Move").astype(int)
    df["Pt_Infra"] = (df["Infrastructure_Score"] >= C.INFRA_MIN).astype(int)
    df["Pt_Amenity"] = (df["Amenity_Count"] >= C.AMENITY_MIN).astype(int)
    df["Pt_Growth"] = (df["Growth_Rate_Pct"] >= C.GROWTH_MIN).astype(int)
    pts = ["Pt_Value", "Pt_Family", "Pt_Ready", "Pt_Infra", "Pt_Amenity", "Pt_Growth"]
    df["Investment_Score"] = df[pts].sum(axis=1)
    df["Good_Investment"] = (df["Investment_Score"] >= C.GOOD_SCORE_MIN).astype(int)
    return df


def add_future_price(df):
    df = df.copy()
    df["Future_Price_5Y_Lakhs"] = (df["Price_in_Lakhs"]
                                   * (1 + df["Growth_Rate_Pct"] / 100) ** C.HORIZON_YEARS).round(2)
    return df


def run():
    raw = load_raw()
    df, report = clean(raw)
    df = engineer_features(df)
    ref = build_reference(df)
    df = score_investment(df, ref)
    df = add_future_price(df)

    report["rows_clean"] = int(len(df))
    report["good_investment_rate"] = round(float(df["Good_Investment"].mean()), 4)
    report["growth_rate_pct_mean"] = round(float(df["Growth_Rate_Pct"].mean()), 3)
    report["growth_rate_pct_min"] = round(float(df["Growth_Rate_Pct"].min()), 3)
    report["growth_rate_pct_max"] = round(float(df["Growth_Rate_Pct"].max()), 3)

    C.PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    C.MODEL_DIR.mkdir(exist_ok=True)
    df.to_csv(C.PROCESSED_PATH, index=False, compression="gzip")
    df.sample(20000, random_state=C.RANDOM_STATE).to_csv(C.SAMPLE_PATH, index=False)
    with open(C.REFERENCE_PATH, "w") as f:
        json.dump(ref, f)
    C.REPORT_DIR.mkdir(exist_ok=True)
    with open(C.REPORT_DIR / "cleaning_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(df["Investment_Score"].value_counts().sort_index().to_dict())
    return df


if __name__ == "__main__":
    run()
