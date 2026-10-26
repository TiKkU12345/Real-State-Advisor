"""Central configuration: paths, constants and business assumptions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "india_housing_prices.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "housing_processed.csv.gz"
SAMPLE_PATH = ROOT / "data" / "processed" / "housing_sample.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
REFERENCE_PATH = MODEL_DIR / "reference_stats.json"
CLF_MODEL_PATH = MODEL_DIR / "classifier.joblib"
REG_MODEL_PATH = MODEL_DIR / "regressor.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
MLFLOW_URI = f"sqlite:///{ROOT / 'mlflow.db'}"

RANDOM_STATE = 42
CURRENT_YEAR = 2025
HORIZON_YEARS = 5
FIXED_RATE = 0.08

AMENITIES = ["Playground", "Gym", "Garden", "Pool", "Clubhouse"]
TRANSPORT_MAP = {"Low": 1, "Medium": 2, "High": 3}

TIER1 = ["Mumbai", "New Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune",
         "Kolkata", "Gurgaon", "Noida", "Ahmedabad", "Dwarka"]
TIER2 = ["Jaipur", "Lucknow", "Indore", "Bhopal", "Nagpur", "Surat", "Kochi",
         "Coimbatore", "Vishakhapatnam", "Trivandrum", "Bhubaneswar", "Dehradun",
         "Faridabad", "Mysore", "Mangalore", "Guwahati", "Vijayawada", "Raipur",
         "Patna", "Ranchi", "Ludhiana", "Amritsar", "Jodhpur"]
BASE_GROWTH = {"tier1": 9.0, "tier2": 7.5, "tier3": 6.0}
TYPE_GROWTH_ADJ = {"Villa": 0.4, "Independent House": 0.2, "Apartment": 0.0}
TRANSPORT_GROWTH_ADJ = {"High": 0.6, "Medium": 0.2, "Low": -0.2}
UNDER_CONSTRUCTION_ADJ = 0.5
GROWTH_CLIP = (4.0, 13.0)

# Multi-factor "Good Investment" rule: a property earns 1 point per factor
INFRA_MIN = 60.0
AMENITY_MIN = 3
BHK_MIN = 3
GROWTH_MIN = 8.0
GOOD_SCORE_MIN = 4

CAT_FEATURES = ["State", "City", "Property_Type", "Furnished_Status",
                "Public_Transport_Accessibility", "Facing", "Owner_Type",
                "Availability_Status"]
NUM_FEATURES = ["BHK", "Size_in_SqFt", "Age_of_Property", "Floor_No", "Total_Floors",
                "Floor_Ratio", "Size_per_BHK", "Nearby_Schools", "Nearby_Hospitals",
                "School_Density_Score", "Hospital_Score", "Transport_Score",
                "Infrastructure_Score", "Amenity_Count", "Parking_Flag",
                "Security_Flag"] + [f"Has_{a}" for a in AMENITIES]
CLF_FEATURES = NUM_FEATURES + CAT_FEATURES
PRICE_LEAK_FEATURES = ["Price_in_Lakhs", "Price_per_SqFt_Rs"]
REG_FEATURES = NUM_FEATURES + ["Price_in_Lakhs"] + CAT_FEATURES
