"""Real Estate Investment Advisor: Streamlit app."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src.preprocess import engineer_features, score_investment

st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")


@st.cache_resource
def load_models():
    return joblib.load(C.CLF_MODEL_PATH), joblib.load(C.REG_MODEL_PATH)


@st.cache_data
def load_reference():
    with open(C.REFERENCE_PATH) as f:
        ref = json.load(f)
    with open(C.METRICS_PATH) as f:
        metrics = json.load(f)
    return ref, metrics


@st.cache_data
def load_sample():
    return pd.read_csv(C.SAMPLE_PATH)


clf, reg = load_models()
ref, metrics = load_reference()
sample = load_sample()
best_clf = metrics["classification"]["best"]
best_reg = metrics["regression"]["best"]
reg_rmse = metrics["regression"]["models"][best_reg]["test"]["rmse"]

st.title("Real Estate Investment Advisor")
st.caption("Is this property a good investment, and what could it be worth in 5 years? "
           "Built on a synthetic 250,000-listing dataset: read the Model Performance tab before trusting any number.")

tab_predict, tab_search, tab_insights, tab_model = st.tabs(
    ["Predict", "Property Search", "Market Insights", "Model Performance and Method"])


def build_row(inp):
    row = pd.DataFrame([inp])
    row["Age_of_Property"] = C.CURRENT_YEAR - row["Year_Built"]
    row["Price_per_SqFt_Rs"] = row["Price_in_Lakhs"] * 1e5 / row["Size_in_SqFt"]
    row = engineer_features(row)
    return score_investment(row, ref)


with tab_predict:
    st.subheader("Enter property details")
    c1, c2, c3 = st.columns(3)
    with c1:
        state = st.selectbox("State", sorted(ref["state_cities"]))
        city = st.selectbox("City", ref["state_cities"][state])
        locality = st.selectbox("Locality", ref["city_localities"][city])
        ptype = st.selectbox("Property type", ref["Property_Type"])
        bhk = st.slider("BHK", 1, 5, 3)
        size = st.number_input("Size (sq ft)", 500, 5000, 2000, step=50)
    with c2:
        price = st.number_input("Asking price (Lakhs)", 10.0, 500.0, 200.0, step=5.0)
        year = st.slider("Year built", ref["ranges"]["year"][0], ref["ranges"]["year"][1], 2010)
        furnished = st.selectbox("Furnishing", ref["Furnished_Status"])
        floor_no = st.number_input("Floor number", 0, 30, 5)
        total_floors = st.number_input("Total floors in building", 1, 30, 12)
        availability = st.selectbox("Availability", ref["Availability_Status"])
    with c3:
        schools = st.slider("Nearby schools", 1, 10, 5)
        hospitals = st.slider("Nearby hospitals", 1, 10, 5)
        transport = st.selectbox("Public transport accessibility", ["Low", "Medium", "High"], index=1)
        parking = st.radio("Parking space", ["Yes", "No"], horizontal=True)
        security = st.radio("Security", ["Yes", "No"], horizontal=True)
        facing = st.selectbox("Facing", ref["Facing"])
        owner = st.selectbox("Owner type", ref["Owner_Type"])
    amenities = st.multiselect("Amenities", C.AMENITIES, default=["Gym", "Garden"])

    if floor_no > total_floors:
        st.error("Floor number cannot be higher than total floors.")
    elif st.button("Analyze investment", type="primary"):
        inp = {"State": state, "City": city, "Locality": locality, "Property_Type": ptype, "BHK": bhk,
               "Size_in_SqFt": size, "Price_in_Lakhs": price, "Year_Built": year,
               "Furnished_Status": furnished, "Floor_No": floor_no, "Total_Floors": total_floors,
               "Nearby_Schools": schools, "Nearby_Hospitals": hospitals,
               "Public_Transport_Accessibility": transport, "Parking_Space": parking,
               "Security": security, "Amenities": ", ".join(amenities), "Facing": facing,
               "Owner_Type": owner, "Availability_Status": availability}
        row = build_row(inp)
        proba = float(clf.predict_proba(row[C.CLF_FEATURES])[0, 1])
        future = float(reg.predict(row[C.REG_FEATURES])[0])
        cagr = ((future / price) ** (1 / C.HORIZON_YEARS) - 1) * 100
        fixed = price * (1 + C.FIXED_RATE) ** C.HORIZON_YEARS
        is_good = proba >= 0.5

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model verdict", "Good Investment" if is_good else "Not a Good Investment")
        conf = max(proba, 1 - proba) * 100
        m2.metric("Model confidence", ">99%" if conf > 99 else f"{conf:.1f}%",
                  help="Probability the model assigns to its own verdict. Because the label is a fixed "
                       "rule, confidence is very high when the property is far from the rule boundary. "
                       "It measures distance from that boundary, not real-world certainty.")
        m3.metric("Estimated price after 5 years", f"{future:,.1f} L", f"{future - price:+,.1f} L")
        m4.metric("Implied growth per year", f"{cagr:.2f}%", f"{cagr - C.FIXED_RATE * 100:+.2f} pts vs fixed 8%")
        st.caption(f"Fixed-8% projection for comparison: {fixed:,.1f} L. The model's own test error is only "
                   f"about {reg_rmse:,.1f} L (RMSE), but that just means it reproduces the growth-rate formula "
                   "used to build the target. The real uncertainty of a 5-year price forecast is far larger "
                   "and cannot be measured with this dataset. Treat the number as a scenario, not a prediction.")

        left, right = st.columns(2)
        with left:
            st.markdown("**Why: investment score breakdown (rule used to create the labels)**")
            r = row.iloc[0]
            med = ref["city_median_pps"].get(city, ref["global_median_pps"])
            rows = [
                ("Price per sq ft at or below city median", f"Rs {r.Price_per_SqFt_Rs:,.0f} vs median Rs {med:,.0f}", r.Pt_Value),
                (f"{C.BHK_MIN}+ BHK", f"{bhk} BHK", r.Pt_Family),
                ("Ready to move", availability.replace("_", " "), r.Pt_Ready),
                (f"Infrastructure score >= {C.INFRA_MIN:.0f}", f"{r.Infrastructure_Score:.0f}", r.Pt_Infra),
                (f"{C.AMENITY_MIN}+ amenities", f"{int(r.Amenity_Count)}", r.Pt_Amenity),
                (f"Assumed growth >= {C.GROWTH_MIN:.0f}% per year", f"{r.Growth_Rate_Pct:.2f}%", r.Pt_Growth),
            ]
            st.table(pd.DataFrame([(a, b, "Yes" if c else "No") for a, b, c in rows],
                                  columns=["Factor", "This property", "Point earned"]))
            rule_good = int(r.Investment_Score) >= C.GOOD_SCORE_MIN
            st.write(f"Score: **{int(r.Investment_Score)} / 6** (needs {C.GOOD_SCORE_MIN}+). "
                     f"Rule-based verdict: **{'Good' if rule_good else 'Not good'}**.")
            if rule_good != is_good:
                st.info("Model and rule disagree. The model never sees the asking price (that would leak the "
                        "label), while the rule does. Trust the rule when price matters most for you.")
        with right:
            years = np.arange(0, C.HORIZON_YEARS + 1)
            fig = go.Figure()
            fig.add_scatter(x=years, y=price * (1 + cagr / 100) ** years, name="Model forecast", mode="lines+markers")
            fig.add_scatter(x=years, y=price * (1 + C.FIXED_RATE) ** years, name="Fixed 8%", mode="lines",
                            line=dict(dash="dash"))
            fig.update_layout(title="Projected value (Lakhs)", xaxis_title="Years from now",
                              height=330, margin=dict(t=40, b=10))
            st.plotly_chart(fig, width="stretch")

        st.markdown("**What drives the models**")
        i1, i2 = st.columns(2)
        for col, key, ttl in [(i1, "classification", "Classifier: top features"),
                              (i2, "regression", "Regressor: top features")]:
            best = metrics[key]["best"]
            imp = list(metrics[key]["models"][best]["importance"].items())[:10][::-1]
            f = px.bar(x=[v * 100 for _, v in imp], y=[k for k, _ in imp], orientation="h",
                       labels={"x": "Importance (%)", "y": ""}, title=ttl)
            f.update_layout(height=340, margin=dict(t=40, b=10))
            col.plotly_chart(f, width="stretch")

with tab_search:
    st.subheader("Filter listings")
    st.caption(f"Searches a random sample of {len(sample):,} listings from the cleaned dataset.")
    f1, f2, f3 = st.columns(3)
    s_states = f1.multiselect("State", sorted(sample["State"].unique()))
    s_types = f1.multiselect("Property type", sorted(sample["Property_Type"].unique()))
    s_bhk = f2.multiselect("BHK", [1, 2, 3, 4, 5])
    s_avail = f2.multiselect("Availability", sorted(sample["Availability_Status"].unique()))
    s_price = f3.slider("Price (Lakhs)", 10, 500, (10, 500))
    s_size = f3.slider("Area (sq ft)", 500, 5000, (500, 5000))
    only_good = st.checkbox("Only properties scoring as Good Investment")
    view = sample[sample["Price_in_Lakhs"].between(*s_price) & sample["Size_in_SqFt"].between(*s_size)]
    for col, vals in [("State", s_states), ("Property_Type", s_types), ("BHK", s_bhk), ("Availability_Status", s_avail)]:
        if vals:
            view = view[view[col].isin(vals)]
    if only_good:
        view = view[view["Good_Investment"] == 1]
    sort_by = st.selectbox("Sort by", ["Investment_Score", "Future_Price_5Y_Lakhs", "Price_in_Lakhs", "Price_per_SqFt_Rs"])
    st.write(f"**{len(view):,}** matching listings (showing top 200)")
    cols = ["State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
            "Price_per_SqFt_Rs", "Availability_Status", "Investment_Score", "Good_Investment",
            "Growth_Rate_Pct", "Future_Price_5Y_Lakhs"]
    st.dataframe(view.sort_values(sort_by, ascending=False).head(200)[cols].round(2),
                 width="stretch", hide_index=True)

with tab_insights:
    st.subheader("Market insights")
    st.warning("This dataset is synthetic: most group differences below are tiny (often under 3%). "
               "Charts are real, but the market patterns they would normally show are not present in the data.")
    a, b = st.columns(2)
    city_price = sample.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(city_price, x="City", y="Price_in_Lakhs", title="Average price by city (Lakhs)")
    a.plotly_chart(fig, width="stretch")
    heat = sample.pivot_table(index="State", columns="Property_Type", values="Price_per_SqFt_Rs", aggfunc="mean")
    fig = px.imshow(heat.round(0), aspect="auto", color_continuous_scale="YlOrRd",
                    title="Location heatmap: price per sq ft (Rs) by state and property type")
    b.plotly_chart(fig, width="stretch")
    a, b = st.columns(2)
    fig = px.scatter(sample.sample(3000, random_state=1), x="Size_in_SqFt", y="Price_in_Lakhs",
                     color="Property_Type", opacity=0.5, title="Size vs price")
    a.plotly_chart(fig, width="stretch")
    good_city = (sample.groupby("City")["Good_Investment"].mean() * 100).sort_values(ascending=False).reset_index()
    fig = px.bar(good_city, x="City", y="Good_Investment", title="Good Investment rate by city (%)")
    b.plotly_chart(fig, width="stretch")
    a, b = st.columns(2)
    trend = sample.assign(Built=(sample["Year_Built"] // 5 * 5)).groupby("Built")["Price_in_Lakhs"].mean().reset_index()
    fig = px.line(trend, x="Built", y="Price_in_Lakhs", markers=True, title="Average price by build period (5-year bins)")
    a.plotly_chart(fig, width="stretch")
    inf = sample.assign(Band=pd.cut(sample["Infrastructure_Score"], [0, 40, 50, 60, 70, 80, 100]).astype(str)) \
        .groupby("Band")[["Price_in_Lakhs", "Future_Price_5Y_Lakhs"]].mean().reset_index()
    fig = px.bar(inf, x="Band", y=["Price_in_Lakhs", "Future_Price_5Y_Lakhs"], barmode="group",
                 title="Infrastructure score band vs current and 5-year price (Lakhs)")
    b.plotly_chart(fig, width="stretch")

with tab_model:
    st.subheader("Model performance (held-out test set, 37,500 listings)")
    cm = metrics["classification"]["models"]
    rows = [{"Model": n, "Uses price features": r["leaky"], "Accuracy": r["test"]["accuracy"],
             "Precision": r["test"]["precision"], "Recall": r["test"]["recall"], "F1": r["test"]["f1"],
             "ROC AUC": r["test"]["roc_auc"]} for n, r in cm.items()]
    st.markdown(f"**Classification: Good Investment** (deployed: `{best_clf}`)")
    st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
    leaky = next((r for r in rows if r["Uses price features"]), None)
    if leaky:
        st.error(f"The `{leaky['Model']}` row shows what happens when price columns are fed to the classifier: "
                 f"accuracy {leaky['Accuracy']*100:.1f}%. That number is data leakage, not skill, "
                 "and that model is NOT deployed.")
    cmx = np.array(cm[best_clf]["confusion_matrix_test"])
    fig = px.imshow(cmx, text_auto=",", x=["Not good", "Good"], y=["Not good", "Good"],
                    labels=dict(x="Predicted", y="Actual"), color_continuous_scale="Blues",
                    title=f"Confusion matrix: {best_clf}")
    fig.update_layout(height=360, width=420)
    st.plotly_chart(fig)

    rg = metrics["regression"]["models"]
    base = metrics["regression"]["baseline_fixed_8pct_test"]
    rrows = [{"Model": "Baseline: fixed 8% growth", "RMSE (L)": base["rmse"], "MAE (L)": base["mae"], "R2": base["r2"]}]
    rrows += [{"Model": n, "RMSE (L)": r["test"]["rmse"], "MAE (L)": r["test"]["mae"], "R2": r["test"]["r2"]}
              for n, r in rg.items()]
    st.markdown(f"**Regression: price after 5 years** (deployed: `{best_reg}`)")
    st.dataframe(pd.DataFrame(rrows).round(4), hide_index=True, width="stretch")

    with st.expander("Method, assumptions and honest limitations", expanded=True):
        st.markdown(f"""
**Data.** 250,000 listings, 20 states, 42 cities. No missing values or duplicates. Floor number was higher than
total floors in about 46% of rows; the pair was swapped to make it valid. Price per sq ft is recomputed in rupees.

**Good Investment label.** Rule-based, as the brief allows: 1 point each for price per sq ft at or below the city
median, {C.BHK_MIN}+ BHK, ready to move, infrastructure score of {C.INFRA_MIN:.0f}+, {C.AMENITY_MIN}+ amenities and assumed
growth of {C.GROWTH_MIN:.0f}%+. Score of {C.GOOD_SCORE_MIN} or more is Good. The classifier is therefore learning this rule
from property attributes, with price hidden. It is not discovering what makes real investments succeed.

**Future price label.** The dataset has no price history, so growth is assumed: city tier base rate (9 / 7.5 / 6%),
adjusted for property type, transport access, construction status, age and infrastructure, clipped to 4 to 13%.
Future price = current price x (1 + rate)^5. The regressor learns this formula, so its R2 is very high by design.
It does not prove that it can forecast real prices.

**Limitations.** Synthetic data (price unrelated to size, city, amenities), no crime-rate column, no sale dates,
Localities are generic labels. Do not use these outputs for a real purchase decision.
""")