"""EDA for the 20 brief questions. Saves figures and an auto-generated findings report."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config as C

sns.set_theme(style="whitegrid", context="notebook")
FINDINGS = []


def save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / f"{name}.png", dpi=110)
    plt.close(fig)


def note(q, title, text):
    FINDINGS.append(f"**Q{q}. {title}**\n\n{text}\n")


def spread_text(means, unit=""):
    """Describe the gap between best and worst group, flag it if negligible."""
    lo, hi = means.min(), means.max()
    gap = (hi - lo) / means.mean() * 100
    verdict = ("negligible, no real effect" if gap < 3 else
               "small" if gap < 10 else "meaningful")
    return (f"group means range from {lo:,.1f}{unit} to {hi:,.1f}{unit} "
            f"(gap {gap:.1f}% of the overall mean: {verdict})")


def main():
    df = pd.read_csv(C.PROCESSED_PATH)
    df["PPS_k"] = df["Price_per_SqFt_Rs"] / 1000

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["Price_in_Lakhs"], bins=50, ax=ax)
    ax.set(title="Q1. Price distribution", xlabel="Price (Lakhs)")
    save(fig, "q01_price_distribution")
    p = df["Price_in_Lakhs"]
    note(1, "Distribution of property prices",
         f"Prices run from {p.min():.0f} to {p.max():.0f} lakhs, mean {p.mean():.1f}, median {p.median():.1f}, "
         f"skew {p.skew():.2f}. The histogram is flat (uniform), not right-skewed like real housing prices.")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["Size_in_SqFt"], bins=50, ax=ax, color="darkorange")
    ax.set(title="Q2. Size distribution", xlabel="Size (sq ft)")
    save(fig, "q02_size_distribution")
    s = df["Size_in_SqFt"]
    note(2, "Distribution of property sizes",
         f"Sizes run from {s.min()} to {s.max()} sq ft, mean {s.mean():.0f}, median {s.median():.0f}, skew {s.skew():.2f}. "
         "Again uniform. A 500 sq ft to 5000 sq ft spread with equal density is a synthetic-data signature.")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="Property_Type", y="PPS_k", ax=ax)
    ax.set(title="Q3. Price per sq ft by property type", ylabel="Price per sq ft (Rs thousand)")
    save(fig, "q03_pps_by_type")
    m = df.groupby("Property_Type")["Price_per_SqFt_Rs"].mean() / 1000
    note(3, "Price per sq ft by property type",
         f"Mean Rs {', '.join(f'{k}: {v:.1f}k' for k, v in m.items())}. {spread_text(m, 'k')}.")

    sample = df.sample(6000, random_state=1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.scatterplot(data=sample, x="Size_in_SqFt", y="Price_in_Lakhs", alpha=0.3, s=12, ax=ax)
    corr = df["Size_in_SqFt"].corr(df["Price_in_Lakhs"])
    ax.set(title=f"Q4. Size vs price (corr = {corr:.3f})")
    save(fig, "q04_size_vs_price")
    note(4, "Relationship between size and price",
         f"Correlation is {corr:.3f}, i.e. none. In real markets this is one of the strongest relationships. "
         "Price here is independent of size, so Price_per_SqFt is driven almost entirely by size (corr "
         f"{df['Size_in_SqFt'].corr(df['Price_per_SqFt_Rs']):.2f}).")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.boxplot(y=df["PPS_k"], ax=axes[0], color="lightcoral")
    axes[0].set(title="Price per sq ft (Rs thousand)")
    sns.boxplot(y=df["Size_in_SqFt"], ax=axes[1], color="lightgreen")
    axes[1].set(title="Size (sq ft)")
    fig.suptitle("Q5. Outlier check")
    save(fig, "q05_outliers")
    note(5, "Outliers in price per sq ft or size",
         f"IQR rule flags {int(df.Outlier_PPS.sum()):,} rows ({df.Outlier_PPS.mean()*100:.1f}%) for price per sq ft "
         f"(small properties with high prices), and {int(df.Outlier_Size.sum())} for size and "
         f"{int(df.Outlier_Price.sum())} for price. They are valid combinations, so they are flagged "
         "(Outlier_PPS column), not deleted.")

    st = df.groupby("State")["Price_per_SqFt_Rs"].mean().sort_values() / 1000
    fig, ax = plt.subplots(figsize=(8, 6))
    st.plot.barh(ax=ax, color="steelblue")
    ax.set(title="Q6. Average price per sq ft by state", xlabel="Rs thousand")
    save(fig, "q06_pps_by_state")
    note(6, "Average price per sq ft by state",
         f"Highest: {st.index[-1]} ({st.iloc[-1]:.2f}k), lowest: {st.index[0]} ({st.iloc[0]:.2f}k). {spread_text(st, 'k')}.")

    ct = df.groupby("City")["Price_in_Lakhs"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 9))
    ct.plot.barh(ax=ax, color="seagreen")
    ax.set(title="Q7. Average price by city", xlabel="Lakhs")
    ax.set_xlim(ct.min() * 0.98, ct.max() * 1.01)
    save(fig, "q07_price_by_city")
    note(7, "Average property price by city",
         f"Highest: {ct.index[-1]} ({ct.iloc[-1]:.1f} L), lowest: {ct.index[0]} ({ct.iloc[0]:.1f} L). {spread_text(ct, ' L')}. "
         f"Mumbai averages {ct['Mumbai']:.1f} L and Silchar {ct['Silchar']:.1f} L, a gap that real markets never show.")

    loc = df.groupby("Locality")["Age_of_Property"].median().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    pd.concat([loc.head(10), loc.tail(10)]).plot.barh(ax=ax, color="slateblue")
    ax.set(title="Q8. Median property age by locality (10 lowest and 10 highest)", xlabel="Years")
    save(fig, "q08_age_by_locality")
    note(8, "Median age of properties by locality",
         f"Across {loc.size} localities the median age ranges from {loc.min():.0f} to {loc.max():.0f} years "
         f"(overall median {df.Age_of_Property.median():.0f}). Localities are labelled Locality_1..500 and are "
         "reused across cities, so a 'locality' is not a real neighbourhood in this data.")

    bhk_city = pd.crosstab(df["City"], df["BHK"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(7, 10))
    sns.heatmap(bhk_city, annot=True, fmt=".0f", cmap="Blues", ax=ax, cbar_kws={"label": "% of listings"})
    ax.set(title="Q9. BHK distribution across cities (%)")
    save(fig, "q09_bhk_by_city")
    note(9, "BHK distribution across cities",
         f"Every city has roughly 20% in each of 1 to 5 BHK (per-cell range {bhk_city.values.min():.1f}% to "
         f"{bhk_city.values.max():.1f}%). No city specialises in any BHK.")

    top5 = df.groupby("Locality")["Price_in_Lakhs"].mean().nlargest(5).index
    t = df[df["Locality"].isin(top5)].copy()
    t["Built_Period"] = (t["Year_Built"] // 5 * 5).astype(int)
    trend = t.groupby(["Built_Period", "Locality"])["Price_in_Lakhs"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(data=trend, x="Built_Period", y="Price_in_Lakhs", hue="Locality", marker="o", ax=ax)
    ax.set(title="Q10. Price by build period, top 5 priciest localities", xlabel="Year built (5-year bins)")
    save(fig, "q10_top5_locality_trend")
    note(10, "Price trends for the top 5 most expensive localities",
         f"Top 5: {', '.join(top5)} (mean {df[df.Locality.isin(top5)].Price_in_Lakhs.mean():.0f} L vs overall "
         f"{df.Price_in_Lakhs.mean():.0f} L). The dataset has no sale-date column, so a true price-over-time trend "
         "is impossible. Price by build period is the closest proxy and it is noisy, with no consistent upward line.")

    num = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_Rs", "Age_of_Property", "Floor_No",
           "Total_Floors", "Nearby_Schools", "Nearby_Hospitals", "Infrastructure_Score", "Amenity_Count",
           "Growth_Rate_Pct", "Future_Price_5Y_Lakhs", "Investment_Score"]
    cm = df[num].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
    ax.set(title="Q11. Correlation matrix")
    save(fig, "q11_correlation")
    pairs = (cm.where(np.triu(np.ones(cm.shape), 1).astype(bool)).stack().abs().sort_values(ascending=False))
    raw_pairs = [(a, b) for (a, b) in pairs.index
                 if "Future" not in a + b and "Investment_Score" not in a + b and "Growth" not in a + b][:3]
    note(11, "Correlation between numeric features",
         "Strongest raw relationships: " + "; ".join(
             f"{a} vs {b} = {cm.loc[a, b]:.2f}" for a, b in raw_pairs) +
         ". Price_per_SqFt is just Price / Size, so those two links are mathematical, not behavioural. "
         "Future_Price_5Y_Lakhs is almost the same as Price_in_Lakhs by construction.")

    for q, col, name, color in [(12, "Nearby_Schools", "schools", "teal"), (13, "Nearby_Hospitals", "hospitals", "crimson")]:
        g = df.groupby(col)["Price_per_SqFt_Rs"].mean() / 1000
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(g.index, g.values, marker="o", color=color)
        ax.set(title=f"Q{q}. Price per sq ft vs nearby {name}", xlabel=col, ylabel="Rs thousand")
        save(fig, f"q{q}_pps_vs_{name}")
        c = df[col].corr(df["Price_per_SqFt_Rs"])
        note(q, f"Nearby {name} vs price per sq ft", f"Correlation {c:.3f}; {spread_text(g, 'k')}.")

    g = df.groupby("Furnished_Status")[["Price_in_Lakhs", "Price_per_SqFt_Rs"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    g["Price_in_Lakhs"].plot.bar(ax=axes[0], color="goldenrod", rot=0)
    axes[0].set(title="Mean price (L)")
    (g["Price_per_SqFt_Rs"] / 1000).plot.bar(ax=axes[1], color="peru", rot=0)
    axes[1].set(title="Mean price per sq ft (Rs k)")
    fig.suptitle("Q14. Price by furnished status")
    save(fig, "q14_price_by_furnishing")
    note(14, "Price by furnished status", f"Mean price: {spread_text(g['Price_in_Lakhs'], ' L')}.")

    g = df.groupby("Facing")["Price_per_SqFt_Rs"].mean() / 1000
    fig, ax = plt.subplots(figsize=(6, 4))
    g.plot.bar(ax=ax, color="indianred", rot=0)
    ax.set(title="Q15. Price per sq ft by facing", ylabel="Rs thousand")
    save(fig, "q15_pps_by_facing")
    note(15, "Price per sq ft by facing direction", f"{spread_text(g, 'k')}.")

    oc = df["Owner_Type"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    oc.plot.bar(ax=ax, color="mediumpurple", rot=0)
    ax.set(title="Q16. Listings by owner type")
    save(fig, "q16_owner_type")
    note(16, "Properties per owner type", ", ".join(f"{k}: {v:,} ({v/len(df)*100:.1f}%)" for k, v in oc.items()) + ".")

    av = df["Availability_Status"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    av.plot.bar(ax=ax, color="darkcyan", rot=0)
    ax.set(title="Q17. Listings by availability status")
    save(fig, "q17_availability")
    note(17, "Properties per availability status",
         ", ".join(f"{k}: {v:,} ({v/len(df)*100:.1f}%)" for k, v in av.items()) +
         ". The brief mentions 'Sold' as a status, but it does not appear in this file.")

    g = df.groupby("Parking_Space")[["Price_in_Lakhs", "Price_per_SqFt_Rs"]].mean()
    fig, ax = plt.subplots(figsize=(5, 4))
    g["Price_in_Lakhs"].plot.bar(ax=ax, color="sienna", rot=0)
    ax.set(title="Q18. Mean price by parking space", ylabel="Lakhs")
    save(fig, "q18_parking")
    note(18, "Does parking space affect price?",
         f"Mean price with parking {g.loc['Yes','Price_in_Lakhs']:.1f} L vs without {g.loc['No','Price_in_Lakhs']:.1f} L: "
         f"{spread_text(g['Price_in_Lakhs'], ' L')}.")

    g = df.groupby("Amenity_Count")["Price_per_SqFt_Rs"].mean() / 1000
    per = {a: df.loc[df[f"Has_{a}"] == 1, "Price_per_SqFt_Rs"].mean() / 1000 for a in C.AMENITIES}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(g.index, g.values, marker="o")
    axes[0].set(title="By number of amenities", xlabel="Amenity count", ylabel="PPS (Rs k)")
    pd.Series(per).plot.bar(ax=axes[1], rot=30, color="olive")
    axes[1].set(title="With each amenity", ylabel="PPS (Rs k)")
    fig.suptitle("Q19. Amenities vs price per sq ft")
    save(fig, "q19_amenities")
    note(19, "How amenities affect price per sq ft", f"By amenity count: {spread_text(g, 'k')}. "
         f"By individual amenity: {spread_text(pd.Series(per), 'k')}.")

    g = df.groupby("Public_Transport_Accessibility").agg(pps=("Price_per_SqFt_Rs", "mean"),
                                                         good=("Good_Investment", "mean"))
    g = g.reindex(["Low", "Medium", "High"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    (g["pps"] / 1000).plot.bar(ax=axes[0], color="navy", rot=0)
    axes[0].set(title="Mean PPS (Rs k)")
    (g["good"] * 100).plot.bar(ax=axes[1], color="darkgreen", rot=0)
    axes[1].set(title="Good Investment rate (%)")
    fig.suptitle("Q20. Public transport accessibility")
    save(fig, "q20_transport")
    note(20, "Public transport vs price per sq ft and investment potential",
         f"PPS: {spread_text(g['pps'] / 1000, 'k')}. Good-Investment rate rises from {g.loc['Low','good']*100:.1f}% (Low) to "
         f"{g.loc['High','good']*100:.1f}% (High). That rise is mechanical: transport feeds the infrastructure score and "
         "the growth rate, which are inputs of the label.")

    gi = df.groupby("City")["Good_Investment"].mean().sort_values() * 100
    fig, ax = plt.subplots(figsize=(8, 9))
    gi.plot.barh(ax=ax, color="darkorange")
    ax.set(title="Extra. Good Investment rate by city (%)")
    save(fig, "extra_good_by_city")
    inf = df.groupby(pd.cut(df["Infrastructure_Score"], [0, 40, 50, 60, 70, 80, 100]), observed=True)[
        ["Price_in_Lakhs", "Future_Price_5Y_Lakhs"]].mean()
    note("X1", "Infrastructure score vs resale value (brief EDA step 2)",
         f"Mean current price by infrastructure band: {spread_text(inf['Price_in_Lakhs'], ' L')}. "
         "There is no crime-rate column in the dataset, so the 'impact of crime rate' analysis cannot be done. "
         "Infrastructure and transport are the closest available proxies.")
    note("X2", "Good Investment by city",
         f"Rate ranges from {gi.min():.1f}% ({gi.index[0]}) to {gi.max():.1f}% ({gi.index[-1]}), driven by the "
         "city-tier growth assumption and the city-median price rule.")

    (C.REPORT_DIR / "eda_findings.md").write_text(
        "# EDA findings (auto-generated by src/eda.py)\n\n" + "\n".join(FINDINGS))
    print("done", len(FINDINGS), "findings")


if __name__ == "__main__":
    main()
