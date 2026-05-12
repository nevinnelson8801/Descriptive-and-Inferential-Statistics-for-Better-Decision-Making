"""
Descriptive and Inferential Statistics for Better Decision-Making
=================================================================
Student : Nevin Nelson (Q1071615)
Programme: BSc Computer Science and Digitisation (2024-2027)
Dataset  : Swiss_data_set.csv (47 Swiss/French municipalities)

Chapters covered:
    1. Dataset Description
    2. Data Cleaning (IQR outlier removal)
    3. Visualization of variable distributions
    4. Min-Max Normalization
    5. Relationship of socioeconomic indicators with Fertility Rate
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = "Swiss_data_set.csv"
NUMERIC_COLS = ["Fertility", "Agriculture", "Examination", "Education",
                "Catholic", "Infant.Mortality"]

# =============================================================================
# CHAPTER 1: LOAD & DESCRIBE THE DATASET
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    """Load the Swiss dataset and return a DataFrame."""
    df = pd.read_csv(path)
    df = df.rename(columns={"Unnamed: 0": "Region"})
    return df


def describe_dataset(df: pd.DataFrame) -> None:
    """Print a description of the dataset (Fig. 1)."""
    print("=" * 60)
    print("CHAPTER 1: DATASET DESCRIPTION")
    print("=" * 60)
    print(f"\nShape : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumn names:\n  {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nDescriptive statistics:\n{df[NUMERIC_COLS].describe().round(2)}")

    # Fig. 1 — visual overview (pair plot)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, col in enumerate(NUMERIC_COLS):
        axes[i].hist(df[col], bins=10, color="steelblue", edgecolor="white")
        axes[i].set_title(col)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Count")
    plt.suptitle("Fig. 1 — Visual Representation of All Variables", fontsize=14)
    plt.tight_layout()
    plt.savefig("fig1_all_variables.png", dpi=150)
    plt.show()
    print("\n[Fig. 1 saved as fig1_all_variables.png]")


# =============================================================================
# CHAPTER 2: DATA CLEANING
# =============================================================================

def compute_key_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean, median, mode, range, variance, and std for each
    numeric column (Fig. 3).
    """
    stats_dict = {}
    for col in NUMERIC_COLS:
        s = df[col]
        stats_dict[col] = {
            "mean"     : s.mean(),
            "median"   : s.median(),
            "mode"     : s.mode()[0],
            "range"    : s.max() - s.min(),
            "variance" : s.var(),
            "std_dev"  : s.std(),
        }
    stats_df = pd.DataFrame(stats_dict).T.round(4)
    return stats_df


def remove_outliers_iqr(df: pd.DataFrame,
                        cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove rows whose values in any of `cols` fall outside
    [Q1 - 1.5*IQR, Q3 + 1.5*IQR].

    Returns
    -------
    df_cleaned   : DataFrame with outlier rows dropped
    outliers_df  : DataFrame of the removed rows
    """
    mask_keep = pd.Series([True] * len(df), index=df.index)
    outlier_info = {}

    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        is_outlier = (df[col] < lower) | (df[col] > upper)
        n_out = is_outlier.sum()
        outlier_info[col] = n_out
        print(f"  {col:<20} lower={lower:.2f}  upper={upper:.2f}  "
              f"outliers={n_out}")
        mask_keep &= ~is_outlier

    outliers_df = df[~mask_keep].copy()
    df_cleaned  = df[mask_keep].copy().reset_index(drop=True)
    return df_cleaned, outliers_df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full data-cleaning pipeline and print results (Fig. 2–4)."""
    print("\n" + "=" * 60)
    print("CHAPTER 2: DATA CLEANING")
    print("=" * 60)

    # Missing values
    print(f"\nMissing values per column:\n{df.isnull().sum()}")

    # Outlier analysis — only remove from Fertility, Education, Infant.Mortality
    print("\nOutlier detection (IQR method):")
    cols_to_clean = ["Fertility", "Education", "Infant.Mortality"]
    df_cleaned, outliers = remove_outliers_iqr(df, cols_to_clean)

    # Fig. 2 — removed outliers
    if not outliers.empty:
        print(f"\nFig. 2 — Removed outliers ({len(outliers)} rows):\n{outliers}")
    else:
        print("\nNo outliers removed.")

    # Fig. 3 — key statistics on cleaned data
    key_stats = compute_key_statistics(df_cleaned)
    print(f"\nFig. 3 — Key Statistics (cleaned dataset):\n{key_stats}")

    # Fig. 4 — visual of cleaned dataset
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, col in enumerate(NUMERIC_COLS):
        axes[i].hist(df_cleaned[col], bins=10, color="seagreen", edgecolor="white")
        axes[i].set_title(col)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Count")
    plt.suptitle("Fig. 4 — Visual Representation of Cleaned Dataset (df_cleaned)",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig("fig4_cleaned_data.png", dpi=150)
    plt.show()
    print("\n[Fig. 4 saved as fig4_cleaned_data.png]")

    print(f"\nOriginal rows : {len(df)}")
    print(f"Cleaned rows  : {len(df_cleaned)}")
    return df_cleaned


# =============================================================================
# CHAPTER 3: VISUALIZE DISTRIBUTIONS
# =============================================================================

FIG_META = {
    "Fertility"       : ("Fig. 5",  "steelblue",    "Distribution of Fertility across Municipalities"),
    "Agriculture"     : ("Fig. 6",  "darkorange",   "Distribution of Agriculture across Municipalities"),
    "Examination"     : ("Fig. 7",  "mediumpurple", "Distribution of Examination across Municipalities"),
    "Education"       : ("Fig. 8",  "crimson",      "Distribution of Education across Municipalities"),
    "Catholic"        : ("Fig. 9",  "goldenrod",    "Distribution of Catholics across Municipalities"),
    "Infant.Mortality": ("Fig. 10", "teal",         "Distribution of Infant Mortality across Municipalities"),
}

DESCRIPTIONS = {
    "Fertility"       : "Slightly right-skewed. Peak ~70. Range ~55–95.",
    "Agriculture"     : "Close to normal, slightly right-skewed. Peak ~60. Range ~10–90.",
    "Examination"     : "Right-skewed. Peak ~10–15. Range ~5–30.",
    "Education"       : "Right-skewed. Peak <10. Range ~2–17.",
    "Catholic"        : "Bimodal — clusters at 0–20% and 80–100%.",
    "Infant.Mortality": "Close to normal, slightly right-skewed. Peak ~20. Range ~15–27.",
}


def visualize_distributions(df_cleaned: pd.DataFrame) -> None:
    """Plot individual histograms for each variable (Fig. 5–10)."""
    print("\n" + "=" * 60)
    print("CHAPTER 3: VARIABLE DISTRIBUTIONS")
    print("=" * 60)

    for col, (fig_label, color, title) in FIG_META.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df_cleaned[col], bins=10, color=color, edgecolor="white",
                alpha=0.85)
        ax.set_title(f"{fig_label} — {title}", fontsize=12)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        mean_val = df_cleaned[col].mean()
        ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.2,
                   label=f"Mean = {mean_val:.2f}")
        ax.legend()

        fname = f"{fig_label.lower().replace('. ', '').replace(' ', '_')}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.show()
        print(f"  {fig_label} saved as {fname}")
        print(f"  Description: {DESCRIPTIONS[col]}")


# =============================================================================
# CHAPTER 4: NORMALIZE THE DATA
# =============================================================================

def normalize_minmax(df_cleaned: pd.DataFrame) -> pd.DataFrame:
    """Apply Min-Max normalization to all numeric columns (Fig. 11)."""
    print("\n" + "=" * 60)
    print("CHAPTER 4: MIN-MAX NORMALIZATION")
    print("=" * 60)

    df_norm = df_cleaned.copy()
    for col in NUMERIC_COLS:
        col_min = df_cleaned[col].min()
        col_max = df_cleaned[col].max()
        df_norm[col] = (df_cleaned[col] - col_min) / (col_max - col_min)

    print(f"\nFig. 11 — Normalized dataset (first 20 rows):\n"
          f"{df_norm.head(20).to_string(index=False)}")
    return df_norm


# =============================================================================
# CHAPTER 5: RELATIONSHIPS WITH FERTILITY RATE
# =============================================================================

SCATTER_META = {
    "Agriculture"     : ("Fig. 12", "darkorange",   "Fertility vs Agriculture"),
    "Education"       : ("Fig. 13", "crimson",      "Fertility vs Education"),
    "Examination"     : ("Fig. 14", "mediumpurple", "Fertility vs Examination"),
    "Catholic"        : ("Fig. 15", "goldenrod",    "Fertility vs Catholics"),
    "Infant.Mortality": ("Fig. 16", "teal",         "Fertility vs Infant Mortality"),
}


def explore_relationships(df_cleaned: pd.DataFrame) -> None:
    """
    Scatter plots with regression trend lines for each socioeconomic
    indicator vs Fertility (Fig. 12–16).
    """
    print("\n" + "=" * 60)
    print("CHAPTER 5: RELATIONSHIPS WITH FERTILITY RATE")
    print("=" * 60)

    for col, (fig_label, color, title) in SCATTER_META.items():
        x = df_cleaned[col].values
        y = df_cleaned["Fertility"].values

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_sq = r_value ** 2
        direction = "Positive" if slope > 0 else "Negative"

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x, y, color=color, alpha=0.75, edgecolors="white", s=60,
                   label="Municipalities")

        x_line = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_line, slope * x_line + intercept, color="black",
                linewidth=1.5, label=f"Regression line\n"
                                     f"R²={r_sq:.3f}, p={p_value:.4f}")

        ax.set_title(f"{fig_label} — {title}", fontsize=12)
        ax.set_xlabel(col)
        ax.set_ylabel("Fertility")
        ax.legend()

        fname = f"{fig_label.lower().replace('. ', '').replace(' ', '_')}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.show()

        print(f"\n  {fig_label}: {title}")
        print(f"    Slope  : {slope:.4f}  ({direction} correlation)")
        print(f"    R²     : {r_sq:.4f}")
        print(f"    p-value: {p_value:.4f}")
        print(f"    Saved  : {fname}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Chapter 1
    df = load_data(DATA_PATH)
    describe_dataset(df)

    # Chapter 2
    df_cleaned = clean_data(df)

    # Chapter 3
    visualize_distributions(df_cleaned)

    # Chapter 4
    df_normalized = normalize_minmax(df_cleaned)

    # Chapter 5
    explore_relationships(df_cleaned)

    print("\n" + "=" * 60)
    print("Analysis complete. All figures saved to the working directory.")
    print("=" * 60)


if __name__ == "__main__":
    main()
