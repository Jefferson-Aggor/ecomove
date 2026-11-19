"""EcoMove sustainable mobility dashboard (COM7021, student ID 25138486).

Run:
    streamlit run ecomove_dashboard.py
The dashboard opens with the supplied dataset ('[5442] EcoMoveMobility.xlsx' or
'EcoMoveMobility.xlsx' in the same folder). A different workbook can be uploaded
from the Data menu; it is checked first (columns, data types, value ranges,
categories) and used only if every check passes.
Needs: pip install streamlit plotly pandas numpy statsmodels openpyxl

Contents of this file, in order:
    1. Settings          column groups, category orders, colours
    2. Data              read and validate the workbook, convert types,
                         add derived measures
    3. Calculations      filters, KPIs, group summaries, adjusted demand index,
                         insight sentences
    4. Page              styling, header with filters and targets, four KPIs,
                         nine tabs (Overview, Demand, Cities & modes, Reliability,
                         Finance, Customer, Accessibility, Sustainability,
                         Data & method)
"""
from __future__ import annotations

import html
import io
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import streamlit as st



# =============================================================================
# 1. SETTINGS
# =============================================================================
RANDOM_SEED = 42  # used by every bootstrap / resampling step


# Schema expected in the supplied workbook
ID_COL = "Record_ID"


DATE_COL = "Date"


CATEGORICAL_COLS = [
    "City", "Country", "Transport_Mode", "Route_Type", "Weather", "Event_Day",
]


# Volume (additive) measures - can be summed across records
VOLUME_COLS = [
    "Passengers", "Trips", "Ticket_Revenue_EUR", "Operating_Cost_EUR",
    "CO2_Saved_KG", "Energy_Consumption_KWh", "Accessibility_Complaints",
]


# Rate / score measures - must be averaged, never summed
RATE_COLS = ["Average_Delay_Minutes", "On_Time_Percentage", "Customer_Rating"]


NUMERIC_COLS = [
    "Passengers", "Trips", "Average_Delay_Minutes", "On_Time_Percentage",
    "Ticket_Revenue_EUR", "Operating_Cost_EUR", "CO2_Saved_KG",
    "Energy_Consumption_KWh", "Customer_Rating", "Accessibility_Complaints",
]


EXPECTED_COLUMNS = [ID_COL, DATE_COL] + CATEGORICAL_COLS + NUMERIC_COLS


# One measure per management lever, used in the correlation heatmap:
# demand, reliability, satisfaction and financial sustainability. Excluded measures
# duplicate one of these (other volume totals rho 0.82-0.99 with passengers;
# on-time % rho -0.97 with delay) or show no relationship with them.
DECISION_VARS = ["Passengers", "Average_Delay_Minutes", "Customer_Rating", "Revenue_Cost_Ratio"]


DECISION_LABELS = {"Passengers": "Passengers\n(demand)",
                   "Average_Delay_Minutes": "Average delay\n(reliability)",
                   "Customer_Rating": "Customer rating\n(satisfaction)",
                   "Revenue_Cost_Ratio": "Revenue / cost\n(finance)"}


# Plausible value ranges used by the validation checks (inclusive)
VALID_RANGES = {
    "Passengers": (0, None),
    "Trips": (1, None),
    "Average_Delay_Minutes": (0, None),
    "On_Time_Percentage": (0, 100),
    "Ticket_Revenue_EUR": (0, None),
    "Operating_Cost_EUR": (0, None),
    "CO2_Saved_KG": (0, None),
    "Energy_Consumption_KWh": (0, None),
    "Customer_Rating": (1, 5),
    "Accessibility_Complaints": (0, None),
}


# Category orders (fixed so that every chart sorts the same way)
MODE_ORDER = ["Metro", "Tram", "Bus", "Park and Ride", "Bike Share", "E-Scooter"]


MICROMOBILITY_MODES = ["Bike Share", "E-Scooter"]


WEATHER_ORDER = ["Clear", "Cloudy", "Windy", "Rain", "Heatwave", "Snow"]


ROUTE_ORDER = [
    "Central Route", "Commuter Corridor", "Airport Connector",
    "Tourist Zone", "University Area", "Suburban Feeder",
]


EVENT_ORDER = ["No", "Yes"]


WEEKDAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# Colour: Okabe-Ito colour-universal-design palette (colour-blind safe).
# Validated: worst adjacent CVD delta-E 15.8 (deuteranopia). Three hues fall
# below 3:1 contrast on white, so every chart carries legends/direct labels.
# Colour follows the entity (mode), never its rank, so it never changes
# between charts or when dashboard filters remove categories.
MODE_COLOURS = {
    "Metro": "#0072B2",          # blue
    "Tram": "#009E73",           # bluish green
    "Bus": "#D55E00",            # vermillion
    "Park and Ride": "#CC79A7",  # reddish purple
    "Bike Share": "#E69F00",     # orange
    "E-Scooter": "#56B4E9",      # sky blue
}


NEUTRAL = "#595959"      # axis text / reference lines


LIGHT_NEUTRAL = "#BFBFBF"


HIGHLIGHT = "#D55E00"


BASE = "#0072B2"

# =============================================================================
# 2. DATA: READ AND VALIDATE THE UPLOAD, CONVERT TYPES, ADD DERIVED MEASURES
# =============================================================================
class SchemaError(ValueError):
    """Raised when the workbook does not contain the expected columns."""


# Validation of an uploaded workbook (used by the dashboard)
# Fingerprint of the supplied assessment dataset: (rows, total passengers).
ORIGINAL_FINGERPRINT = (5000, 40_909_721)


ALLOWED_VALUES = {
    "Transport_Mode": MODE_ORDER,
    "Weather": WEATHER_ORDER,
    "Route_Type": ROUTE_ORDER,
    "Event_Day": EVENT_ORDER,
}


MIN_ROWS = 30


def read_workbook(content: bytes) -> pd.DataFrame:
    """Read an uploaded .xlsx file and return the sheet that holds the dataset.

    Column names are matched ignoring case, spaces and underscores, so
    'transport mode' is accepted as 'Transport_Mode'. Raises SchemaError with a
    plain-language message when the file cannot be used.
    """
    try:
        sheets = pd.read_excel(io.BytesIO(content), sheet_name=None, engine="openpyxl")
    except Exception as exc:  # not an Excel workbook, corrupt or password-protected
        raise SchemaError("The file could not be opened as an Excel workbook (.xlsx). "
                          f"Check that it is not corrupt or password-protected. Details: {exc}") from exc
    if not sheets:
        raise SchemaError("The workbook contains no worksheets.")

    key = lambda name: "".join(ch for ch in str(name).lower() if ch.isalnum())
    wanted = {key(c): c for c in EXPECTED_COLUMNS}
    best, best_missing = None, None
    for sheet in sheets.values():
        sheet = sheet.rename(columns={c: wanted[key(c)] for c in sheet.columns if key(c) in wanted})
        missing = [c for c in EXPECTED_COLUMNS if c not in sheet.columns]
        if best is None or len(missing) < len(best_missing):
            best, best_missing = sheet, missing
    if best_missing:
        raise SchemaError(f"The workbook is missing {len(best_missing)} of the {len(EXPECTED_COLUMNS)} "
                          "required columns: " + ", ".join(best_missing) + ".")
    return best


def validate_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Check that a workbook matches what the analysis expects.

    Returns (clean copy, errors, warnings). Any error means the dashboard must
    not use the data; warnings are shown but do not block it. Nothing is
    silently repaired apart from trimming spaces around text values.
    """
    df, errors, warnings = raw.copy(), [], []
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra:
        warnings.append(f"{len(extra)} extra column(s) were ignored: " + ", ".join(map(str, extra[:8])) + ".")
    df = df[EXPECTED_COLUMNS].dropna(how="all")
    if len(df) < MIN_ROWS:
        return df, [f"The file has {len(df)} data rows; at least {MIN_ROWS} are needed for the charts."], warnings

    # 1. missing values
    blanks = df.isna().sum()
    for col, n in blanks[blanks > 0].items():
        errors.append(f"{col}: {n:,} empty cell(s). Every column must be complete.")

    # 2. numeric columns must be numbers, within their valid range
    for col in NUMERIC_COLS:
        values = pd.to_numeric(df[col], errors="coerce")
        bad = int((values.isna() & df[col].notna()).sum())
        if bad:
            errors.append(f"{col}: {bad:,} value(s) are not numbers (for example '{df.loc[values.isna() & df[col].notna(), col].iloc[0]}').")
            continue
        df[col] = values
        if (values < 0).any():
            errors.append(f"{col}: {int((values < 0).sum()):,} negative value(s).")
    limits = {"On_Time_Percentage": (0, 100, "0 to 100"), "Customer_Rating": (1, 5, "1 to 5")}
    for col, (low, high, label) in limits.items():
        if pd.api.types.is_numeric_dtype(df[col]) and (~df[col].dropna().between(low, high)).any():
            errors.append(f"{col}: {int((~df[col].dropna().between(low, high)).sum()):,} value(s) outside {label}.")
    for col in ("Passengers", "Operating_Cost_EUR"):   # used as denominators for rates and ratios
        if pd.api.types.is_numeric_dtype(df[col]) and (df[col] == 0).any():
            errors.append(f"{col}: {int((df[col] == 0).sum()):,} zero value(s); rates cannot be calculated.")

    # 3. dates: Excel serial numbers or real dates, within a plausible period
    try:
        dates = convert_dates(df[DATE_COL].dropna())
        if dates.min().year < 2000 or dates.max().year > 2100:
            errors.append(f"Date: values run from {dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}, "
                          "outside the plausible range 2000-2100.")
    except Exception:
        errors.append("Date: values could not be read as dates (expected Excel dates or serial numbers).")

    # 4. categories the charts, colours and comparisons are built on
    for col in CATEGORICAL_COLS:
        if df[col].notna().any():
            df[col] = df[col].astype("string").str.strip()
    for col, allowed in ALLOWED_VALUES.items():
        unknown = sorted(set(df[col].dropna()) - set(allowed))
        if unknown:
            errors.append(f"{col}: unexpected value(s) " + ", ".join(f"'{u}'" for u in unknown[:6])
                          + f". Allowed: {', '.join(allowed)}.")

    # 5. things worth knowing that do not stop the dashboard
    duplicates = int(df[ID_COL].duplicated().sum())
    if duplicates:
        warnings.append(f"{duplicates:,} duplicate Record_ID value(s); duplicated records are counted twice.")
    if not errors and (len(df), int(df["Passengers"].sum())) != ORIGINAL_FINGERPRINT:
        warnings.append("This is not the dataset the report was based on (different rows or passenger total). "
                        "The dashboard will work, but its figures will differ from the report.")
    return df, errors, warnings


def convert_dates(series: pd.Series) -> pd.Series:
    """Convert a date column to datetime64.

    The workbook stores dates as Excel serial numbers (days since 1899-12-30,
    the origin Excel uses because of its 1900 leap-year bug). Values that are
    already datetimes or ISO strings are handled too, so the function is safe
    to reuse on a re-exported file.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="D", origin="1899-12-30")
    return pd.to_datetime(series, errors="raise")


def _ordered_categorical(series: pd.Series, order: list[str] | None) -> pd.Series:
    """Cast to category, using a fixed order when one is defined.

    Any value not in the expected order is kept (appended), never silently
    turned into NaN, so unexpected labels surface in the audit.
    """
    values = series.astype("string").str.strip()
    if order is None:
        return values.astype("category")
    extras = sorted(set(values.dropna()) - set(order))
    return pd.Categorical(values, categories=order + extras, ordered=False)


def prepare_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Return a typed copy of the raw data and a log of what was changed."""
    df = raw.copy()
    log: list[dict] = []

    before = str(df[DATE_COL].dtype)
    df[DATE_COL] = convert_dates(df[DATE_COL])
    log.append({
        "step": "Convert Date",
        "columns": DATE_COL,
        "change": f"{before} (Excel serial) -> datetime64",
        "rationale": "Enables time-series grouping; serial numbers are not readable dates.",
    })

    orders = {
        "Transport_Mode": MODE_ORDER,
        "Weather": WEATHER_ORDER,
        "Route_Type": ROUTE_ORDER,
        "Event_Day": EVENT_ORDER,
    }
    for col in CATEGORICAL_COLS:
        df[col] = _ordered_categorical(df[col], orders.get(col))
    log.append({
        "step": "Cast categoricals",
        "columns": ", ".join(CATEGORICAL_COLS),
        "change": "text -> category (whitespace stripped; fixed display order)",
        "rationale": "Correct semantics, ~90% less memory, consistent ordering in every chart.",
    })

    df["Is_Event_Day"] = df["Event_Day"].eq("Yes")
    log.append({
        "step": "Boolean event flag",
        "columns": "Is_Event_Day",
        "change": "Yes/No -> True/False (original column kept)",
        "rationale": "Simplifies filtering and group comparisons.",
    })

    int_cols = ["Passengers", "Trips", "Accessibility_Complaints"]
    df[int_cols] = df[int_cols].astype("int64")
    log.append({
        "step": "Confirm numeric types",
        "columns": ", ".join(NUMERIC_COLS),
        "change": "counts held as int64, measures as float64 (no values altered)",
        "rationale": "Counts cannot be fractional; guards against silent text-to-number issues.",
    })

    log.append({
        "step": "Rows removed / values altered",
        "columns": "-",
        "change": "None (0 missing, 0 duplicates, 0 out-of-range values)",
        "rationale": "IQR 'outliers' are legitimate high-volume Metro records; kept, "
                     "skew handled with medians, log scales and rank-based tests.",
    })
    return df, log


def safe_divide(numerator, denominator, scale: float = 1.0):
    """Divide element-wise, returning NaN (not inf) where the denominator is 0."""
    num = np.asarray(numerator, dtype="float64")
    den = np.asarray(denominator, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den == 0, np.nan, num / den) * scale
    if isinstance(numerator, pd.Series):
        return pd.Series(out, index=numerator.index)
    return out


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add financial, efficiency, accessibility, environmental and calendar fields."""
    out = df.copy()

    # Financial
    out["Operating_Surplus_EUR"] = out["Ticket_Revenue_EUR"] - out["Operating_Cost_EUR"]
    out["Revenue_Cost_Ratio"] = safe_divide(out["Ticket_Revenue_EUR"], out["Operating_Cost_EUR"])
    out["Is_Loss_Making"] = out["Operating_Surplus_EUR"] < 0
    out["Revenue_per_Passenger"] = safe_divide(out["Ticket_Revenue_EUR"], out["Passengers"])
    out["Cost_per_Passenger"] = safe_divide(out["Operating_Cost_EUR"], out["Passengers"])

    # Accessibility, normalised by demand (raw counts mostly reflect volume)
    out["Complaints_per_10k_Pax"] = safe_divide(
        out["Accessibility_Complaints"], out["Passengers"], scale=10_000)

    # Environmental efficiency
    out["CO2_Saved_per_Passenger_KG"] = safe_divide(out["CO2_Saved_KG"], out["Passengers"])
    out["Energy_per_Passenger_KWh"] = safe_divide(out["Energy_Consumption_KWh"], out["Passengers"])

    # Operational. NB: for Bike Share / E-Scooter 'Trips' equals 'Passengers'
    # (one rental = one trip), so this is only comparable within mass transit.
    out["Passengers_per_Trip"] = safe_divide(out["Passengers"], out["Trips"])

    # Grouping and calendar fields
    out["Mode_Group"] = np.where(
        out["Transport_Mode"].isin(MICROMOBILITY_MODES), "Micromobility", "Mass transit")
    out["Year"] = out[DATE_COL].dt.year
    out["Month"] = out[DATE_COL].dt.month
    out["Year_Month"] = out[DATE_COL].dt.to_period("M").dt.to_timestamp()
    out["Quarter"] = out[DATE_COL].dt.to_period("Q").astype(str)
    out["Weekday"] = pd.Categorical(
        out[DATE_COL].dt.strftime("%a"), categories=WEEKDAY_ORDER)
    out["Is_Weekend"] = out[DATE_COL].dt.day_of_week >= 5
    return out


def group_ratio(df: pd.DataFrame, by, numerator: str, denominator: str,
                scale: float = 1.0) -> pd.Series:
    """Volume-weighted ratio for groups: sum(numerator) / sum(denominator)."""
    sums = df.groupby(by, observed=True)[[numerator, denominator]].sum()
    return safe_divide(sums[numerator], sums[denominator], scale=scale)

# =============================================================================
# 3. CALCULATIONS: FILTERS, KPIS, SUMMARIES, INSIGHTS
# =============================================================================
# Filters
@dataclass
class FilterState:
    """The user's current selection. Empty list = 'all' for that dimension."""
    start: date | None = None
    end: date | None = None
    countries: list[str] = field(default_factory=list)
    cities: list[str] = field(default_factory=list)
    modes: list[str] = field(default_factory=list)
    routes: list[str] = field(default_factory=list)
    weather: list[str] = field(default_factory=list)
    event: str = "All"  # "All" | "Event days only" | "Non-event days only"

    def active_count(self) -> int:
        dims = [self.countries, self.cities, self.modes, self.routes, self.weather]
        return sum(bool(d) for d in dims) + (self.event != "All")


def apply_filters(df: pd.DataFrame, f: FilterState) -> pd.DataFrame:
    """Return the subset of records matching every active filter."""
    mask = pd.Series(True, index=df.index)
    if f.start is not None:
        mask &= df[DATE_COL] >= pd.Timestamp(f.start)
    if f.end is not None:
        mask &= df[DATE_COL] <= pd.Timestamp(f.end)
    for col, chosen in (("Country", f.countries), ("City", f.cities),
                        ("Transport_Mode", f.modes), ("Route_Type", f.routes),
                        ("Weather", f.weather)):
        if chosen:
            mask &= df[col].isin(chosen)
    if f.event == "Event days only":
        mask &= df["Is_Event_Day"]
    elif f.event == "Non-event days only":
        mask &= ~df["Is_Event_Day"]
    return df.loc[mask]


def cities_for(df: pd.DataFrame, countries: list[str]) -> list[str]:
    """City options restricted to the selected countries (cascading filter)."""
    sub = df if not countries else df[df["Country"].isin(countries)]
    return sorted(sub["City"].astype(str).unique())


# KPIs
KPI_SPECS = [
    # key, label, kind ('sum' or 'mean'), help text
    ("passengers", "Passengers (in records)", "sum",
     "Sum of passengers across the selected records. The data are service records, not a full count "
     "of all journeys, so totals depend on how many records are selected."),
    ("trips", "Trips (in records)", "sum",
     "Sum of trips. For Bike Share / E-Scooter Trips always equals Passengers, so a trip there appears "
     "to mean one rental; for other modes it appears to mean one vehicle journey."),
    ("revenue", "Ticket revenue", "sum", "Sum of ticket revenue (EUR). Excludes subsidies."),
    ("cost", "Operating cost", "sum", "Sum of operating cost (EUR)."),
    ("surplus", "Operating surplus", "sum", "Revenue minus operating cost (EUR). Negative = deficit."),
    ("delay", "Average delay", "mean", "Mean of record-level average delay (minutes)."),
    ("on_time", "On-time performance", "mean", "Mean on-time percentage across records."),
    ("co2", "CO2 saved (in records)", "sum", "Sum of CO2 saved across the selected records (tonnes)."),
    ("complaints", "Accessibility complaints", "sum", "Total complaints; rate per 10k passengers shown below."),
    ("rating", "Customer rating", "mean", "Mean customer rating across records (1-5 scale)."),
    ("revenue_cost", "Cost recovery", "ratio",
     "Ticket revenue / operating cost, pooled over the selected records. 1.0 = break-even. Excludes subsidies."),
    ("co2_per_pax", "CO2 saved per passenger", "ratio",
     "CO2 saved (kg) / passengers, pooled. Ignores trip length."),
    ("complaint_rate", "Complaints per 10k passengers", "ratio",
     "Accessibility complaints / passengers x 10,000, pooled."),
]


# The four headline KPIs in the dashboard's KPI strip, one per management decision
# in the brief that a single number can track:
#   delay          service reliability; rating moves with delay (Spearman -0.81, and
#                  -0.63 to -0.72 within every mode), so rating is shown in its chip
#   revenue_cost   investment priorities; 1.0 = break-even, comparable across filters
#   co2_per_pax    sustainability performance; total tonnes shown in its chip
#   complaint_rate accessibility improvements; total complaints shown in its chip
# All four are averages or rates, so a filtered selection can be compared with the
# network. Passenger totals depend on how many records are selected, so demand
# (route planning) is shown in the scope line and the Demand and Cities tabs instead.
HEADLINE_KPIS = ["delay", "revenue_cost", "co2_per_pax", "complaint_rate"]


def compute_kpis(df: pd.DataFrame) -> dict[str, float]:
    """Headline KPIs for a (filtered) dataset. Returns NaN-safe values."""
    if df.empty:
        return {k: np.nan for k, *_ in KPI_SPECS} | {"records": 0}
    pax = float(df["Passengers"].sum())
    revenue = float(df["Ticket_Revenue_EUR"].sum())
    cost = float(df["Operating_Cost_EUR"].sum())
    complaints = float(df["Accessibility_Complaints"].sum())
    return {
        "records": len(df),
        "passengers": pax,
        "trips": float(df["Trips"].sum()),
        "revenue": revenue,
        "cost": cost,
        "surplus": revenue - cost,
        "delay": float(df["Average_Delay_Minutes"].mean()),
        "on_time": float(df["On_Time_Percentage"].mean()),
        "rating": float(df["Customer_Rating"].mean()),
        "co2": float(df["CO2_Saved_KG"].sum()) / 1000,  # tonnes
        "complaints": complaints,
        "complaint_rate": complaints / pax * 10_000 if pax else np.nan,
        "revenue_cost": revenue / cost if cost else np.nan,
        "co2_per_pax": float(df["CO2_Saved_KG"].sum()) / pax if pax else np.nan,  # kg
    }


def format_number(value: float, kind: str = "count") -> str:
    """Human-readable KPI formatting (EUR m / k, thousands separators)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if kind == "eur":
        sign = "-" if value < 0 else ""
        a = abs(value)
        return f"{sign}€{a / 1e6:,.2f}m" if a >= 1e6 else f"{sign}€{a / 1e3:,.1f}k"
    if kind == "minutes":
        return f"{value:.2f} min"
    if kind == "rating":
        return f"{value:.2f} / 5"
    if kind == "percent":
        return f"{value:.1f}%"
    if kind == "tonnes":
        return f"{value:,.0f} t"
    if kind == "ratio":
        return f"{value:.3f}"
    if kind == "kg":
        return f"{value:.3f} kg"
    if kind == "rate":
        return f"{value:.1f}"
    if value >= 1e6:
        return f"{value / 1e6:,.2f}m"
    return f"{value:,.0f}"


KPI_FORMATS = {"passengers": "count", "trips": "count", "revenue": "eur", "cost": "eur",
               "surplus": "eur", "delay": "minutes", "on_time": "percent", "co2": "tonnes",
               "complaints": "count", "rating": "rating", "revenue_cost": "ratio",
               "co2_per_pax": "kg", "complaint_rate": "rate"}


# Aggregations used by the charts
def _pooled(g, num: str, den: str, scale: float = 1.0) -> pd.Series:
    s = g[[num, den]].sum()
    return safe_divide(s[num], s[den], scale)


def group_summary(df: pd.DataFrame, by) -> pd.DataFrame:
    """One row per group with every KPI the dashboard displays."""
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(by, observed=True)
    sums = g[["Passengers", "Trips", "Ticket_Revenue_EUR", "Operating_Cost_EUR",
              "CO2_Saved_KG", "Energy_Consumption_KWh", "Accessibility_Complaints"]].sum()
    out = pd.DataFrame({
        "Records": g.size(),
        "Passengers": sums["Passengers"],
        "Mean passengers per record": g["Passengers"].mean(),
        "Revenue (EUR)": sums["Ticket_Revenue_EUR"],
        "Operating cost (EUR)": sums["Operating_Cost_EUR"],
        "Surplus (EUR)": sums["Ticket_Revenue_EUR"] - sums["Operating_Cost_EUR"],
        "Revenue / cost": safe_divide(sums["Ticket_Revenue_EUR"], sums["Operating_Cost_EUR"]),
        "Mean delay (min)": g["Average_Delay_Minutes"].mean(),
        "On-time (%)": g["On_Time_Percentage"].mean(),
        "Mean rating": g["Customer_Rating"].mean(),
        "Complaints": sums["Accessibility_Complaints"],
        "Complaints per 10k pax": safe_divide(sums["Accessibility_Complaints"], sums["Passengers"], 10_000),
        "CO2 saved (t)": sums["CO2_Saved_KG"] / 1000,
        "CO2 per passenger (kg)": safe_divide(sums["CO2_Saved_KG"], sums["Passengers"]),
        "Energy per passenger (kWh)": safe_divide(sums["Energy_Consumption_KWh"], sums["Passengers"]),
    })
    return out.reset_index()


def monthly_by_mode(df: pd.DataFrame, measure: str = "mean") -> pd.DataFrame:
    """Monthly passengers per mode: mean per record (default) or total."""
    agg = "mean" if measure == "mean" else "sum"
    return (df.groupby(["Year_Month", "Transport_Mode"], observed=True)["Passengers"]
            .agg([agg, "size"]).rename(columns={agg: "Passengers", "size": "Records"})
            .reset_index())


def monthly_financials(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Year_Month")
    out = g[["Ticket_Revenue_EUR", "Operating_Cost_EUR", "CO2_Saved_KG"]].sum()
    out["Surplus (EUR)"] = out["Ticket_Revenue_EUR"] - out["Operating_Cost_EUR"]
    out["Records"] = g.size()
    out["CO2 saved per record (kg)"] = out["CO2_Saved_KG"] / out["Records"]
    return out.reset_index()


def pivot_metric(df: pd.DataFrame, rows: str, cols: str, metric: str) -> pd.DataFrame:
    """Two-way table of a pooled or mean metric (for heatmaps)."""
    if metric == "complaint_rate":
        t = df.groupby([rows, cols], observed=True).apply(
            lambda g: _pooled(g, "Accessibility_Complaints", "Passengers", 10_000), include_groups=False)
    else:
        t = df.groupby([rows, cols], observed=True)[metric].mean()
    return t.unstack(cols)


def weekday_profile(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("Weekday", observed=True)["Passengers"].agg(["mean", "size"])
            .rename(columns={"mean": "Mean passengers per record", "size": "Records"})
            .reindex(WEEKDAY_ORDER).dropna().reset_index())


def event_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Mean passengers / delay / rating per mode on event vs non-event days."""
    t = (df.groupby(["Transport_Mode", "Event_Day"], observed=True)
         [["Passengers", "Average_Delay_Minutes", "Customer_Rating"]].mean().reset_index())
    t["Event day"] = t["Event_Day"].map({"Yes": "Event day", "No": "Non-event day"})
    return t


# Plain-language insight sentences (computed from the current selection)
def _extreme(summary: pd.DataFrame, key: str, col: str, highest: bool = True):
    s = summary.dropna(subset=[col])
    if s.empty:
        return None, np.nan
    row = s.loc[s[col].idxmax() if highest else s[col].idxmin()]
    return row[key], row[col]


def insights(df: pd.DataFrame) -> dict[str, str]:
    """One sentence per dashboard section, generated from the filtered data.
    Comparisons need at least two groups; otherwise a neutral note is returned."""
    out: dict[str, str] = {}
    if df.empty:
        return out
    by_mode = group_summary(df, "Transport_Mode")
    by_weather = group_summary(df, "Weather")
    by_route = group_summary(df, "Route_Type")

    weekend = df.groupby("Is_Weekend")["Passengers"].mean()
    if {True, False} <= set(weekend.index):
        out["demand"] = (f"Weekend records carry {100 * (1 - weekend[True] / weekend[False]):.0f}% fewer "
                         "passengers on average than weekday records in this selection.")
    if len(by_mode) > 1:
        m, v = _extreme(by_mode, "Transport_Mode", "Mean delay (min)")
        m2, v2 = _extreme(by_mode, "Transport_Mode", "Mean delay (min)", highest=False)
        out["reliability_mode"] = (f"{m} has the longest average delay ({v:.1f} min); "
                                   f"{m2} the shortest ({v2:.1f} min).")
    if len(by_weather) > 1:
        w, v = _extreme(by_weather, "Weather", "Mean delay (min)")
        out["reliability_weather"] = f"Delays are longest in {w.lower()} conditions ({v:.1f} min on average)."
    if len(by_mode) > 1:
        losers = by_mode[by_mode["Surplus (EUR)"] < 0]
        if not losers.empty:
            names = ", ".join(losers["Transport_Mode"].astype(str))
            out["finance"] = (f"Loss-making modes in this selection: {names} "
                              f"(combined deficit {format_number(abs(losers['Surplus (EUR)'].sum()), 'eur')}).")
        else:
            out["finance"] = "Every mode in this selection covers its operating costs."
    rho = df["Average_Delay_Minutes"].corr(df["Customer_Rating"], method="spearman")
    if not np.isnan(rho):
        out["customer"] = (f"Delay and customer rating are {'strongly' if abs(rho) >= 0.5 else 'moderately' if abs(rho) >= 0.3 else 'weakly'} "
                           f"related in this selection (Spearman rho = {rho:.2f}); longer delays go with lower ratings.")
    if len(by_route) > 1:
        r, v = _extreme(by_route, "Route_Type", "Complaints per 10k pax")
        out["accessibility"] = f"{r} routes have the highest accessibility complaint rate ({v:.1f} per 10,000 passengers)."
    if len(by_mode) > 1:
        m, v = _extreme(by_mode, "Transport_Mode", "CO2 per passenger (kg)")
        m2, t = _extreme(by_mode, "Transport_Mode", "CO2 saved (t)")
        out["sustainability"] = (f"{m2} saves the most CO2 in total ({t:,.0f} t); "
                                 f"{m} saves the most per passenger ({v:.2f} kg).")
    return out


def event_uplift(df: pd.DataFrame) -> pd.DataFrame:
    """Per mode: mean passengers on event vs non-event days and % uplift.
    Modes lacking either day type are dropped (no comparison possible)."""
    t = (df.groupby(["Transport_Mode", "Event_Day"], observed=True)["Passengers"].mean()
         .unstack("Event_Day"))
    if not {"Yes", "No"} <= set(t.columns):
        return pd.DataFrame(columns=["Transport_Mode", "Passengers (no event)", "Passengers (event)",
                                     "Passenger uplift (%)"])
    out = pd.DataFrame({"Passengers (no event)": t["No"], "Passengers (event)": t["Yes"]}).dropna()
    out["Passenger uplift (%)"] = 100 * (out["Passengers (event)"] / out["Passengers (no event)"] - 1)
    order = [m for m in MODE_ORDER if m in out.index]
    return out.reindex(order).reset_index()


# Demand index adjusted for service mix (same method as the report's Figure 8)
MIX_FORMULA = ("np.log(Passengers) ~ C(Transport_Mode) + C(Route_Type) + C(Weather)"
               " + Is_Event_Day + C(Month) + C(Weekday)")


def adjusted_demand_index(df: pd.DataFrame, level: str = "City") -> pd.DataFrame:
    """Observed / expected passengers for each city (or country).

    'Expected' comes from a model of log(passengers) on mode, route type, weather,
    event day, month and weekday, but not place, so it reflects each place's mix of
    services. 1.10 means 10% more passengers than that mix predicts. Returns an
    empty table if the model cannot be fitted (for example, too few records).
    """
    try:
        model = smf.ols(MIX_FORMULA, data=df).fit()
    except Exception:
        return pd.DataFrame(columns=[level, "Adjusted index", "Unadjusted index"])
    smear = np.mean(np.exp(model.resid))          # back-transform from the log scale
    d = df.assign(Expected=np.exp(model.fittedvalues) * smear)
    g = d.groupby(level, observed=True)
    out = pd.DataFrame({"Adjusted index": g["Passengers"].sum() / g["Expected"].sum(),
                        "Unadjusted index": g["Passengers"].mean() / df["Passengers"].mean()})
    return out.sort_values("Adjusted index", ascending=False).reset_index()

# =============================================================================
# 4. PAGE
# =============================================================================
# The card styling assumes Streamlit's light theme; set it here so the file needs no config.toml.
for _key, _value in {"theme.base": "light", "theme.primaryColor": "#0072B2", "theme.backgroundColor": "#FFFFFF",
                     "theme.secondaryBackgroundColor": "#F4F6F8", "theme.textColor": "#262626"}.items():
    if st.get_option(_key) != _value:
        st._config.set_option(_key, _value)
        st.session_state["_theme_changed"] = True
if st.session_state.pop("_theme_changed", False):
    st.rerun()   # reload once so the browser receives the light theme

st.set_page_config(page_title="EcoMove | Mobility Performance", page_icon=":material/directions_bus:",
                   layout="wide", initial_sidebar_state="collapsed")

# --------------------------------------------------------------------------- #
# Design tokens
# --------------------------------------------------------------------------- #
ACCENT = "#D55E00"          # Okabe-Ito vermillion: brand accent, 'needs attention'
GOOD = "#0072B2"            # Okabe-Ito blue: improvement (colour-blind safe with ACCENT)
INK, MUTED, LINE = "#111827", "#6B7280", "#E7E9EE"
FONT = "Inter, 'Segoe UI', Arial, sans-serif"
PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                 "toImageButtonOptions": {"format": "png", "scale": 2}}
MIN_RECORDS_WARNING = 30

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, [data-testid="stMarkdownContainer"], button, input {{ font-family: {FONT}; }}
.stApp {{ background: #EEF0F3; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1480px; }}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{ display: none; }}

/* cards: every st.container(key="card_...") */
div[class*="st-key-card"] {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 18px;
    padding: 18px 20px 12px 20px; box-shadow: 0 1px 2px rgba(16,24,40,.04); }}
.card-head {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px; }}
.card-title {{ font-size: 18px; font-weight: 500; color: {INK}; }}
.card-sub {{ font-size: 13px; color: {MUTED}; margin-bottom: 4px; }}
.card-icon {{ color: #9CA3AF; font-size: 16px; }}

/* header */
.brand {{ display: flex; align-items: center; gap: 12px; }}
.logo {{ width: 42px; height: 42px; border-radius: 12px; background: #111827; color: #fff;
    display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 18px; }}
.brand-name {{ font-weight: 700; font-size: 20px; color: {INK}; letter-spacing: -.3px; }}
.page-title {{ font-size: 36px; font-weight: 500; color: {INK}; letter-spacing: -.8px; margin: 6px 0 0 0; }}
.scope {{ color: {MUTED}; font-size: 14px; margin: 2px 0 10px 0; }}

/* KPI strip */
.kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); }}
@media (max-width: 900px) {{ .kpi-grid {{ grid-template-columns: repeat(2, 1fr); row-gap: 14px; }}
    .kpi:nth-child(3) {{ border-left: none; padding-left: 4px; }} }}
.kpi {{ padding: 6px 18px 10px 18px; border-left: 1px solid {LINE}; }}
.kpi:first-child {{ border-left: none; padding-left: 4px; }}
.kpi-label {{ color: {MUTED}; font-size: 15px; cursor: help; }}
.kpi-value {{ color: {INK}; font-size: 36px; font-weight: 500; letter-spacing: -1px; line-height: 1.25;
    margin: 6px 0 8px 0; white-space: nowrap; }}
.chip {{ display: inline-flex; align-items: center; gap: 4px; font-size: 13px; font-weight: 500;
    padding: 2px 8px; border-radius: 6px; }}
.chip.good {{ background: #E6F1F8; color: {GOOD}; }}
.chip.bad {{ background: #FBEDE4; color: {ACCENT}; }}
.chip.neutral {{ background: #F1F2F4; color: #4B5563; }}

/* pill navigation (st.tabs); role-based selectors work across Streamlit versions */
div[role="tablist"] {{ background: #E3E6EA !important; border-radius: 14px; padding: 4px !important; gap: 2px !important;
    border: none !important; width: fit-content; max-width: 100%; overflow-x: auto; margin-bottom: 8px; }}
div[role="tablist"]:after, div[role="tablist"]:before {{ display: none !important; }}
[role="tab"] {{ border-radius: 10px !important; padding: 7px 16px !important; height: auto !important; margin: 0 !important;
    background: transparent !important; border: none !important; white-space: nowrap; }}
[role="tab"] p {{ font-size: 14px !important; color: #374151 !important; font-weight: 400; }}
[role="tab"]:hover p {{ color: {INK} !important; }}
[role="tab"][aria-selected="true"] {{ background: #FFFFFF !important; box-shadow: 0 1px 3px rgba(0,0,0,.10); }}
[role="tab"][aria-selected="true"] p {{ color: {INK} !important; font-weight: 600; }}
[role="tab"] .react-aria-SelectionIndicator, div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"]
    {{ display: none !important; }}
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"] {{ display: none; }}

/* upload page */
div[class*="st-key-card_upload"] {{ margin-top: 28px; padding: 26px 30px 18px 30px; }}
.upload-title {{ font-size: 24px; font-weight: 500; color: {INK}; letter-spacing: -.4px; }}
.upload-sub {{ font-size: 14px; color: {MUTED}; margin: 4px 0 10px 0; }}

/* insight banner */
.insight {{ display: flex; gap: 10px; align-items: flex-start; border: 1px solid {LINE}; border-radius: 12px;
    padding: 10px 14px; margin: 6px 0 10px 0; font-size: 14px; color: #374151; background: #FFFFFF; }}
.insight b {{ color: {GOOD}; }}
.insight .dot {{ width: 22px; height: 22px; border-radius: 6px; background: {GOOD}; color: #fff;
    display: flex; align-items: center; justify-content: center; font-size: 13px; flex: none; }}

/* goal tracking */
.goals {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 14px 0 6px 0; }}
.goal {{ position: relative; border-radius: 4px 14px 14px 14px; padding: 34px 14px 12px 14px; }}
.goal:before {{ content: ''; position: absolute; top: -9px; left: 0; width: 44%; height: 14px;
    border-radius: 10px 10px 0 0; background: inherit; }}
.goal .g-label {{ font-size: 13px; }}
.goal .g-value {{ font-size: 32px; font-weight: 500; letter-spacing: -.5px; }}
.goal.on {{ background: #DDF1E8; color: #00664D; }}
.goal.risk {{ background: #FFEFD2; color: #8A5A00; }}
.goal.off {{ background: {ACCENT}; color: #FFFFFF; }}
.target {{ margin: 12px 0 4px 0; }}
.target .t-name {{ font-size: 14px; color: {INK}; font-weight: 500; }}
.target .t-row {{ display: flex; justify-content: space-between; font-size: 12px; color: {MUTED}; margin-top: 4px; }}
.target .t-vals {{ display: flex; justify-content: space-between; font-size: 14px; color: {INK}; }}
.bar {{ position: relative; height: 4px; background: #E5E7EB; border-radius: 4px; margin: 6px 0; }}
.bar span {{ position: absolute; left: 0; top: 0; height: 4px; border-radius: 4px; }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Data: upload, validate, prepare
# --------------------------------------------------------------------------- #
FILTER_KEYS = ["f_dates", "f_countries", "f_cities", "f_modes", "f_routes", "f_weather", "f_event"]
BRAND = "<div class='brand'><div class='logo'>E</div><span class='brand-name'>EcoMove</span></div>"
COLUMN_RULES = {
    "Record_ID": "Text. A unique ID for each record.",
    "Date": "Excel date (or Excel serial number), between 2000 and 2100.",
    "City": "Text.", "Country": "Text.",
    "Transport_Mode": "One of: " + ", ".join(MODE_ORDER) + ".",
    "Route_Type": "One of: " + ", ".join(ROUTE_ORDER) + ".",
    "Weather": "One of: " + ", ".join(WEATHER_ORDER) + ".",
    "Event_Day": "Yes or No.",
    "Passengers": "Number above 0.", "Trips": "Number, 0 or more.",
    "Average_Delay_Minutes": "Number, 0 or more.", "On_Time_Percentage": "Number from 0 to 100.",
    "Ticket_Revenue_EUR": "Number, 0 or more.", "Operating_Cost_EUR": "Number above 0.",
    "CO2_Saved_KG": "Number, 0 or more.", "Energy_Consumption_KWh": "Number, 0 or more.",
    "Customer_Rating": "Number from 1 to 5.", "Accessibility_Complaints": "Number, 0 or more.",
}


def esc(text: str) -> str:
    return html.escape(str(text))


@st.cache_data(show_spinner="Checking the dataset...")
def load_dataset(content: bytes):
    """Uploaded file -> (prepared data or None, errors, warnings).

    The file is used only if every check passes, so the charts never run on
    data they were not designed for."""
    try:
        raw = read_workbook(content)
    except SchemaError as exc:
        return None, [str(exc)], []
    clean, errors, warnings = validate_dataset(raw)
    if errors:
        return None, errors, warnings
    try:
        prepared, _ = prepare_data(clean)
        return add_derived_features(prepared), [], warnings
    except Exception as exc:  # a problem the checks did not anticipate: explain, never crash
        return None, [f"The file passed the checks but could not be prepared: {exc}"], warnings


DEFAULT_FILES = ["[5442] EcoMoveMobility.xlsx", "EcoMoveMobility.xlsx"]   # supplied dataset, same folder


def bundled_dataset() -> dict | None:
    """The supplied workbook stored next to this file, if it is there."""
    for name in DEFAULT_FILES:
        path = Path(__file__).parent / name
        if path.exists():
            return {"name": name, "content": path.read_bytes()}
    return None


def clear_dataset() -> None:
    """Forget the current dataset and every filter that depends on it, then show the upload page."""
    for key in ["upload", *FILTER_KEYS]:
        st.session_state.pop(key, None)
    st.session_state["choose_file"] = True


def use_bundled() -> None:
    """Go back to the supplied dataset."""
    for key in FILTER_KEYS:
        st.session_state.pop(key, None)
    st.session_state["upload"] = bundled_dataset()
    st.session_state["choose_file"] = False


def upload_page() -> None:
    """Start page: ask for the workbook and report every problem found in it."""
    st.markdown(BRAND, unsafe_allow_html=True)
    _, middle, _ = st.columns([1, 2.2, 1])
    with middle, st.container(key="card_upload"):
        st.markdown("<div class='upload-title'>Upload an EcoMove workbook</div><div class='upload-sub'>"
                    "Upload an Excel workbook (.xlsx) with the same columns as the supplied dataset, for "
                    "example a newer extract. The file is checked first and is used only if it has the "
                    "columns and values the dashboard is built on.</div>", unsafe_allow_html=True)
        file = st.file_uploader("EcoMove workbook (.xlsx)", type=["xlsx"], key="uploader")
        if file is not None:
            content = file.getvalue()
            _, errors, warnings = load_dataset(content)
            if errors:
                shown = "".join(f"\n- {e}" for e in errors[:12])
                more = f"\n- ...and {len(errors) - 12} more." if len(errors) > 12 else ""
                st.error(f"**{file.name} cannot be used.** {len(errors)} problem(s) found:{shown}{more}")
                for w in warnings:
                    st.warning(w)
                st.caption("Correct the file, or choose a different one, and upload it again.")
            else:
                clear_dataset()
                st.session_state["upload"] = {"name": file.name, "content": content}
                st.session_state["choose_file"] = False
                st.rerun()
        if bundled_dataset() is not None:
            st.button("Use the supplied dataset instead", on_click=use_bundled, icon=":material/undo:")
        with st.expander("What the file must contain"):
            st.caption(f"{len(COLUMN_RULES)} columns with these exact names (capital letters, spaces and "
                       f"underscores are ignored), no empty cells, and at least {MIN_ROWS} rows. "
                       "Extra columns are ignored.")
            st.dataframe(pd.DataFrame(COLUMN_RULES.items(), columns=["Column", "Expected content"]),
                         hide_index=True, height=38 + 35 * len(COLUMN_RULES))
    st.stop()


if "upload" not in st.session_state and not st.session_state.get("choose_file"):
    st.session_state["upload"] = bundled_dataset()      # open on the supplied dataset
upload = st.session_state.get("upload")
if upload is None:
    upload_page()
data, load_errors, load_warnings = load_dataset(upload["content"])
if data is None:  # should not happen (checked on upload), but never show charts on bad data
    clear_dataset()
    upload_page()

MIN_DATE, MAX_DATE = data[DATE_COL].min().date(), data[DATE_COL].max().date()
DEFAULTS = {"f_dates": (MIN_DATE, MAX_DATE), "f_countries": [], "f_cities": [], "f_modes": [],
            "f_routes": [], "f_weather": [], "f_event": "All"}
TARGETS = {"t_ontime": 85.0, "t_rating": 4.2, "t_rc": 1.0, "t_complaints": 14.0}
for k, v in {**DEFAULTS, **TARGETS}.items():
    st.session_state.setdefault(k, v)


def reset_filters() -> None:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


def card_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"<div class='card-head'><span class='card-title'>{esc(title)}</span>"
                f"<span class='card-icon'>&#8599;</span></div>"
                + (f"<div class='card-sub'>{esc(subtitle)}</div>" if subtitle else ""),
                unsafe_allow_html=True)


def insight_box(text: str | None) -> None:
    if text:
        st.markdown(f"<div class='insight'><span class='dot'>i</span><span><b>Insight:</b> {esc(text)}</span></div>",
                    unsafe_allow_html=True)


def show(fig: go.Figure) -> None:
    # theme=None keeps the report's colour-blind-safe palette and reference lines
    st.plotly_chart(fig, config=PLOTLY_CONFIG, theme=None)


# --------------------------------------------------------------------------- #
# Header: brand, title, date range, filters, targets
# --------------------------------------------------------------------------- #
left, right = st.columns([9, 10], vertical_alignment="bottom")
with left:
    st.markdown(BRAND + "<div class='page-title'>Sustainable Mobility Performance</div>", unsafe_allow_html=True)
with right:
    c_date, c_filter, c_target, c_data = st.columns([2.1, 1.3, 1.25, 1.05], vertical_alignment="bottom")
    with c_date:
        dates = st.date_input("Date range", min_value=MIN_DATE, max_value=MAX_DATE, key="f_dates",
                              help=f"Data cover {MIN_DATE:%d %b %Y} to {MAX_DATE:%d %b %Y}.")
    active = sum(bool(st.session_state[k]) for k in
                 ["f_countries", "f_cities", "f_modes", "f_routes", "f_weather"]) + (st.session_state["f_event"] != "All")
    with c_filter:
        with st.popover(f"Filters ({active})" if active else "Filters", icon=":material/tune:", width="stretch"):
            countries = st.multiselect("Country", sorted(data["Country"].astype(str).unique()), key="f_countries")
            city_options = cities_for(data, countries)
            st.session_state["f_cities"] = [c for c in st.session_state["f_cities"] if c in city_options]
            cities = st.multiselect("City", city_options, key="f_cities",
                                    help="Options are limited to the selected countries.")
            modes = st.multiselect("Transport mode", MODE_ORDER, key="f_modes")
            routes = st.multiselect("Route type", ROUTE_ORDER, key="f_routes")
            weather = st.multiselect("Weather", WEATHER_ORDER, key="f_weather")
            event = st.radio("Event day", ["All", "Event days only", "Non-event days only"], key="f_event",
                             horizontal=True)
            st.button("Reset all filters", on_click=reset_filters, width="stretch", icon=":material/restart_alt:")
    with c_target:
        with st.popover("Targets", icon=":material/flag:", width="stretch"):
            st.caption("Management targets used in Goal tracking. They are illustrative assumptions, "
                       "not part of the dataset; adjust them to EcoMove's own targets.")
            st.number_input("On-time target (%)", 50.0, 100.0, step=0.5, key="t_ontime")
            st.number_input("Customer rating target", 1.0, 5.0, step=0.05, key="t_rating")
            st.number_input("Cost recovery target (revenue / cost)", 0.0, 2.0, step=0.05, key="t_rc")
            st.number_input("Max complaints per 10,000 passengers", 0.0, 60.0, step=0.5, key="t_complaints")

    with c_data:
        with st.popover("Data", icon=":material/database:", width="stretch"):
            st.markdown(f"**{upload['name']}**")
            st.caption(f"{len(data):,} records · {MIN_DATE:%d %b %Y} to {MAX_DATE:%d %b %Y} · "
                       f"{data['City'].nunique()} cities. All checks passed.")
            for w in load_warnings:
                st.warning(w)
            st.button("Upload a different file", on_click=clear_dataset, width="stretch", icon=":material/upload:")
            if bundled_dataset() is not None and upload["name"] not in DEFAULT_FILES:
                st.button("Back to the supplied dataset", on_click=use_bundled, width="stretch",
                          icon=":material/undo:")

start, end = (dates[0], dates[1]) if isinstance(dates, (tuple, list)) and len(dates) == 2 else (MIN_DATE, MAX_DATE)
filters = FilterState(start=start, end=end, countries=countries, cities=cities, modes=modes,
                        routes=routes, weather=weather, event=event)
df = apply_filters(data, filters)
baseline = compute_kpis(data)
kpis = compute_kpis(df)

chosen = [", ".join(x) for x in (countries, cities, modes, routes, weather) if x] + ([event] if event != "All" else [])
st.markdown(f"<div class='scope'>Showing <b>{len(df):,} of {len(data):,} records</b> · "
            f"{format_number(kpis['passengers'])} passengers in these records · {start:%d %b %Y} – "
            f"{end:%d %b %Y}{' · ' + esc(' · '.join(chosen)) if chosen else ' · all cities, modes and conditions'}</div>",
            unsafe_allow_html=True)
if isinstance(dates, (tuple, list)) and len(dates) == 1:
    st.info("Select an end date to apply the date filter; the full period is shown meanwhile.")
for w in load_warnings:
    st.info(w)
if df.empty:
    st.warning("No records match the current filters. Widen the selection or press **Reset all filters**.")
    st.stop()
if len(df) < MIN_RECORDS_WARNING:
    st.warning(f"Only {len(df)} records match. Treat comparisons with caution: small samples are unstable.")


# --------------------------------------------------------------------------- #
# KPI strip (four headline KPIs)
# --------------------------------------------------------------------------- #
def context_chip(key: str) -> tuple[str, str]:
    """Chip under a KPI. Unfiltered: the supporting figure for that KPI (rating
    and on-time % for delay, surplus for cost recovery, totals for the two
    rates). Filtered: difference from the whole network, coloured blue (better)
    or orange (worse)."""
    if kpis["records"] == baseline["records"]:
        return {
            "delay": f"on-time {kpis['on_time']:.1f}% · rating {kpis['rating']:.2f}",
            "revenue_cost": f"operating surplus {format_number(kpis['surplus'], 'eur')}",
            "co2_per_pax": f"{kpis['co2']:,.0f} t saved in total",
            "complaint_rate": f"{kpis['complaints']:,.0f} complaints in total",
        }[key], "neutral"

    def signed(diff: float, text: str, higher_is_better: bool, decimals: int) -> tuple[str, str]:
        if round(diff, decimals) == 0:  # no visible difference: no arrow, no colour
            return "same as network", "neutral"
        good = diff > 0 if higher_is_better else diff < 0
        return f"{'▲' if diff > 0 else '▼'} {text}", ("good" if good else "bad")

    diff = kpis[key] - baseline[key]
    return {
        "delay": lambda: signed(diff, f"{diff:+.2f} min vs network", False, 2),
        "revenue_cost": lambda: signed(diff, f"{diff:+.3f} vs network", True, 3),
        "co2_per_pax": lambda: signed(diff, f"{diff:+.3f} kg vs network", True, 3),
        "complaint_rate": lambda: signed(diff, f"{diff:+.1f} vs network", False, 1),
    }[key]()


SPECS = {k: (k, label, kind, help_text) for k, label, kind, help_text in KPI_SPECS}

with st.container(key="card_kpis"):
    cells = []
    for key in HEADLINE_KPIS:
        _, label, _, help_text = SPECS[key]
        chip, cls = context_chip(key)
        value = format_number(kpis[key], KPI_FORMATS[key])
        cells.append(f"<div class='kpi'><div class='kpi-label' title='{esc(help_text)}'>{esc(label)}</div>"
                     f"<div class='kpi-value'>{esc(value)}</div><span class='chip {cls}'>{esc(chip)}</span></div>")
    st.markdown(f"<div class='kpi-grid'>{''.join(cells)}</div>", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def cached_index(content: bytes, level: str) -> pd.DataFrame:
    """Adjusted demand index on all records of the loaded file (does not change with filters)."""
    full, _, _ = load_dataset(content)
    return adjusted_demand_index(full, level)


insight = insights(df)
by_mode = group_summary(df, "Transport_Mode")
by_city = group_summary(df, "City")
by_country = group_summary(df, "Country")
city_idx = cached_index(upload["content"], "City")
country_idx = cached_index(upload["content"], "Country")
shown_idx = city_idx[city_idx["City"].astype(str).isin(set(df["City"].astype(str)))]
if len(shown_idx) > 1:
    top, bottom = shown_idx.iloc[0], shown_idx.iloc[-1]
    def more_or_fewer(index: float) -> str:
        return f"{abs(100 * (index - 1)):.0f}% {'more' if index >= 1 else 'fewer'}"
    insight["cities"] = (f"After allowing for service mix, {top['City']} carries {more_or_fewer(top['Adjusted index'])} "
                         f"passengers than expected and {bottom['City']} {more_or_fewer(bottom['Adjusted index'])} "
                         "(all records).")
modes_present = [m for m in MODE_ORDER if m in set(by_mode["Transport_Mode"])]


# --------------------------------------------------------------------------- #
# Chart helpers
# --------------------------------------------------------------------------- #
def style(fig: go.Figure, height: int = 330, legend: bool = False) -> go.Figure:
    fig.update_layout(template="plotly_white", height=height, font=dict(family=FONT, size=12, color="#374151"),
                      margin=dict(l=8, r=8, t=10, b=8), paper_bgcolor="white", plot_bgcolor="white",
                      showlegend=legend, hoverlabel=dict(font_family=FONT, font_size=13),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""))
    fig.update_xaxes(showgrid=False, linecolor=LINE, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor="#F0F1F4", zeroline=False, tickfont=dict(color=MUTED))
    return fig


def mode_hbar(summary: pd.DataFrame, metric: str, title: str, fmt: str = ",.2f",
              ref: float | None = None, ref_label: str = "") -> go.Figure:
    s = summary.set_index("Transport_Mode").reindex(modes_present).reset_index()
    fig = go.Figure(go.Bar(y=s["Transport_Mode"].astype(str), x=s[metric], orientation="h",
                           marker=dict(color=[MODE_COLOURS[m] for m in s["Transport_Mode"]], cornerradius=4),
                           text=[format(v, fmt) for v in s[metric]], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}<br>" + title + ": %{x:" + fmt + "}<extra></extra>"))
    if ref is not None:
        fig.add_vline(x=ref, line_dash="dash", line_color=INK, line_width=1.2, layer="above",
                      annotation_text=ref_label, annotation_position="top")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title=title, showgrid=True, gridcolor="#F0F1F4",
                     range=[0, max(float(s[metric].max()), ref or 0) * 1.22])
    fig = style(fig, max(190, 80 + 40 * len(s)))
    if ref is not None:  # room for the target label above the plot
        fig.update_layout(margin=dict(t=30))
    return fig


def heatmap(table: pd.DataFrame, title: str, scale: str = "Oranges", fmt: str = ".1f") -> go.Figure:
    t = table[[c for c in MODE_ORDER if c in table.columns]]
    for order in (WEATHER_ORDER, ROUTE_ORDER):
        if set(t.index) <= set(order):
            t = t.reindex([o for o in order if o in t.index])
    labels = [str(c).replace(" and ", " and<br>").replace("Bike Share", "Bike<br>Share") for c in t.columns]
    fig = go.Figure(go.Heatmap(z=t.values, x=labels, y=[str(i) for i in t.index], colorscale=scale,
                               texttemplate="%{z:" + fmt + "}", textfont=dict(size=13), xgap=3, ygap=3,
                               colorbar=dict(title=dict(text=title, side="right"), thickness=10),
                               hovertemplate="%{y} / %{x}<br>" + title + ": %{z:" + fmt + "}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(side="top", tickangle=0)
    return style(fig, max(220, 110 + 44 * len(t)))


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tabs = st.tabs(["Overview", "Demand", "Cities & modes", "Reliability", "Finance", "Customer",
                "Accessibility", "Sustainability", "Data & method"])

# ============================== OVERVIEW =================================== #
with tabs[0]:
    c1, c2, c3 = st.columns([1.25, 1, 1])
    with c1, st.container(key="card_demand_mode"):
        card_header("Demand by transport mode", "Total passengers and share of demand")
        s = by_mode.set_index("Transport_Mode").reindex(modes_present).reset_index()
        share = 100 * s["Passengers"] / s["Passengers"].sum()
        top = float(s["Passengers"].max()) * 1.18
        fig = go.Figure()
        fig.add_bar(x=s["Transport_Mode"].astype(str), y=[top] * len(s), hoverinfo="skip",
                    marker=dict(color="white", line=dict(width=0), cornerradius=10,
                                pattern=dict(shape="/", fgcolor="#E5E7EB", size=8, solidity=0.25)))
        fig.add_bar(x=s["Transport_Mode"].astype(str), y=s["Passengers"],
                    marker=dict(color=[MODE_COLOURS[m] for m in s["Transport_Mode"]], cornerradius=10),
                    text=[format_number(v) for v in s["Passengers"]], textposition="auto",
                    textfont=dict(color="white", size=12), customdata=share,
                    hovertemplate="%{x}<br>%{y:,.0f} passengers<br>%{customdata:.1f}% of demand<extra></extra>")
        for x, y, sh in zip(s["Transport_Mode"].astype(str), s["Passengers"], share):
            fig.add_annotation(x=x, y=y, text=f"{sh:.1f}%", showarrow=False, yshift=14,
                               font=dict(size=11, color=INK), bgcolor="white", bordercolor=LINE, borderpad=3)
        fig.update_layout(barmode="overlay", bargap=0.18)
        fig.update_yaxes(visible=False, range=[0, top])
        show(style(fig, 330))
    with c2, st.container(key="card_trend"):
        card_header("Demand trend", "Mean passengers per record, weekdays vs weekends")
        t = (df.assign(Day=np.where(df["Is_Weekend"], "Weekend", "Weekday"))
             .groupby(["Year_Month", "Day"])["Passengers"].mean().unstack())
        fig = go.Figure()
        for col, colour, fill in (("Weekday", GOOD, "rgba(0,114,178,0.10)"),
                                  ("Weekend", "#7B61FF", "rgba(123,97,255,0.10)")):
            if col in t:
                fig.add_scatter(x=t.index, y=t[col], name=col, mode="lines", line=dict(color=colour, width=2,
                                shape="hv"), fill="tozeroy", fillcolor=fill,
                                hovertemplate="%{x|%b %Y}<br>" + col + ": %{y:,.0f} per record<extra></extra>")
        fig.update_xaxes(dtick="M6", tickformat="%b %Y")
        fig.update_yaxes(tickformat="~s")
        show(style(fig, 330, legend=True))
    with c3, st.container(key="card_surplus"):
        card_header("Operating surplus by mode", "Ticket revenue minus operating cost (EUR m)")
        s = by_mode.set_index("Transport_Mode").reindex(modes_present).reset_index()
        v = s["Surplus (EUR)"] / 1e6
        fig = go.Figure(go.Bar(
            y=s["Transport_Mode"].astype(str), x=v, orientation="h",
            text=[f"{'−' if x < 0 else '+'}€{abs(x):.2f}m" for x in v], textposition="outside", cliponaxis=False,
            textfont=dict(size=12, color=INK),
            marker=dict(color=[MODE_COLOURS[m] for m in s["Transport_Mode"]], cornerradius=4,
                        pattern=dict(shape=["/" if x < 0 else "" for x in v], fgcolor="white", solidity=0.3)),
            customdata=np.c_[s["Revenue / cost"]],
            hovertemplate="%{y}<br>Surplus: €%{x:,.2f}m<br>Revenue / cost: %{customdata[0]:.2f}<extra></extra>"))
        if (v < 0).any():
            i = int(v.idxmin())
            fig.add_annotation(x=v[i], y=str(s.loc[i, "Transport_Mode"]), text="largest deficit", showarrow=False,
                               xanchor="left", yanchor="bottom", yshift=14, font=dict(color="white", size=11),
                               bgcolor=INK, borderpad=4)
        fig.add_vline(x=0, line_color=INK, line_width=1)
        pad = 0.45 * max(float(v.max() - min(v.min(), 0)), 0.5)
        fig.update_xaxes(range=[min(v.min(), 0) - pad, max(v.max(), 0) + pad], showgrid=True, gridcolor="#F0F1F4")
        fig.update_yaxes(autorange="reversed", showgrid=False)
        show(style(fig, 330))

    c4, c5 = st.columns([1.75, 1])
    with c4, st.container(key="card_modes_table"):
        card_header("Transport mode profile", "The six modes: group, sampled records and performance; "
                                              "sparkline = monthly mean passengers per record")
        monthly = df.groupby(["Transport_Mode", "Year_Month"], observed=True)["Passengers"].mean()
        trend = {str(m): monthly.xs(m, level=0).round(1).tolist() for m in monthly.index.get_level_values(0).unique()}
        table = by_mode.assign(
            Group=np.where(by_mode["Transport_Mode"].isin(MICROMOBILITY_MODES), "Micromobility", "Mass transit"),
            Trend=[trend.get(str(m), []) for m in by_mode["Transport_Mode"]])
        st.dataframe(
            table, hide_index=True, height=38 + 35 * len(table),
            column_order=["Transport_Mode", "Group", "Records", "Passengers", "Revenue / cost", "Mean delay (min)",
                          "Mean rating", "Complaints per 10k pax", "Trend"],
            column_config={
                "Transport_Mode": st.column_config.TextColumn("Mode"),
                "Passengers": st.column_config.NumberColumn(format="localized"),
                "Revenue / cost": st.column_config.NumberColumn(format="%.2f"),
                "Mean delay (min)": st.column_config.NumberColumn("Delay (min)", format="%.1f"),
                "On-time (%)": st.column_config.NumberColumn(format="%.1f"),
                "Mean rating": st.column_config.NumberColumn("Rating", format="%.2f"),
                "Complaints per 10k pax": st.column_config.NumberColumn("Complaints /10k", format="%.1f"),
                "Trend": st.column_config.LineChartColumn("Demand trend", color=GOOD),
            })
        insight_box(" ".join(insight[k] for k in ("reliability_mode", "finance") if k in insight))
    with c5, st.container(key="card_goals"):
        tgt = {k: st.session_state[k] for k in TARGETS}
        card_header("Service goal tracking", f"City–mode services against the {tgt['t_ontime']:.1f}% on-time target")
        svc = df.groupby(["City", "Transport_Mode"], observed=True)["On_Time_Percentage"].mean()
        on = (svc >= tgt["t_ontime"]).mean() * 100
        risk = ((svc < tgt["t_ontime"]) & (svc >= tgt["t_ontime"] - 3)).mean() * 100
        off = 100 - on - risk
        st.markdown(f"<div class='goals'><div class='goal on'><div class='g-label'>On track</div>"
                    f"<div class='g-value'>{on:.0f}%</div></div><div class='goal risk'><div class='g-label'>"
                    f"At risk (within 3 pp)</div><div class='g-value'>{risk:.0f}%</div></div>"
                    f"<div class='goal off'><div class='g-label'>Off track</div><div class='g-value'>{off:.0f}%"
                    f"</div></div></div>", unsafe_allow_html=True)

        def target_bar(name: str, current: float, target: float, fmt: str, lower_is_better: bool = False) -> str:
            met = current <= target if lower_is_better else current >= target
            ratio = (target / current if lower_is_better else current / target) if current and target else 0
            colour = GOOD if met else ACCENT
            status = "met" if met else "not met"
            return (f"<div class='target'><div class='t-name'>{esc(name)}</div><div class='t-row'><span>Current"
                    f"</span><span>Target ({status})</span></div><div class='bar'><span style='width:"
                    f"{min(ratio, 1) * 100:.0f}%;background:{colour}'></span></div><div class='t-vals'><span>"
                    f"{format(current, fmt)}</span><span>{format(target, fmt)}</span></div></div>")
        st.markdown(target_bar("On-time performance (%)", kpis["on_time"], tgt["t_ontime"], ".1f")
                    + target_bar("Customer rating", kpis["rating"], tgt["t_rating"], ".2f")
                    + target_bar("Cost recovery (revenue / cost)", kpis["revenue_cost"], tgt["t_rc"], ".3f")
                    + target_bar("Accessibility complaints per 10k (max)", kpis["complaint_rate"],
                                 tgt["t_complaints"], ".1f", lower_is_better=True), unsafe_allow_html=True)
        st.caption("Targets are illustrative; change them under Targets.")

# =============================== DEMAND ==================================== #
with tabs[1]:
    insight_box(insight.get("demand"))
    with st.container(key="card_monthly"):
        measure = st.radio("Measure", ["Mean passengers per record", "Total passengers"], horizontal=True,
                           help="Totals depend on how many records were sampled each month; the mean per "
                                "record is the safer basis for trends.")
        mean = measure.startswith("Mean")
        card_header("Monthly demand by transport mode",
                    "Each panel has its own scale: compare shapes here and levels in Cities & modes"
                    + ("" if mean else ". Totals move with the number of sampled records (188-228 per month)."))
        m = monthly_by_mode(df, "mean" if mean else "sum")
        fig = px.line(m, x="Year_Month", y="Passengers", color="Transport_Mode", facet_col="Transport_Mode",
                      facet_col_wrap=3, facet_row_spacing=0.16, facet_col_spacing=0.06,
                      color_discrete_map=MODE_COLOURS, category_orders={"Transport_Mode": modes_present},
                      custom_data=["Records"], labels={"Passengers": "", "Year_Month": ""})
        fig.update_traces(line_width=2, hovertemplate="%{x|%b %Y}<br>%{y:,.0f}<br>Records: %{customdata[0]}"
                                                      "<extra>%{fullData.name}</extra>")
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font_size=13))
        fig.update_yaxes(matches=None, showticklabels=True, rangemode="tozero", tickformat="~s")
        fig.update_xaxes(dtick="M6", tickformat="%b %Y", title="")
        show(style(fig, 290 * int(np.ceil(len(modes_present) / 3)) + 30))
    c1, c2 = st.columns(2)
    with c1, st.container(key="card_weekday"):
        card_header("Demand by day of week", "Mean passengers per record; weekends highlighted")
        w = weekday_profile(df)
        fig = go.Figure(go.Bar(x=w["Weekday"].astype(str), y=w["Mean passengers per record"],
                               marker=dict(color=[GOOD if d not in ("Sat", "Sun") else "#8DB8DD"
                                                  for d in w["Weekday"]], cornerradius=6),
                               customdata=w["Records"], hovertemplate="%{x}<br>%{y:,.0f} per record<br>"
                                                                      "Records: %{customdata}<extra></extra>"))
        show(style(fig, 300))
    with c2, st.container(key="card_event"):
        card_header("Passenger uplift on event days", "Change in mean passengers per record by mode")
        u = event_uplift(df)
        if u.empty:
            st.caption("Both event and non-event days must be in the selection for this comparison.")
        else:
            fig = go.Figure(go.Bar(y=u["Transport_Mode"].astype(str), x=u["Passenger uplift (%)"], orientation="h",
                                   marker=dict(color=[MODE_COLOURS[m] for m in u["Transport_Mode"]], cornerradius=4),
                                   text=[f"{x:+.0f}%" for x in u["Passenger uplift (%)"]], textposition="outside",
                                   cliponaxis=False, hovertemplate="%{y}: %{x:+.1f}%<extra></extra>"))
            fig.update_yaxes(autorange="reversed", showgrid=False)
            fig.update_xaxes(range=[min(0, u["Passenger uplift (%)"].min() * 1.3),
                                    max(0, u["Passenger uplift (%)"].max()) * 1.25 + 1])
            show(style(fig, 300))

# ============================ CITIES & MODES ============================== #
with tabs[2]:
    insight_box(insight.get("cities"))
    c1, c2 = st.columns([3, 2])
    with c1, st.container(key="card_city_index"):
        card_header("City demand adjusted for service mix",
                    "Passengers compared with what each city's mix of modes, routes and conditions predicts "
                    "(1.0 = as expected); hollow dot = raw average")
        if shown_idx.empty:
            st.caption("The adjusted index could not be calculated for this file.")
        else:
            si = shown_idx.sort_values("Adjusted index")
            names = si["City"].astype(str)
            fig = go.Figure()
            fig.add_scatter(x=si["Unadjusted index"], y=names, mode="markers", name="Raw average (not adjusted)",
                            marker=dict(size=10, color="white", line=dict(color=MUTED, width=1.5)),
                            hovertemplate="%{y}<br>Raw index: %{x:.2f}<extra></extra>")
            fig.add_scatter(x=si["Adjusted index"], y=names, mode="markers", name="Adjusted for service mix",
                            marker=dict(size=12, color=GOOD),
                            hovertemplate="%{y}<br>Adjusted index: %{x:.2f}<extra></extra>")
            fig.add_scatter(x=np.maximum(si["Adjusted index"], si["Unadjusted index"]) + 0.015, y=names,
                            mode="text", text=[f"{100 * (v - 1):+.0f}%" for v in si["Adjusted index"]],
                            textposition="middle right", showlegend=False, hoverinfo="skip",
                            textfont=dict(color=INK, size=12))
            fig.add_vline(x=1, line_color=INK, line_width=1)
            fig.update_xaxes(showgrid=True, gridcolor="#F0F1F4", title="Demand index")
            show(style(fig, max(260, 70 + 30 * len(si)), legend=True))
            st.caption("Calculated once on all records, so the filters only choose which cities are shown.")
    with c2, st.container(key="card_mode_mean"):
        card_header("Demand per service", "Mean passengers per record by mode")
        show(mode_hbar(by_mode, "Mean passengers per record", "Mean passengers per record", ",.0f"))
    c3, c4 = st.columns([3, 2])
    with c3, st.container(key="card_city_mode"):
        card_header("Passengers by city and mode", "Total passengers in the selected records")
        t = df.groupby(["City", "Transport_Mode"], observed=True)["Passengers"].sum().reset_index()
        order = t.groupby("City", observed=True)["Passengers"].sum().sort_values().index.tolist()
        fig = px.bar(t, y="City", x="Passengers", color="Transport_Mode", orientation="h",
                     color_discrete_map=MODE_COLOURS, labels={"Transport_Mode": "Mode", "City": ""},
                     category_orders={"Transport_Mode": modes_present, "City": order[::-1]})
        fig.update_traces(marker_line_color="white", marker_line_width=1,
                          hovertemplate="%{y}<br>%{fullData.name}: %{x:,.0f}<extra></extra>")
        fig.update_xaxes(showgrid=True, gridcolor="#F0F1F4", tickformat="~s")
        show(style(fig, 440, legend=True))
    with c4, st.container(key="card_country_table"):
        card_header("Countries", "Adjusted demand index (all records) and service measures for the selection")
        ct = by_country.merge(country_idx[["Country", "Adjusted index"]].astype({"Country": str}),
                              on="Country", how="left") if not country_idx.empty else by_country
        st.dataframe(ct.sort_values("Adjusted index" if "Adjusted index" in ct else "Records", ascending=False),
                     hide_index=True, height=38 + 35 * len(ct),
                     column_order=["Country", "Adjusted index", "Mean delay (min)", "Mean rating",
                                   "Complaints per 10k pax"],
                     column_config={
                         "Adjusted index": st.column_config.NumberColumn("Demand index", format="%.2f"),
                         "Mean delay (min)": st.column_config.NumberColumn("Delay (min)", format="%.2f"),
                         "Mean rating": st.column_config.NumberColumn("Rating", format="%.2f"),
                         "Revenue / cost": st.column_config.NumberColumn(format="%.3f"),
                         "Complaints per 10k pax": st.column_config.NumberColumn("Complaints /10k", format="%.1f")})
        st.caption("Service measures differ little between countries; demand differs more.")
    with st.container(key="card_city_table"):
        card_header("City scorecard", "Demand index adjusted for service mix (all records) beside raw averages "
                                      "for the selected records")
        cs = by_city.astype({"City": str}).merge(city_idx[["City", "Adjusted index"]].astype({"City": str}),
                                                 on="City", how="left")
        st.dataframe(cs.sort_values("Adjusted index", ascending=False), hide_index=True,
                     height=38 + 35 * len(cs),
                     column_order=["City", "Records", "Adjusted index", "Mean passengers per record",
                                   "Mean delay (min)", "On-time (%)", "Mean rating", "Revenue / cost",
                                   "Complaints per 10k pax", "CO2 per passenger (kg)"],
                     column_config={
                         "Adjusted index": st.column_config.NumberColumn("Demand index (adjusted)", format="%.2f"),
                         "Mean passengers per record": st.column_config.ProgressColumn(
                             "Mean passengers per record (raw)", format="%.0f", min_value=0,
                             max_value=float(cs["Mean passengers per record"].max())),
                         "Mean delay (min)": st.column_config.NumberColumn(format="%.2f"),
                         "On-time (%)": st.column_config.NumberColumn(format="%.1f"),
                         "Mean rating": st.column_config.NumberColumn(format="%.2f"),
                         "Revenue / cost": st.column_config.NumberColumn(format="%.3f"),
                         "Complaints per 10k pax": st.column_config.NumberColumn(format="%.1f"),
                         "CO2 per passenger (kg)": st.column_config.NumberColumn(format="%.3f")})

# ============================== RELIABILITY =============================== #
with tabs[3]:
    insight_box(" ".join(insight[k] for k in ("reliability_mode", "reliability_weather") if k in insight))
    c1, c2 = st.columns(2)
    with c1, st.container(key="card_delay_box"):
        card_header("Delay distribution by weather", "Average delay per record; boxes = middle 50%")
        order = (df.groupby("Weather", observed=True)["Average_Delay_Minutes"].median()
                 .sort_values().index.astype(str).tolist())
        fig = px.box(df, x="Average_Delay_Minutes", y="Weather", color="Mode_Group", points=False,
                     category_orders={"Weather": order[::-1], "Mode_Group": ["Mass transit", "Micromobility"]},
                     color_discrete_map={"Mass transit": GOOD, "Micromobility": MODE_COLOURS["Bike Share"]},
                     labels={"Average_Delay_Minutes": "Average delay (minutes)", "Mode_Group": "", "Weather": ""})
        show(style(fig, 400, legend=True))
    with c2, st.container(key="card_delay_heat"):
        card_header("Mean delay by weather and mode", "Minutes; darker = longer")
        show(heatmap(pivot_metric(df, "Weather", "Transport_Mode", "Average_Delay_Minutes"), "Minutes"))
    c3, c4 = st.columns(2)
    with c3, st.container(key="card_ontime"):
        card_header("On-time performance by mode", "Mean on-time %, dashed line = target")
        show(mode_hbar(by_mode, "On-time (%)", "Mean on-time (%)", ".1f",
                       ref=st.session_state["t_ontime"], ref_label="target"))
    with c4, st.container(key="card_event_delay"):
        card_header("Delay on event days", "Mean delay per record, event vs other days")
        if df["Event_Day"].nunique() > 1:
            comp = event_comparison(df)
            fig = px.bar(comp, x="Transport_Mode", y="Average_Delay_Minutes", color="Event day", barmode="group",
                         category_orders={"Transport_Mode": modes_present,
                                          "Event day": ["Non-event day", "Event day"]},
                         color_discrete_map={"Non-event day": "#C9D3DE", "Event day": ACCENT},
                         labels={"Average_Delay_Minutes": "Mean delay (min)", "Transport_Mode": ""})
            fig.update_traces(marker_cornerradius=4)
            show(style(fig, 320, legend=True))
        else:
            st.caption("Both event and non-event days must be in the selection for this comparison.")

# ================================ FINANCE ================================= #
with tabs[4]:
    insight_box(insight.get("finance"))
    c1, c2 = st.columns(2)
    with c1, st.container(key="card_rc_mode"):
        card_header("Cost recovery by mode", "Ticket revenue / operating cost; dashed line = target")
        show(mode_hbar(by_mode, "Revenue / cost", "Revenue / operating cost", ".2f",
                       ref=st.session_state["t_rc"], ref_label="target"))
    with c2, st.container(key="card_rev_cost"):
        card_header("Monthly revenue and operating cost", "EUR million, sampled records")
        mf = monthly_financials(df)
        fig = go.Figure()
        fig.add_scatter(x=mf["Year_Month"], y=mf["Ticket_Revenue_EUR"] / 1e6, name="Ticket revenue",
                        line=dict(color=GOOD, width=2.2), hovertemplate="%{x|%b %Y}: €%{y:.2f}m<extra></extra>")
        fig.add_scatter(x=mf["Year_Month"], y=mf["Operating_Cost_EUR"] / 1e6, name="Operating cost",
                        line=dict(color=ACCENT, width=2.2, dash="dot"),
                        hovertemplate="%{x|%b %Y}: €%{y:.2f}m<extra></extra>")
        fig.update_xaxes(dtick="M3", tickformat="%b %Y")
        show(style(fig, 300, legend=True))
    with st.container(key="card_rc_city"):
        card_header("Cost recovery by city", "Revenue / operating cost; dashed line = break-even")
        s = by_city.sort_values("Revenue / cost")
        fig = go.Figure(go.Bar(y=s["City"].astype(str), x=s["Revenue / cost"], orientation="h",
                               marker=dict(color=GOOD, cornerradius=4), text=[f"{x:.3f}" for x in s["Revenue / cost"]],
                               textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.add_vline(x=1, line_dash="dash", line_color=INK, line_width=1.2, layer="above",
                      annotation_text="break-even", annotation_position="top")
        fig.update_xaxes(range=[max(0, s["Revenue / cost"].min() - 0.1), s["Revenue / cost"].max() + 0.08])
        st.caption("Revenue covers ticket income only; subsidies and other income are not in the dataset.")
        show(style(fig, 400))

# =============================== CUSTOMER ================================= #
with tabs[5]:
    insight_box(insight.get("customer"))
    c1, c2 = st.columns([3, 2])
    with c1, st.container(key="card_scatter"):
        card_header("Delay vs customer rating", "Each point is a record (up to 2,500); black line = overall trend")
        d = df.sample(2500, random_state=RANDOM_SEED) if len(df) > 2500 else df
        fig = px.scatter(d, x="Average_Delay_Minutes", y="Customer_Rating", color="Transport_Mode", opacity=0.55,
                         color_discrete_map=MODE_COLOURS, category_orders={"Transport_Mode": modes_present},
                         trendline="ols" if len(d) >= 10 else None, trendline_scope="overall",
                         trendline_color_override=INK, hover_data={"City": True, "Weather": True},
                         labels={"Average_Delay_Minutes": "Average delay (minutes)",
                                 "Customer_Rating": "Customer rating (1-5)", "Transport_Mode": "Mode"})
        fig.update_traces(marker=dict(size=6), selector=dict(mode="markers"))
        show(style(fig, 440, legend=True))
    with c2, st.container(key="card_rating_heat"):
        card_header("Mean rating by weather and mode", "1-5 scale; darker = higher")
        show(heatmap(pivot_metric(df, "Weather", "Transport_Mode", "Customer_Rating"), "Rating", "Blues", ".2f"))
        show(mode_hbar(by_mode, "Mean rating", "Mean customer rating (1-5)", ".2f",
                       ref=st.session_state["t_rating"], ref_label="target"))

# ============================= ACCESSIBILITY ============================== #
with tabs[6]:
    insight_box(insight.get("accessibility"))
    with st.container(key="card_access_heat"):
        card_header("Accessibility complaints per 10,000 passengers", "Route type x mode, pooled rates; darker = worse")
        show(heatmap(pivot_metric(df, "Route_Type", "Transport_Mode", "complaint_rate"), "Per 10k"))
        st.caption("Cells with few passengers, especially micromobility, can move sharply with small changes "
                   "in complaints.")
    c1, c2 = st.columns(2)
    with c1, st.container(key="card_access_mode"):
        card_header("Complaint rate by mode", "Complaints per 10,000 passengers; dashed line = maximum target")
        show(mode_hbar(by_mode, "Complaints per 10k pax", "Complaints per 10,000 passengers", ".1f",
                       ref=st.session_state["t_complaints"], ref_label="target"))
    with c2, st.container(key="card_access_city"):
        card_header("Complaint rate by city", "Complaints per 10,000 passengers")
        s = by_city.sort_values("Complaints per 10k pax")
        fig = go.Figure(go.Bar(y=s["City"].astype(str), x=s["Complaints per 10k pax"], orientation="h",
                               marker=dict(color=GOOD, cornerradius=4), textposition="outside", cliponaxis=False,
                               text=[f"{x:.1f}" for x in s["Complaints per 10k pax"]],
                               hovertemplate="%{y}: %{x:.1f} per 10k<extra></extra>"))
        fig.update_xaxes(range=[0, s["Complaints per 10k pax"].max() * 1.18])
        show(style(fig, 400))

# ============================ SUSTAINABILITY ============================== #
with tabs[7]:
    insight_box(insight.get("sustainability"))
    c1, c2 = st.columns(2)
    with c1, st.container(key="card_co2_total"):
        card_header("Total CO2 saved by mode", "Tonnes")
        show(mode_hbar(by_mode, "CO2 saved (t)", "CO2 saved (tonnes)", ",.0f"))
    with c2, st.container(key="card_co2_pax"):
        card_header("CO2 saved per passenger by mode", "Kilograms per passenger")
        show(mode_hbar(by_mode, "CO2 per passenger (kg)", "CO2 saved per passenger (kg)", ".3f"))
    c3, c4 = st.columns(2)
    with c3, st.container(key="card_bubble"):
        card_header("Environmental benefit vs cost recovery", "Bubble area = passengers; right of the line = covers costs")
        s = by_mode.set_index("Transport_Mode").reindex(modes_present).reset_index()
        fig = go.Figure()
        sizeref = 2.0 * s["Passengers"].max() / (60 ** 2)
        for _, r in s.iterrows():
            fig.add_scatter(x=[r["Revenue / cost"]], y=[r["CO2 per passenger (kg)"]], mode="markers+text",
                            name=str(r["Transport_Mode"]), text=[str(r["Transport_Mode"])], textposition="top center",
                            marker=dict(size=[r["Passengers"]], sizemode="area", sizeref=sizeref, sizemin=8,
                                        color=MODE_COLOURS[r["Transport_Mode"]], line=dict(color="white", width=1.5)),
                            hovertemplate="%{text}<br>Revenue / cost: %{x:.2f}<br>CO2 per passenger: %{y:.3f} kg"
                                          "<extra></extra>")
        fig.add_vline(x=1, line_dash="dash", line_color=INK, line_width=1.2, annotation_text="break-even")
        y0, y1 = float(s["CO2 per passenger (kg)"].min()), float(s["CO2 per passenger (kg)"].max())
        fig.update_xaxes(title="Revenue / operating cost", range=[-0.12, max(1.1, float(s["Revenue / cost"].max())) + 0.2],
                         showgrid=True, gridcolor="#F0F1F4")
        fig.update_yaxes(title="CO2 saved per passenger (kg)", range=[y0 - 0.25 * (y1 - y0 + 0.01),
                                                                      y1 + 0.3 * (y1 - y0 + 0.01)])
        show(style(fig, 400))
    with c4, st.container(key="card_co2_trend"):
        card_header("Monthly CO2 saved per record", "Mean kilograms per record")
        mf = monthly_financials(df)
        fig = go.Figure(go.Scatter(x=mf["Year_Month"], y=mf["CO2 saved per record (kg)"], mode="lines",
                                   line=dict(color=MODE_COLOURS["Tram"], width=2.2, shape="hv"), fill="tozeroy",
                                   fillcolor="rgba(0,158,115,0.10)",
                                   hovertemplate="%{x|%b %Y}: %{y:,.0f} kg per record<extra></extra>"))
        fig.update_xaxes(dtick="M3", tickformat="%b %Y")
        show(style(fig, 230))
        card_header("Energy per passenger by mode", "kWh per passenger")
        show(mode_hbar(by_mode, "Energy per passenger (kWh)", "Energy per passenger (kWh)", ".3f"))

# ============================= DATA & METHOD ============================== #
with tabs[8]:
    c1, c2 = st.columns(2)
    with c1, st.container(key="card_defs"):
        card_header("Measure definitions", "Headline KPIs plus the supporting measures used in chips and tabs")
        st.dataframe(pd.DataFrame([(lbl, h) for _, lbl, _, h in KPI_SPECS], columns=["KPI", "Definition"]),
                     hide_index=True)
    with c2, st.container(key="card_notes"):
        card_header("Data notes and limitations")
        st.markdown(
            f"- {len(data):,} service records, {MIN_DATE:%b %Y} – {MAX_DATE:%b %Y}, {data['City'].nunique()} cities in "
            f"{data['Country'].nunique()} countries. The file was checked when loaded (complete, valid values); "
            "no records removed.\n"
            "- **Records, not a full count:** totals add up the selected records and change with how many "
            "records there are; trends therefore use the average per record.\n"
            "- **'Trips'** always equals Passengers for Bike Share / E-Scooter, so it appears to mean rentals "
            "there and vehicle journeys for other modes; it is not compared across modes.\n"
            "- **Weather** labels do not follow the seasons (snow in summer, heatwaves in winter), so weather "
            "results describe this dataset only.\n"
            "- **City demand index** is adjusted for each city's mix of modes, routes and conditions; other "
            "city figures are raw averages.\n"
            "- **Associations, not causes:** group differences may reflect other factors, especially mode.\n"
            "- Revenue is ticket income only (no subsidies); goal-tracking targets are illustrative.")
    with st.container(key="card_records"):
        card_header("Filtered records", f"{len(df):,} records")
        cols = ["Record_ID", "Date", "City", "Country", "Transport_Mode", "Route_Type", "Weather",
                "Event_Day"] + NUMERIC_COLS
        st.dataframe(df[cols], hide_index=True, height=320)
        st.download_button("Download filtered data (CSV)", df[cols].to_csv(index=False).encode("utf-8"),
                           file_name="ecomove_filtered.csv", mime="text/csv", icon=":material/download:")

st.caption(f"Source: {upload['name']} · COM7021 Data Visualisation · Student ID 25138486")
