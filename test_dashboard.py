"""Automated tests for ecomove_dashboard.py (COM7021, student ID 25138486).

Run:  pytest -q test_dashboard.py   (in the folder that holds the dashboard and the dataset)
Needs: pip install pytest streamlit plotly pandas numpy statsmodels openpyxl

The first group tests the data functions (reading, checking, preparing, filtering and
summarising). The second group starts the real app with Streamlit's AppTest and checks
what a user would see.
"""
import io
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

HERE = Path(__file__).parent
APP_FILE = str(HERE / "ecomove_dashboard.py")
DATA_FILE = next(HERE / n for n in ["[5442] EcoMoveMobility.xlsx", "EcoMoveMobility.xlsx"] if (HERE / n).exists())

# Load sections 1-3 of the dashboard (settings, data and calculation functions) without
# starting the page in section 4, so the functions can be tested on their own.
app = types.ModuleType("dashboard_functions")
sys.modules[app.__name__] = app
exec(compile(Path(APP_FILE).read_text().split("# 4. PAGE")[0], APP_FILE, "exec"), app.__dict__)


# ----------------------------------------------------------------------------- helpers
@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return app.read_workbook(DATA_FILE.read_bytes())


@pytest.fixture(scope="module")
def data(raw) -> pd.DataFrame:
    clean, errors, _ = app.validate_dataset(raw)
    assert not errors
    prepared, _ = app.prepare_data(clean)
    return app.add_derived_features(prepared)


def to_xlsx(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    return buffer.getvalue()


def errors_for(raw: pd.DataFrame, change) -> list[str]:
    bad = raw.copy()
    change(bad)
    return app.validate_dataset(bad)[1]


# ----------------------------------------------------------------------------- reading the file
def test_supplied_workbook_has_every_column(raw):
    assert list(raw.columns[:len(app.EXPECTED_COLUMNS)]) == app.EXPECTED_COLUMNS
    assert len(raw) == 5000


def test_column_names_match_ignoring_case_and_spaces(raw):
    renamed = raw.rename(columns={"Transport_Mode": "transport mode", "Customer_Rating": "CUSTOMER RATING"})
    assert "Transport_Mode" in app.read_workbook(to_xlsx(renamed)).columns


def test_non_excel_file_is_rejected():
    with pytest.raises(app.SchemaError, match="could not be opened"):
        app.read_workbook(b"this is not a workbook")


def test_missing_columns_are_named(raw):
    with pytest.raises(app.SchemaError, match="Weather"):
        app.read_workbook(to_xlsx(raw.drop(columns=["Weather"])))


# ----------------------------------------------------------------------------- checking the values
def test_supplied_dataset_passes_every_check(raw):
    _, errors, warnings = app.validate_dataset(raw)
    assert errors == [] and warnings == []


@pytest.mark.parametrize("change, message", [
    (lambda d: d.__setitem__("Passengers", d["Passengers"].astype(object).where(d.index != 0, None)), "empty cell"),
    (lambda d: d.__setitem__("Trips", d["Trips"].astype(object).where(d.index != 0, "many")), "not numbers"),
    (lambda d: d.__setitem__("Ticket_Revenue_EUR", d["Ticket_Revenue_EUR"].where(d.index != 0, -5.0)), "negative"),
    (lambda d: d.__setitem__("Customer_Rating", d["Customer_Rating"].where(d.index != 0, 7.0)), "outside 1 to 5"),
    (lambda d: d.__setitem__("On_Time_Percentage", d["On_Time_Percentage"].where(d.index != 0, 140.0)),
     "outside 0 to 100"),
    (lambda d: d.__setitem__("Passengers", d["Passengers"].where(d.index != 0, 0)), "zero value"),
    (lambda d: d.__setitem__("Transport_Mode", d["Transport_Mode"].where(d.index != 0, "Ferry")), "Ferry"),
    (lambda d: d.__setitem__("Date", d["Date"].where(d.index != 0, 1)), "plausible range"),
])
def test_invalid_values_are_reported(raw, change, message):
    errors = errors_for(raw, change)
    assert any(message in e for e in errors), errors


def test_too_few_rows_is_an_error(raw):
    assert any("at least" in e for e in app.validate_dataset(raw.head(10))[1])


def test_a_different_dataset_gives_a_warning_not_an_error(raw):
    _, errors, warnings = app.validate_dataset(raw.head(1000))
    assert errors == [] and any("not the dataset" in w for w in warnings)


# ----------------------------------------------------------------------------- preparing the data
def test_dates_convert_from_excel_serial_numbers():
    out = app.convert_dates(pd.Series([45292, 46022]))
    assert list(out.dt.date.astype(str)) == ["2024-01-01", "2025-12-31"]


def test_preparation_keeps_every_record_and_passenger(raw, data):
    assert len(data) == len(raw)
    assert data["Passengers"].sum() == raw["Passengers"].sum() == 40_909_721


def test_derived_measures_are_correct(data):
    row = data.iloc[0]
    assert row["Operating_Surplus_EUR"] == pytest.approx(row["Ticket_Revenue_EUR"] - row["Operating_Cost_EUR"])
    assert row["Complaints_per_10k_Pax"] == pytest.approx(row["Accessibility_Complaints"] / row["Passengers"] * 1e4)


def test_safe_divide_returns_nan_for_zero():
    out = app.safe_divide(pd.Series([1.0, 2.0]), pd.Series([0.0, 4.0]))
    assert np.isnan(out.iloc[0]) and out.iloc[1] == 0.5


# ----------------------------------------------------------------------------- filters and KPIs
@pytest.mark.parametrize("filters, column, allowed", [
    (app.FilterState(countries=["Germany"]), "Country", {"Germany"}),
    (app.FilterState(cities=["Paris", "Rome"]), "City", {"Paris", "Rome"}),
    (app.FilterState(modes=["Bus"]), "Transport_Mode", {"Bus"}),
    (app.FilterState(routes=["Suburban Feeder"]), "Route_Type", {"Suburban Feeder"}),
    (app.FilterState(weather=["Snow", "Rain"]), "Weather", {"Snow", "Rain"}),
    (app.FilterState(event="Event days only"), "Event_Day", {"Yes"}),
])
def test_each_filter_keeps_only_matching_records(data, filters, column, allowed):
    out = app.apply_filters(data, filters)
    assert len(out) > 0 and set(out[column].astype(str)) == allowed


def test_date_filter(data):
    out = app.apply_filters(data, app.FilterState(start=pd.Timestamp("2025-01-01").date()))
    assert out["Date"].min() >= pd.Timestamp("2025-01-01") and len(out) == 2433


def test_network_kpis_match_the_report(data):
    k = app.compute_kpis(data)
    assert k["records"] == 5000 and k["passengers"] == 40_909_721
    assert round(k["revenue_cost"], 3) == 0.996 and round(k["delay"], 2) == 4.35


def test_headline_kpis_are_four_comparable_rates(data):
    assert app.HEADLINE_KPIS == ["delay", "revenue_cost", "co2_per_pax", "complaint_rate"]
    k = app.compute_kpis(data)
    assert round(k["co2_per_pax"], 3) == 0.110 and round(k["complaint_rate"], 1) == 15.3


def test_kpis_for_an_empty_selection_are_safe(data):
    k = app.compute_kpis(data.iloc[0:0])
    assert k["records"] == 0 and np.isnan(k["passengers"])


def test_mode_totals_add_up_to_the_network(data):
    by_mode = app.group_summary(data, "Transport_Mode")
    assert by_mode["Passengers"].sum() == data["Passengers"].sum()
    bus = by_mode.set_index("Transport_Mode").loc["Bus"]
    assert bus["Revenue / cost"] == pytest.approx(bus["Revenue (EUR)"] / bus["Operating cost (EUR)"])


def test_adjusted_city_index_matches_the_report(data):
    idx = app.adjusted_demand_index(data, "City").set_index("City")["Adjusted index"].round(2)
    assert (idx["Paris"], idx["Berlin"], idx["Rotterdam"]) == (1.31, 1.24, 0.79)


def test_event_uplift_needs_both_kinds_of_day(data):
    only_event = app.apply_filters(data, app.FilterState(event="Event days only"))
    assert app.event_uplift(only_event).empty
    assert len(app.event_uplift(data)) == 6


# ----------------------------------------------------------------------------- the running app
def test_app_opens_on_the_supplied_dataset():
    at = AppTest.from_file(APP_FILE, default_timeout=120).run()
    assert not at.exception
    assert len(at.tabs) == 9
    page = " ".join(m.value for m in at.markdown)
    assert "Showing <b>5,000 of 5,000 records</b>" in page and "40.91m" in page
    assert page.count("<div class='kpi'>") == 4


def test_a_mode_filter_updates_the_page():
    at = AppTest.from_file(APP_FILE, default_timeout=120).run()
    at.multiselect(key="f_modes").select("Bus").run()
    assert not at.exception
    page = " ".join(m.value for m in at.markdown)
    assert "Showing <b>1,375 of 5,000 records</b>" in page


def test_an_empty_selection_shows_a_message_not_an_error():
    at = AppTest.from_file(APP_FILE, default_timeout=120).run()
    at.multiselect(key="f_countries").select("Austria").run()
    at.multiselect(key="f_modes").select("E-Scooter").run()
    at.multiselect(key="f_weather").select("Snow").run()      # no snowy E-Scooter records in Austria
    assert not at.exception
    assert any("No records match" in w.value for w in at.warning)


def test_upload_page_offers_the_supplied_dataset():
    at = AppTest.from_file(APP_FILE, default_timeout=120)
    at.session_state["choose_file"] = True
    at.run()
    assert not at.exception and len(at.tabs) == 0
    at.button[0].click().run()          # "Use the supplied dataset instead"
    assert not at.exception and len(at.tabs) == 9
