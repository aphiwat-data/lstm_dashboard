"""Next-day gold price forecast: the dashboard.

Run from the repository root:  python3 -m streamlit run dashboard/app.py
The page is assembled from the small modules in `ui/` (header, sidebar, one module per panel). Nothing here needs TensorFlow:
the forecast comes from the daily pipeline on AWS and the evaluation from the result files committed in `daily/results`.

Layout: the header (the forecast, its chart, the accuracy scorecard) and a navigation bar of four sections in one row. Nothing is shown until a section is clicked, and clicking
the active one hides it again: three sections of visualisations (forecast vs. actual, actual market data, the reliability tests), then one that only explains what was built
and tried (About).
"""
import streamlit as st

st.set_page_config(page_title="Gold Price Forecast", page_icon=":material/show_chart:", layout="wide", initial_sidebar_state=320)  # a width in pixels: "auto" behaviour with a sidebar of that width

from ui import data, header, sidebar, tab_about, tab_market, tab_reliability, tab_track  # noqa: E402  (after set_page_config)

res, preds = data.results()
reg = data.regression()
cur = data.current_forecast(res)
if cur is None:
    st.error("No forecast is available right now. Please try again in a few minutes.", icon=":material/error:")
    st.stop()

log, vol = data.live_log(), data.volatility()
fh = data.forecast_history(preds, log, vol[0] if vol else None)

sidebar.render()
header.render(cur, data.price_history(preds, cur), fh, res, reg, log)

SECTIONS = {"forecast": (":material/compare_arrows:", "Forecast vs. actual"), "market": (":material/candlestick_chart:", "Actual market data"),
            "reliable": (":material/verified:", "How reliable?"), "about": (":material/menu_book:", "About the project")}

st.write("")
section = st.segmented_control("Sections", list(SECTIONS), format_func=lambda k: f"{SECTIONS[k][0]} **{SECTIONS[k][1]}**", selection_mode="single", default=None, key="section",
                               label_visibility="collapsed", width="stretch")  # the navigation bar: nothing selected at first, so nothing is shown
if section == "forecast":
    tab_track.render(preds, log, fh)
elif section == "market":
    tab_market.render(data.prices(), cur, vol)
elif section == "reliable":
    tab_reliability.render(res, preds, reg, *data.interval_tests())
elif section == "about":
    tab_about.render(res, reg, data.prices())
