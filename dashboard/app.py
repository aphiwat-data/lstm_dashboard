"""Next-day gold price forecast: the dashboard.

Run from the repository root:  python3 -m streamlit run dashboard/app.py
The page is assembled from the small modules in `ui/` (header, sidebar, one module per panel). Nothing here needs TensorFlow:
the forecast comes from the daily pipeline on AWS and the evaluation from the result files committed in `daily/results`.

Layout: the header (the forecast, its chart, the accuracy scorecard) and a navigation bar of three sections in one row. Nothing is shown until a section is clicked, and clicking
the active one hides it again: forecast vs. actual, actual market data and the reliability tests. What was built and tried is documented in the README.
"""
import streamlit as st

st.set_page_config(page_title="Gold Price Forecast", page_icon=":material/show_chart:", layout="wide", initial_sidebar_state=320)  # a width in pixels: "auto" behaviour with a sidebar of that width

from ui import data, header, sidebar, tab_market, tab_reliability, tab_track  # noqa: E402  (after set_page_config)

res, preds = data.results()
reg = data.regression()
log = data.live_log()
key = data.headline_key(res, log, data.live_forecast())  # the cross-asset model while the daily pipeline publishes its forecasts, otherwise the LSTM
cur = data.current_forecast(res, key)
if cur is None:
    st.error("No forecast is available right now. Please try again in a few minutes.", icon=":material/error:")
    st.stop()

vol = data.volatility()
fh = data.forecast_history(preds, log, vol[0] if vol else None, key)

sidebar.render()
header.render(cur, data.price_history(preds, cur), fh, res, reg, log, key)

SECTIONS = {"forecast": (":material/compare_arrows:", "Forecast vs. actual"), "market": (":material/candlestick_chart:", "Actual market data"),
            "reliable": (":material/verified:", "How reliable?")}

st.write("")
section = st.segmented_control("Sections", list(SECTIONS), format_func=lambda k: f"{SECTIONS[k][0]} **{SECTIONS[k][1]}**", selection_mode="single", default=None, key="section",
                               label_visibility="collapsed", width="stretch")  # the navigation bar: nothing selected at first, so nothing is shown
if section == "forecast":
    tab_track.render(preds, log, fh, key)
elif section == "market":
    tab_market.render(data.prices(), cur, vol, data.crossasset_inputs() if key == data.HEADLINE else None, key)
elif section == "reliable":
    tab_reliability.render(res, preds, reg, *data.interval_tests(key), key)
