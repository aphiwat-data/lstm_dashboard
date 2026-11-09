"""Next-day gold price forecast: the dashboard.

Run from the repository root:  python3 -m streamlit run dashboard/app.py
The page is assembled from the small modules in `ui/` (header, sidebar, one module per tab). Nothing here needs TensorFlow:
the forecast comes from the daily pipeline on AWS and the evaluation from the result files committed in `daily/results`.

Layout: the header (briefing, the forecast, its chart, the accuracy scorecard) and three tabs of visualisations (forecast vs. actual, actual market data,
the reliability tests), then one tab that only explains what was built and tried (About).
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

st.write("")
track, explorer, reliable, about = st.tabs([":material/compare_arrows: Forecast vs. actual", ":material/candlestick_chart: Actual market data", ":material/verified: How reliable?",
                                            ":material/menu_book: About the project"])
with track:
    tab_track.render(preds, log, fh)
with explorer:
    tab_market.render(data.prices(), cur, vol)
with reliable:
    tab_reliability.render(res, preds, reg, *data.interval_tests())
with about:
    tab_about.render(res, reg, data.prices())
