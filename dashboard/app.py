"""Next-day gold price forecast: the dashboard.

Run from the repository root:  python3 -m streamlit run dashboard/app.py
The page is assembled from the small modules in `ui/` (header, sidebar, one module per panel). Nothing here needs TensorFlow:
the forecast comes from the daily pipeline on AWS and the evaluation from the result files committed in `daily/results`.

Layout: the header (the forecast, its chart, the accuracy scorecard) and four collapsible panels that stay closed until clicked: three of visualisations (forecast vs. actual,
actual market data, the reliability tests), then one that only explains what was built and tried (About).
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
with st.expander("**Forecast vs. actual**", icon=":material/compare_arrows:"):
    tab_track.render(preds, log, fh)
with st.expander("**Actual market data**", icon=":material/candlestick_chart:"):
    tab_market.render(data.prices(), cur, vol)
with st.expander("**How reliable?**", icon=":material/verified:"):
    tab_reliability.render(res, preds, reg, *data.interval_tests())
with st.expander("**About the project**", icon=":material/menu_book:"):
    tab_about.render(res, reg, data.prices())
