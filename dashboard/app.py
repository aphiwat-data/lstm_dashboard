"""Next-day gold price forecast: the dashboard.

Run from the repository root:  python3 -m streamlit run dashboard/app.py
The page is assembled from the small modules in `ui/` (header, sidebar, one module per tab). Nothing here needs TensorFlow:
the forecast comes from the daily pipeline on AWS and the evaluation from the result files committed in `daily/results`.
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

sidebar.render()
header.render(cur, data.price_history(preds, cur), res, preds, reg)

st.write("")
track, explorer, reliable, about = st.tabs([":material/history: Track record", ":material/monitoring: Data explorer", ":material/verified: How reliable?", ":material/info: About"])
with track:
    tab_track.render(preds, data.live_log())
with explorer:
    tab_market.render(data.prices())
with reliable:
    tab_reliability.render(res, preds, reg)
with about:
    tab_about.render(res)
