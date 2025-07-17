"""
Streamlit UI for the LangGraph‑powered Fashion Assistant.
──────────────────────────────────────────────────────────
Minimal surface: the UI only collects user input, forwards it to the
FastAPI `/chat` endpoint, and renders the reply + thumbnails.
All business logic (memory, search, cart, styling) lives in the backend.
"""
from __future__ import annotations
import os, re, json, asyncio, uuid, httpx, pandas as pd
from pathlib import Path
from typing import List, Dict

import streamlit as st
from audio_recorder_streamlit import audio_recorder
from openai import OpenAI  # whisper – voice optional

# ───────────────────────── CONFIG ─────────────────────────
API_ROOT   = os.getenv("FASHION_API_URL", "http://localhost:8000")
SKU_CSV    = os.getenv("SKU_IMAGE_CSV",  "skus_url.csv")
LANGUAGES  = ["English", "Arabic", "Hindi"]

# ───────────────────────── DATA ───────────────────────────
@st.cache_data(show_spinner=False)
def load_sku_csv(p: str) -> pd.DataFrame:
    return pd.read_csv(p, dtype=str) if Path(p).exists() else pd.DataFrame(columns=["sku", "url"])

sku_df = load_sku_csv(SKU_CSV)

# ───────────────────────── HELPERS ────────────────────────

def html_img(url: str, key: str) -> str:
    """Return a <button><img/></button> snippet so the image is clickable."""
    return (
        f"<button type='submit' name='sel' value='{key}' "
        f"style='border:none;padding:0;background:none;cursor:pointer;'>"
        f"<img src='{url}' width='120' style='border-radius:6px'/>"
        f"</button>"
    )


def thumb_strip_html(items: List[Dict[str, Any]]) -> str:
    html = "<form>"
    for it in items:
        url = sku_df.loc[sku_df.sku == str(it["sku"]), "url"].values
        if url.size:
            html += html_img(url[0], key=str(it["sku"]))
    html += "</form>"
    return f"<div style='display:flex;gap:10px;overflow-x:auto;'>{html}</div>"


async def post_json(url: str, payload: dict):
    async with httpx.AsyncClient(timeout=90) as cli:
        r = await cli.post(url, json=payload)
        r.raise_for_status()
        return r.json()


# ───────────────────────── ORDINAL HELPERS ────────────────
ORD_WORD = {"first":1,"second":2,"third":3,"fourth":4,"fifth":5,
            "sixth":6,"seventh":7,"eighth":8,"ninth":9,"tenth":10}

def resolve_ordinal(msg: str, last_items: List[Dict[str, Any]]):
    """Return SKU if user wrote '3rd sku' or 'third one'."""
    idx = None
    m = re.search(r"\b(\d+)(?:st|nd|rd|th)?\b", msg)
    if m:
        idx = int(m.group(1))
    else:
        for w, n in ORD_WORD.items():
            if re.search(rf"\b{w}\b", msg, re.I):
                idx = n; break
    if idx and 1 <= idx <= len(last_items):
        return str(last_items[idx-1]["sku"])
    return None

# ───────────────────────── STREAMLIT PAGE SETUP ───────────
st.set_page_config(page_title="Fashion Assistant", page_icon="👗")
st.title("Fashion Stylist")

# ───── session init
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
for k, v in {
    "language": LANGUAGES[0],
    "selected_sku": None,
    "last_items": [],
}.items():
    st.session_state.setdefault(k, v)

# language selector (sends nothing to backend until next turn)
st.selectbox("Language", LANGUAGES, index=LANGUAGES.index(st.session_state.language), key="language")

# ───────────────────────── INPUT (voice + text) ───────────
voice_txt, text_in = None, st.chat_input("Describe what you’re looking for…")
with st.expander("🎤 Voice"):
    audio = audio_recorder(icon_size="2x")
    if audio:
        voice_txt = OpenAI().audio.transcriptions.create(model="whisper-1", file=audio).text.strip()

# ───── thumbnail click
clicked = st.query_params.get("sel")
if clicked:
    st.session_state.selected_sku = clicked

# ───── compose user message
user_msg = voice_txt or text_in

# ───────────────────────── MAIN CONVERSE CALL ─────────────
async def converse(msg: str):
    # auto‑resolve ordinal references
    if not st.session_state.selected_sku:
        maybe = resolve_ordinal(msg, st.session_state.last_items)
        if maybe:
            st.session_state.selected_sku = maybe

    payload = {
        "session_id":  st.session_state.session_id,
        "message":     msg,
        "language":    st.session_state.language,
    }
    if clicked or maybe:
        payload["selected_sku"] = st.session_state.selected_sku
    data = await post_json(f"{API_ROOT}/chat", payload)

    # render assistant reply
    st.chat_message("assistant").markdown(data["assistant_reply"])

    # thumbnails
    st.session_state.last_items = data.get("last_items") or []
    if st.session_state.last_items:
        st.markdown(thumb_strip_html(st.session_state.last_items), unsafe_allow_html=True)

    # backend might change selection
    st.session_state.selected_sku = data.get("selected_sku")

# ───────────────────────── RUN ON SUBMIT ────────────────
if user_msg:
    st.chat_message("user").markdown(user_msg)
    asyncio.run(converse(user_msg))
