"""
fashion_langgraph.py  ✧  LangGraph‑based rewrite of the original fashion‑assistant logic.
──────────────────────────────────────────────────────────────────────────────
• Keeps **all prompts identical** – chains are copy‑pasted unchanged.
• Encapsulates the entire conversational flow inside a LangGraph `StateGraph`.
• Exposes one public helper `async run_graph(session_id:str, user_msg:str, language:str)`
  that you can call from your FastAPI or Streamlit layers.
• Maintains an in‑memory `SESSIONS` dict for per‑user state (swap for Redis etc. later).

Python ≥3.10 | pip install langgraph langchain langchain-openai fastapi faiss-cpu
"""
from __future__ import annotations
import asyncio, json, os, tempfile, socket
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Optional

import numpy as np, faiss                                  # type: ignore
from fastapi import HTTPException

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langgraph.graph import StateGraph, END, START               # ✧ LANGGRAPH ✧
from color_mapping import COLOR_MAPPING
# ─────────────────────────── CONFIG (unchanged) ────────────────────────────
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "openai").lower()
EMBED_MODEL   = "text-embedding-3-small"
EMBED_DIM     = 1536
BATCH_SIZE    = 100
INDEX_PATH    = Path(os.getenv("CATALOG_INDEX_PATH", "./catalog_2.index"))
META_PATH     = Path(os.getenv("CATALOG_META_PATH",  "./catalog_2.meta.jsonl"))
SUPPORTED_LANGS = {"english", "arabic"}
DEFAULT_LANG    = "english"

# ─────────────────────────── LLMs / Embeddings ─────────────────────────────
llm_kw      = ChatOpenAI(model_name="gpt-4o", temperature=0.2)
llm_filters = ChatOpenAI(model_name="gpt-4o", temperature=0)
llm_stylist = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
llm_chat    = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

if EMBED_BACKEND == "openai":
    embedder = OpenAIEmbeddings(model=EMBED_MODEL)
else:
    local_model_name = os.getenv("LOCAL_MODEL", "all-MiniLM-L12-v2")
    embedder = SentenceTransformerEmbeddings(model_name=local_model_name)
    EMBED_DIM = embedder.client.get_sentence_embedding_dimension()
    BATCH_SIZE = 1024

# ─────────────────────────── FAISS (unchanged) ─────────────────────────────
def _load_faiss(idx_path: Path, meta_path: Path):
    idx = faiss.read_index(str(idx_path))
    metas = [json.loads(l) for l in meta_path.open()]
    return idx, metas

faiss_idx, metas = _load_faiss(INDEX_PATH, META_PATH)

# ─────────────────────────── Prompt & Chains (UNCHANGED) ───────────────────
_trim = lambda txt: txt[-4000:]  # lightweight token trim helper

REPLY_ITEM_FIELDS = ("sku", "brand", "level_1", "level_2", "level_3", "color", "material", "description")

def _clean_for_json(v):
    # minimal NaN scrub (works w/ float + numpy)
    try:
        if v != v:  # NaN check
            return None
    except Exception:
        pass
    return v

def _reply_items_subset(items, fields=REPLY_ITEM_FIELDS, max_items=10, max_chars=4000):
    subset = []
    for m in items[:max_items]:
        subset.append({f: _clean_for_json(m.get(f)) for f in fields})
    txt = json.dumps(subset, ensure_ascii=False)
    if len(txt) > max_chars:
        txt = txt[:max_chars]
    return txt

# ───────── SKU greeting (when user clicks a product) ─────────

prompt_targeted_styling_reply = ChatPromptTemplate.from_messages([
    ("system",
     "You are a friendly, concise human like fashion assistant.\n"
     "Be crisp and to the point, no long prose.\n"
     "Input gives:\n"
     "• target_item (JSON)\n"
     "• user_request (raw text)\n"
     "• recs (JSON list; each has sku, brand, level_1, level_2, level_3, color, material, description)\n\n"
     "Write a SHORT reply (≤2 sentences) that:\n"
     "1. Acknowledges what the shopper asked for & lightly reference the overall genre of items being suggested.\n"
     "2. Lets them know you've pulled a few matching options (no counts; vary phrasing).\n"
     "3. (Optional) quick styling hook based on patterns across recs (e.g., lots of silver evening heels; mixed metallics; formal vibe).\n"
     "Do NOT list products, prices, or brands inline — UI shows items.\n"
     "Stay upbeat, personal, brief and friendly."
     "If user is asking about some other specific item which you do not have details about, guide them by telling them they may click the sku or explain what they are looking for so you can better serve them.\n"
    ),
    ("human",
     "target_item: {target_item}\n"
     "user_request: {user_request}\n"
     "recs: {recs}")
])
targeted_styling_reply_chain = prompt_targeted_styling_reply | llm_stylist

prompt_kw = ChatPromptTemplate.from_messages([
    ("system",
     "You are a fashion merchandising assistant. "
        "When given a customer intent or scenario, respond with a JSON array "
        "of 5‑10 concise, extremely relevant fashion search keywords or phrases related to the customer intent."),
    ("human", "{query}"),
])
kw_chain = prompt_kw | llm_kw | JsonOutputParser()

prompt_filt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract shopping filters from the prompt. Return JSON with keys exactly: category, include_brands, "
            "exclude_brands, include_colors, exclude_colors, price_min, price_max, gender(MEN, WOMEN, KIDS)," 
            "sleeves (sleeveless, half sleeve, full sleeve),"
            "gender should be set to kids if you detect the user is looking for kids clothing, "
            "Leave a field empty (null or []) if not specified.",
        ),
        "For 'category' infer high-level level_2 such as CLOTHING, SHOES, BAGS.",
        ("human", "{query}"),
    ]
)
filter_chain = prompt_filt | llm_filters | JsonOutputParser()

# Add this new chain definition in fashion_langgraph.py (after the existing chains):

prompt_targeted_style = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a fashion stylist. The user has selected a product and wants to find a specific type of item to pair with it.\n"
            "Based on the selected product details and the user's request, generate 5-8 relevant search keywords "
            "that would help find the requested item type that pairs well with the selected product.\n\n"
            "Consider:\n"
            "- Colors that complement or match\n"
            "- Styles that work together\n"
            "- Formality level matching\n"
            "- Seasonal appropriateness\n\n"
            "Return ONLY a JSON array of keywords: [\"keyword1\", \"keyword2\", ...]\n"
            "No explanations, just the keyword array."
        ),
        ("human", "Selected product: {selected_product}\n\nUser request: {user_request}")
    ]
)

targeted_style_chain = prompt_targeted_style | llm_stylist | JsonOutputParser()


prompt_comp_kw = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a fashion stylist. From the product below, extract up to three relevant PRODUCT CATEGORIES "
            "that would pair well (e.g. 'bags', 'shoes', 'pants', 'dress') and, for each category, up to five "
            "concise search keywords. Return STRICT JSON: {{\"category\": [\"kw1\", \"kw2\", ...]}}. No prose."
            "You dont need to have all the categories, just the ones that are relevant to the user chosen product (example- pants or shirt might be irrelevant if the user has selected a dress, similarly a dress might be irrelevant to a shirt but might be relevant only to heels).\n"
            "the keywords should be mentioned as full descriptive phrases (example- slim pants instead of just slim, dress shoes instead of just shoes). ",
        ),
        ("human", "{answer}"),
    ]
)

comp_kw_chain = prompt_comp_kw | llm_kw | JsonOutputParser()

prompt_state = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You manage session state for a fashion shopping assistant."
            "Reply with JSON containing keys: mode and memory.\n\n"
            "Rules:\n"
            "- mode must be 'catalog', 'reset', or 'chat' or 'sku'.\n"
            "- memory must be ONE sentence detailing the CURRENT desired outfit if mode='catalog', else ''."
            "Goal of the memory is to intepret the user's overall intent in the best way possible such that it can be used to generate good keywords"
            "You may use previous memory but priority to be given to new reqirements.\n"
            "If a styling suggestion is made, it should be mapped to the last item requested by the user.\n"
            "example: 'can you show matching shoes for these' should be mapped to the last item shown to the user.\n",
        ),
        ("human", "{payload}"),
    ]
)
update_state_chain = prompt_state | llm_chat | JsonOutputParser()

intent_chain = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Classify intent about the selected product. Respond with one word: question_sku / buy / style / other_products.",
            ),
            ("human", "{text}"),
        ]
    )
    | llm_chat
)

qa_chain = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                # ─────────────── start of new text ───────────────
                "You are a product-expert stylist. You have two knowledge sources:\n"
                "1. **facts** → a JSON blob for ONE SKU (keys like material, color, sleeve, category).\n"
                "2. **general garment knowledge** (e.g. cotton is usually machine-washable, silk is delicate).\n\n"
                "Answer the user’s question by COMBINING those two sources:\n"
                "• If the answer can be *reliably inferred* from the JSON **or** common-sense garment care rules, give the best-guess answer **and** a brief reason in parentheses.\n"
                "    – Example: “Yes, it’s machine-washable (cotton fabrics typically withstand gentle cycles).”\n"
                "• If you genuinely lack enough info to form a reasonable inference, give whatever facts you have in a confident manner.\n"
                "Only say what is absolutely certain based on the JSON and your garment knowledge.\n"
                "• Never invent brand names, prices, or features that contradict the JSON.\n\n"
                "If user is asking about some other specific item which you do not have details about, guide them by telling them they may click the sku tile or explain what they are looking for so you can better serve them.\n"
                "{facts}"
                # ─────────────── end of new text ───────────────
            ),
            ("human", "{question}"),
        ]
    )
    | llm_chat
)

prompt_cart_confirmation = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful shopping assistant. The user just added an item to their cart. "
            "Write a brief, enthusiastic confirmation message (1-2 sentences) that:\n"
            "- Confirms the item was added to cart\n"
            "- Mentions you'll show complementary items\n"
            "- Varies the wording each time to feel natural and friendly\n"
            "Keep it friendly and conversational."
            "Keep a suggestive tone so that the user feels like they can ask for more items or help.\n"
        ),
        ("human", "Item added to cart: {product_info}")
    ]
)
cart_confirmation_chain = prompt_cart_confirmation | llm_chat

# concise_chat_chain = (
#     ChatPromptTemplate.from_messages([("human", "{history}")]) | llm_chat
# )
concise_chat_chain = (
    ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a warm, concise fashion assistant.\n"
            "Respond briefly (≤2 sentences unless the user asks for detail).\n"
            "If the user asks *why* or *on what basis* items were recommended and product results "
            "were recently shown, explain at a high level what your thought process was "
        ),
        ("human", "{history}")
    ])
    | llm_chat
)

prompt_styling_intent = ChatPromptTemplate.from_messages([
    ("system",
     "You are an intent classifier for a fashion assistant. "
     "The user has a product in their cart and is currently looking at a specific item. "
     "Determine the user's intent based on their message.\n\n"
     "Respond with ONLY one word:\n"
     "- 'styling' if user wants complementary items for their CURRENT cart item\n"
     "- 'new_search' if user is starting a completely new search (different person, different occasion, unrelated items)\n"
     "- 'other' if it's a product question, general chat, or not about finding items\n\n"
     "Examples of 'styling' (complementary to current item):\n"
     "- 'show me pants for this'\n"
     "- 'what shoes would go with this?'\n"
     "- 'do you have khaki pants instead?'\n"
     "- 'find me a matching bag'\n"
     "- 'different colored options'\n\n"
     "- 'show me something else'\n\n"
     "Examples of 'new_search' (completely different context):\n"
     "- 'looking for a red dress for my wife'\n"
     "- 'I need women's shoes'\n"
     "- 'find me a gift for my daughter'\n"
     "- 'looking for kids clothing'\n"
     "- 'I want something for work'\n"
     "- 'searching for party wear'\n\n"
     "Key rule: If user wants items to complement/match their current selection, it's 'styling'. "
     "Only use 'new_search' if they're shopping for someone else or a completely different context.\n\n"
     "Examples of 'other':\n"
     "- 'what material is this?'\n"
     "- 'add to cart'\n"
     "- 'what size should I get?'\n"
     "- 'hello'\n"
     "- 'what's the return policy?'"),
    ("human", "{user_message}")
])

styling_intent_chain = prompt_styling_intent | llm_chat

# ─────────────────────────── Helper functions (unchanged) ──────────────────

def _embed_query(text: str) -> np.ndarray:
    vec = embedder.embed_query(text)
    vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec


def _meta_pass(meta, *, cat, inc_c, exc_c, inc_b, exc_b,gender, pmin, pmax, relax_brand, relax_price, sleeves_filt) -> bool:
    colour = str(meta.get("color", "")).lower()
    brand  = str(meta.get("brand", "")).lower()
    price  = float(meta.get("sale_price", 0) or 0)
    level_1 = str(meta.get("level_1", "")).upper()
    sleeves = str(meta.get("sleeves", "")).lower()
    if gender and meta.get("level_1", "").upper() != gender:
        return False
    if cat and meta.get("level_2", "").upper() != cat:
        return False
    if exc_c and any(c in colour for c in exc_c):
        return False
    if inc_c and not any(c in colour for c in inc_c):
            return False
    if not relax_brand and inc_b and not any(b in brand for b in inc_b):
        return False
    if exc_b and any(b in brand for b in exc_b):
        return False
    print(f"DEBUG: Checking sleeves - {sleeves_filt} vs {meta.get('sleeves', '')}")
    if sleeves_filt and sleeves_filt not in str(meta.get("sleeves", "")).lower():
        return False
    if not relax_price:
        if pmin is not None and price < pmin:
            return False
        if pmax is not None and price > pmax:
            return False
    return True

def expand_color_list(colors):
    """Expand a list of colors to include similar/related colors."""
    if not colors:
        return []
    
    expanded = set()
    for color in colors:
        color_lower = color.lower().strip()
        # Add the original color
        expanded.add(color_lower)
        # Add all mapped colors
        if color_lower in COLOR_MAPPING:
            expanded.update(COLOR_MAPPING[color_lower])
    
    return list(expanded)

def search_catalog(
    query: List[str] | str,
    *,
    filters: Dict[str, Any] | None = None,
    top_k: int = 10,
):
    """FAISS similarity search w/ 2‑stage price & brand relaxation (unchanged)."""
    note = ""
    txt  = ", ".join(query) if isinstance(query, list) else query
    qvec = _embed_query(txt)
    print('txt-',txt)
    if not filters:
        D, I = faiss_idx.search(qvec, top_k)
        print('no filters')
        return [metas[i] for i in I[0] if i != -1], note

    cat   = (filters.get("category") or "").upper()
    gender = (filters.get("gender") or "").upper()
    inc_b = [b.lower() for b in filters.get("include_brands", [])]
    exc_b = [b.lower() for b in filters.get("exclude_brands", [])]
    original_inc_c = [c.lower() for c in filters.get("include_colors", [])]
    original_exc_c = [c.lower() for c in filters.get("exclude_colors", [])]
    pmin  = filters.get("price_min")
    pmax  = filters.get("price_max")
    sleeves = filters.get("sleeves")

    inc_c = expand_color_list(original_inc_c)
    exc_c = expand_color_list(original_exc_c)
    
    print(f"DEBUG: Original include_colors: {original_inc_c}")
    print(f"DEBUG: Expanded include_colors: {inc_c}")


    for relax_price, relax_brand in [(False, False), (True, False), (True, True)]:
        pool_idx = [
            i
            for i, m in enumerate(metas)
            if _meta_pass(
                m,
                cat=cat,
                inc_c=inc_c,
                exc_c=exc_c,
                inc_b=inc_b,
                exc_b=exc_b,
                gender=gender,
                pmin=pmin,
                pmax=pmax,
                relax_brand=relax_brand,
                relax_price=relax_price,
                sleeves_filt=sleeves
            )
        ]
        if pool_idx:
            if relax_price and not note:
                note = "Price range relaxed."
            if relax_brand:
                note = note or "Brand relaxed – closest matches."
            break
    else:
        return [], "No items match."

    vecs = np.vstack([faiss_idx.reconstruct(i) for i in pool_idx]).astype(np.float32)
    faiss.normalize_L2(vecs)
    sub = faiss.IndexFlatIP(EMBED_DIM)
    sub.add(vecs)
    _, I = sub.search(qvec, min(top_k * 3, len(pool_idx)))
    items = [metas[pool_idx[i]] for i in I[0] if i != -1][:top_k]
    return items, note


def get_sku_meta(sku: str) -> Dict[str, Any] | None:
    for m in metas:
        if str(m.get("sku")) == str(sku):
            return m
    return None

# ─────────────────────────── LangGraph State definition ────────────────────

class AgentState(TypedDict, total=False):
    """Shared mutable state passed between nodes."""
    user_msg: str
    language: str
    style_mem: Optional[str]
    selected_sku: Optional[str]
    last_cart_sku: Optional[str]  # MAKE SURE THIS LINE EXISTS
    last_items: List[Dict[str, Any]]
    chat_hist: List[Dict[str, str]]
    assistant_reply: str
    note: str
    mode: str

async def node_continue(state: AgentState) -> AgentState:
    """Decide whether to continue conversation or end"""
    # You can add logic here to determine if conversation should continue
    # For now, let's always allow continuation
    return state


async def node_catalog(state: AgentState) -> AgentState:
    """
    Handle catalog-search mode.

    NEW:  we embed the *previous memory* (style_mem) into the prompt so that
    attributes such as colour/brand persist when the user only tweaks length,
    silhouette, etc.
    """
    # 1. compose prompt = current user text  +  memory sentence
    combined_prompt = (
        f"{state['user_msg']}\n{state['style_mem']}"
        if state.get("style_mem") else state["user_msg"]
    )

    # 2. extract keywords & filters from the combined prompt
    print(combined_prompt)
    kws = await kw_chain.ainvoke({"query": combined_prompt})
    print(kws)
    filters = filter_chain.invoke({"query": combined_prompt})

    # 3. FAISS similarity search
    items, note = search_catalog(kws, filters=filters, top_k=10)
    state["last_items"] = items
    state["note"] = note

    # 4. concise reply (reuse stylist caption if memory present)
    if state.get("style_mem"):
        cap = await (ChatPromptTemplate.from_messages([
            ("system", 
             "You are a veteran fashion stylist who are very friendly and caring. "
             "Rewrite in ≤30 words, with 1-2 extra styling tip. Get straight to the point but in a soft tone"
             "Your entire scope is within the company catalog only when it comes to specific brands or websites."
             "You may act as general stylist but only within the catalog scope."
             "Strictly do not mention any websites or brands outside the catalog scope."
             "Keep the tone friendly and helpful and human like, do not keep using the same sentences."
             "The user should feel more personal connection with you."
             "Keep the ending open ended, so that the user can ask more questions."
            ),
            ("human", "{sentence}")
        ]) | llm_stylist
).ainvoke({"sentence": state["style_mem"]})
        state["assistant_reply"] = cap.content
    else:
        state["assistant_reply"] = (
            "Here are some options." if items
            else "Sorry, no matching items found."
        )
    return state

async def node_update_state(state: AgentState) -> AgentState:
    """Classify top‑level mode & refresh memory using LLM-based intent detection."""
    
    print(f"DEBUG UPDATE: user_msg = '{state['user_msg']}'")
    print(f"DEBUG UPDATE: last_cart_sku = {state.get('last_cart_sku')}")
    print(f"DEBUG UPDATE: selected_sku = {state.get('selected_sku')}")
    print(f"DEBUG UPDATE: current mode = {state.get('mode')}")
    
    # Check for targeted styling requests if we have EITHER a cart item OR selected item
    has_cart_item = state.get("last_cart_sku") is not None
    has_selected_item = state.get("selected_sku") is not None
    currently_in_sku_mode = state.get("mode") == "sku"
    
    # Trigger styling detection if we have ANY selected item and are in SKU mode
    if (has_cart_item or has_selected_item) and currently_in_sku_mode:
        print("DEBUG: Checking styling intent with LLM...")
        try:
            # Use LLM to determine if this is a styling request
            intent_response = await styling_intent_chain.ainvoke({
                "user_message": state["user_msg"]
            })
            detected_intent = intent_response.content.strip().lower()
            print(f"DEBUG: LLM detected intent = '{detected_intent}'")
            
            if detected_intent == "styling":
                state["mode"] = "targeted_styling"
                print(f"Detected targeted styling request: {state['user_msg']}")
                return state
            elif detected_intent == "new_search":
                print(f"Detected new search context: {state['user_msg']}")
                # Clear cart context and route to catalog
                state["selected_sku"] = None
                state["last_cart_sku"] = None
                state["mode"] = None  # Let it go through normal flow
                
        except Exception as e:
            print(f"DEBUG: Error in intent detection: {e}")
            # Fallback to original logic if LLM fails
    
    # Return early for SKU mode (normal SKU questions) - but not for new_search
    if state.get("mode") == "sku":
        return state
    
    # Original logic for initial searches and other cases
    payload = json.dumps({"prev_memory": state.get("style_mem") or "", "message": state["user_msg"]})
    print(payload)
    upd = await update_state_chain.ainvoke({"payload": payload})
    state["mode"] = upd["mode"]
    new_mem = upd.get("memory") or None
    if state["mode"] == "catalog":
        state["style_mem"] = new_mem
        state["selected_sku"] = None
    elif state["mode"] == "reset":
        # wipe session specific keys
        state["style_mem"] = None
        state["selected_sku"] = None
        state["last_items"] = []
        state["chat_hist"] = []
        print(state)
    return state

async def node_targeted_styling(state: AgentState) -> AgentState:
    """Handle specific styling requests after cart additions or item selections."""
    
    # Try cart SKU first, then fall back to selected SKU
    target_sku = state.get("last_cart_sku") or state.get("selected_sku")
    
    if not target_sku:
        print("No target SKU found, falling back to catalog")
        return await node_catalog(state)
    
    # Get the target item details
    target_meta = get_sku_meta(target_sku)
    if not target_meta:
        print(f"Target item {target_sku} not found, falling back to catalog")
        return await node_catalog(state)
    
    print(f"Using target item {target_sku} for targeted styling")
    
    try:
        # Build context that includes both target item AND previous memory
        context_info = {
            "cart_item": target_meta,  # Using target_meta instead of cart_meta
            "previous_memory": state.get("style_mem") or "",
            "user_request": state["user_msg"]
        }
        
        # Use enhanced targeted styling chain with memory context
        keywords = await memory_aware_style_chain.ainvoke(context_info)
        
        print(f"Generated keywords: {keywords}")
        
        # Extract filters from user request for more precise search
        filters = filter_chain.invoke({"query": state["user_msg"]})
        print(f"Generated filters: {filters}")
        
        # Search for items using the generated keywords and filters
        items, note = search_catalog(keywords, filters=filters, top_k=10)
        
        # Update state with found items
        state["last_items"] = items
        state["note"] = note
        
        if items:
            recs_payload = _reply_items_subset(items, max_items=10)
            resp = await targeted_styling_reply_chain.ainvoke({
                "target_item": json.dumps(target_meta),
                "user_request": state["user_msg"],
                "recs": recs_payload,
            })
            state["assistant_reply"] = resp.content
            #state["assistant_reply"] = f"Here are some great options that would pair perfectly with your selected item:"
        else:
            state["assistant_reply"] = "I couldn't find exactly what you're looking for right now. Let me try a broader search."
            # Fallback: try without filters
            items, note = search_catalog(keywords, top_k=10)
            state["last_items"] = items
            
    except Exception as e:
        print(f"Targeted styling error: {e}")
        # Fallback to regular catalog search
        return await node_catalog(state)
    
    return state

# Add this new memory-aware chain definition:

prompt_memory_aware_style = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a fashion stylist. The user has a specific item in their cart and wants to find complementary items.\n"
            "Consider ALL context to generate relevant search keywords:\n\n"
            "1. **Cart Item Details**: Use the gender, style, formality, color, and brand preferences\n"
            "2. **Previous Context**: Maintain any gender, style preferences, or specific requirements mentioned before\n"
            "3. **Current Request**: Focus on the specific item type requested\n\n"
            "Generate 5-8 targeted search keywords that:\n"
            "- Match the gender from cart item or previous context\n"
            "- Complement the style and formality level\n"
            "- Consider color coordination\n"
            "- Maintain brand/price preferences if mentioned\n\n"
            "- Have basic idea as a fashion stylist about complimentory colors etc while generating keywords\n"
            "- keep specific colors in mind while generating keywords\n"
            "- keep the occassion if any in mind while generating keywords\n(cannot generate casual pants for a formal shirt)\n\n"
            "Return ONLY a JSON array of keywords: [\"keyword1\", \"keyword2\", ...]\n"
            "No explanations, just the keyword array."
        ),
        ("human", 
         "Cart Item: {cart_item}\n\n"
         "Previous Context: {previous_memory}\n\n"
         "User Request: {user_request}")
    ]
)

memory_aware_style_chain = prompt_memory_aware_style | llm_stylist | JsonOutputParser()


async def node_chat(state: AgentState) -> AgentState:
    """Free‑form chat when no SKU selected and not catalog searching."""
    hist = state.get("chat_hist", []) + [{"role": "user", "content": state["user_msg"]}]
    reply = (await concise_chat_chain.ainvoke({"history": hist})).content
    hist.append({"role": "assistant", "content": reply})
    state["chat_hist"] = hist[-20:]  # keep last 20 turns
    state["assistant_reply"] = reply
    return state


async def node_sku(state: AgentState) -> AgentState:
    """Path when a SKU is already selected – decide Q&A vs buy, etc."""
    intent = (await intent_chain.ainvoke({"text": state["user_msg"]})).content.strip().lower()

    if intent == "question_sku":
        meta = get_sku_meta(state["selected_sku"])
        answer = (await qa_chain.ainvoke({"facts": json.dumps(meta), "question": state["user_msg"]})).content
        state["assistant_reply"] = answer

    elif intent == "style":
        # ask for styling suggestions based on the selected SKU
        meta = get_sku_meta(state["selected_sku"])
        if not meta:
            state["assistant_reply"] = "Sorry, I couldn't find that product."
            return state
        
        # Generate targeted keywords based on user request and selected product
        try:
            keywords = await targeted_style_chain.ainvoke({
                "selected_product": json.dumps(meta),
                "user_request": state["user_msg"]
            })
            
            # Search for items using the generated keywords
            items, note = search_catalog(keywords, top_k=10)
            
            # Update state with found items
            state["last_items"] = items
            state["note"] = note
            
            if items:
                recs_payload = _reply_items_subset(items, max_items=10)
                resp = await targeted_styling_reply_chain.ainvoke({
                "target_item": json.dumps(meta),
                "user_request": state["user_msg"],
                "recs": recs_payload,
                })
                state["assistant_reply"] = resp.content
                #state["assistant_reply"] = f"Here are some great options that would pair well with your selected item:"
            else:
                state["assistant_reply"] = "I couldn't find matching items right now. Try describing what you're looking for in more detail."
                
        except Exception as e:
            print(f"Targeted style search error: {e}")
            state["assistant_reply"] = "Let me help you find what you're looking for. Could you be more specific about the type of item you want?"

    elif intent == "buy":
        print(f"DEBUG BUY: selected_sku = {state.get('selected_sku')}")
        meta = get_sku_meta(state["selected_sku"])
        if not meta:
            state["assistant_reply"] = "Sorry, I couldn't find that product."
            return state
        
        # Store the cart item for future reference
        state["last_cart_sku"] = state["selected_sku"]
        print(f"DEBUG BUY: Stored last_cart_sku = {state.get('last_cart_sku')}")
        try:
            # Generate dynamic cart confirmation message
            cart_msg = await cart_confirmation_chain.ainvoke({
                "product_info": json.dumps(meta)
            })
            
            # Generate complementary items
            comp_kws = await comp_kw_chain.ainvoke({"answer": json.dumps(meta)})
            print(comp_kws)
            
            # Search for complementary items
            all_items = []
            for category, keywords in comp_kws.items():
                if isinstance(keywords, list):
                    items, _ = search_catalog(keywords[:3], top_k=4)
                    all_items.extend(items)
            
            # Update state with found items
            state["last_items"] = all_items[:8]
            
            # Combine cart confirmation with complementary suggestions
            if all_items:
                state["assistant_reply"] = f"{cart_msg.content}\n\nHere are some items that we think would help you complete your look:"
            else:
                state["assistant_reply"] = f"{cart_msg.content}"
            print(f"DEBUG BUY END: Final state last_cart_sku = {state.get('last_cart_sku')}")
        except Exception as e:
            print(f"Buy intent error: {e}")
            state["assistant_reply"] = f"🛒 Added SKU {state['selected_sku']} to cart! What else can I help you find?"

        # Log the purchase (or handle it in your e-commerce system
    else:
        # fallback small‑talk
        state = await node_chat(state)
    return state


# Router helper
def router(state: AgentState) -> str:
    mode = state["mode"]
    if mode == "targeted_styling":
        return "targeted_styling"
    return mode

def continue_router(state: AgentState) -> str:
    """Route based on user's next message"""
    user_msg = state.get("user_msg", "").lower()
    
    # Check if user wants to end
    if any(word in user_msg for word in ["bye", "exit", "quit", "thanks", "thank you"]):
        return "end"
    
    # Check if user has selected a SKU from results
    if state.get("selected_sku"):
        return "sku"
    
    # Default to update_state to re-evaluate
    return "update_state"
# ─────────────────────────── Build the graph ───────────────────────────────

graph = StateGraph(AgentState)

# ① entry
graph.add_node("update_state", node_update_state)
graph.add_edge(START, "update_state")
# ② branches
graph.add_conditional_edges(
    "update_state",
    router,
    {
        "reset": END,
        "catalog": "catalog",
        "chat": "chat",
        "sku": "sku",
        "targeted_styling": "targeted_styling",
    },
)

graph.add_node("catalog", node_catalog)
graph.add_edge("catalog", END)  # Back to original

graph.add_node("chat", node_chat)
graph.add_edge("chat", END)  # Back to original

graph.add_node("sku", node_sku)
graph.add_edge("sku", END)  # Back to original

graph.add_node("targeted_styling", node_targeted_styling)
graph.add_edge("targeted_styling", END)  # Back to original


# compile async callable
assistant_graph = graph.compile()

# ─────────────────────────── Session helper ────────────────────────────────
SESSIONS: Dict[str, AgentState] = {}

async def run_graph(session_id: str, user_msg: str, language: str = "English", *, selected_sku: str | None = None) -> AgentState:
    print(f"DEBUG RUN_GRAPH START: session_id = {session_id}")
    if session_id not in SESSIONS:
        print("DEBUG: Creating new session")
        SESSIONS[session_id] = AgentState(
            user_msg="",
            language=language,
            style_mem=None,
            selected_sku=None,
            last_cart_sku=None,
            last_items=[],
            chat_hist=[],
            note="",
            assistant_reply="",
        )
    else:
        print("DEBUG: Loading existing session")
        print(f"DEBUG: Loaded session last_cart_sku = {SESSIONS[session_id].get('last_cart_sku')}")
    state = SESSIONS[session_id]
    state["user_msg"] = user_msg
    state["language"] = language.lower() if language.lower() in SUPPORTED_LANGS else DEFAULT_LANG
    if selected_sku:
        state["selected_sku"] = selected_sku
        state["mode"] = "sku"
    else:
        state.pop("mode", None)
    print(f"DEBUG: Before graph execution, last_cart_sku = {state.get('last_cart_sku')}")
    new_state = await assistant_graph.ainvoke(state)
    print(f"DEBUG: After graph execution, last_cart_sku = {new_state.get('last_cart_sku')}")
    SESSIONS[session_id] = new_state
    print(f"DEBUG: Saved to session, last_cart_sku = {SESSIONS[session_id].get('last_cart_sku')}")
    return new_state


# simple CLI test
if __name__ == "__main__":
    import uuid, asyncio

    async def quick_demo():
        sid = str(uuid.uuid4())
        for q in [
            "Looking for outfits for a black tie event",
            "Maybe something navy instead of black",
            "Third one looks great, tell me about it",
            "Buy",
        ]:
            out = await run_graph(sid, q)
            print("USER:", q)
            print("BOT :", out["assistant_reply"])
            print("MODE:", out["mode"], "\n")
    asyncio.run(quick_demo())
