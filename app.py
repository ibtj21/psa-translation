"""
PSA Translation Demo — English/Kiswahili -> Ekegusii
Streamlit app serving the fine-tuned mT5-small and NLLB-200-distilled-600M models.

Expects the two fine-tuned model folders (produced by mt5_training.ipynb and
nllb_training.ipynb) to be available at the paths below -- update MT5_MODEL_DIR /
NLLB_MODEL_DIR if you place them elsewhere, or push them to the Hugging Face Hub
and swap these for repo ids (e.g. "your-username/mt5-en-guz").

Run locally with:
    streamlit run app.py
"""

import os
import torch
import pandas as pd
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import repo_exists
import gc

MT5_MODEL_DIR = "HanaHailemariam/mt5-en-guz"
NLLB_MODEL_DIR = "HanaHailemariam/nllb-en-guz"
NLLB_TGT_PLACEHOLDER = "swh_Latn"  # placeholder tag for Ekegusii (unsupported by NLLB-200)
MAX_LEN = 128
FEEDBACK_PATH = "logs/feedback.csv"
NO_EXAMPLE = "— Type your own —"

st.set_page_config(page_title="PSA Translator — Ekegusii", page_icon="🌍")


# ---------------------------------------------------------------------------
# Model loading -- guarded so a missing checkpoint gives a clear message
# instead of a raw traceback (e.g. if this is opened before training finishes).
# ---------------------------------------------------------------------------


MT5_AVAILABLE = repo_exists(MT5_MODEL_DIR)
NLLB_AVAILABLE = repo_exists(NLLB_MODEL_DIR)

if not MT5_AVAILABLE and not NLLB_AVAILABLE:
    st.error(
        "No trained model checkpoints found yet.\n\n"
        f"Expected a finished model at `{MT5_MODEL_DIR}` and/or `{NLLB_MODEL_DIR}`. "
        "Run `mt5_training.ipynb` and/or `nllb_training.ipynb` to completion first, "
        "then reload this app."
    )
    st.stop()




import gc

@st.cache_resource
def _model_tracker():
    # A mutable container st.cache_resource keeps alive across reruns AND across
    # every user session -- this is what actually tracks which model is loaded app-wide.
    return {"which": None}


@st.cache_resource
def _load_model_resource(which):
    model_dir = MT5_MODEL_DIR if which == "mt5" else NLLB_MODEL_DIR
    dtype = torch.bfloat16 if which == "nllb" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, dtype=dtype)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return tok, model


def _load_model(which):
    tracker = _model_tracker()
    previously = tracker["which"]
    if previously is not None and previously != which:
        st.cache_resource.clear()
        gc.collect()
        tracker = _model_tracker()
    tracker["which"] = which
    return _load_model_resource(which)


def load_mt5():
    return _load_model("mt5")


def load_nllb():
    return _load_model("nllb")


def mt5_prefix(source_lang):
    return "translate English to Ekegusii: " if source_lang == "en" else "translate Kiswahili to Ekegusii: "


def nllb_src_code(source_lang):
    return "eng_Latn" if source_lang == "en" else "swh_Latn"


def _seq_confidence(model, out):
    """Mean exponentiated log-probability of the generated tokens, roughly 0-1."""
    if getattr(out, "beam_indices", None) is not None:
        transition_scores = model.compute_transition_scores(out.sequences, out.scores, out.beam_indices, normalize_logits=True)
    else:
        transition_scores = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
    mask = transition_scores > -1e9
    seq_logprob = (transition_scores * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return torch.exp(seq_logprob).item()


def translate_mt5(text, source_lang):
    tok, model = load_mt5()
    inputs = tok(mt5_prefix(source_lang) + text, return_tensors="pt",
                 truncation=True, max_length=MAX_LEN).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=60,
                              num_beams=4, early_stopping=True,
                              no_repeat_ngram_size=3, repetition_penalty=1.3,
                              output_scores=True, return_dict_in_generate=True)
    text_out = tok.decode(out.sequences[0], skip_special_tokens=True)
    return text_out, _seq_confidence(model, out)


def translate_nllb(text, source_lang):
    tok, model = load_nllb()
    tok.src_lang = nllb_src_code(source_lang)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(model.device)
    forced_bos = tok.convert_tokens_to_ids(NLLB_TGT_PLACEHOLDER)
    with torch.no_grad():
        out = model.generate(**inputs, forced_bos_token_id=forced_bos, max_length=60,
                              num_beams=4, early_stopping=True,
                              no_repeat_ngram_size=3, repetition_penalty=1.3,
                              output_scores=True, return_dict_in_generate=True)
    text_out = tok.decode(out.sequences[0], skip_special_tokens=True)
    return text_out, _seq_confidence(model, out)


@st.cache_data
def load_example_psas(n=6):
    """Pull real held-out PSA sentences from the saved test-set predictions CSV
    (written by section 4.5 of the training notebooks), so the demo shows genuine
    examples rather than made-up placeholder text. Falls back to a small hardcoded
    set if neither predictions file exists yet (e.g. running this before training)."""
    for path in ["logs/mt5_test_predictions.csv", "logs/nllb_test_predictions.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path).drop_duplicates(subset=["source_text"])
            sample = df.sample(min(n, len(df)), random_state=42)
            return [
                (f"{row.Domain} — {'English' if row.source_lang == 'en' else 'Kiswahili'}",
                 row.source_lang, row.source_text)
                for row in sample.itertuples()
            ]
    return [
        ("Health — English", "en",
         "Parents are encouraged to complete their children's vaccination schedule at the nearest clinic."),
        ("Agriculture — English", "en",
         "Farmers are urged to prioritize safe agrochemical usage this season."),
        ("Health — Kiswahili", "sw",
         "Wazazi wanahimizwa kukamilisha ratiba ya chanjo ya watoto wao katika kituo cha afya kilicho karibu."),
        ("Agriculture — Kiswahili", "sw",
         "Wakulima wanahimizwa kutumia kemikali za kilimo kwa usalama msimu huu."),
    ]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🌍 PSA Translator: English/Kiswahili → Ekegusii")
st.caption(
    "Fine-tuned mT5-small and NLLB-200-distilled-600M models trained on a curated "
    "Kenyan Public Service Announcement corpus, targeting **Ekegusii** — a Bantu "
    "language with no coverage in either model's original pretraining data. "
    "**Target language is fixed to Ekegusii**; the selector below picks the *source* "
    "language you're translating from."
)

examples = load_example_psas()
example_labels = [NO_EXAMPLE] + [e[0] for e in examples]

if "text_input" not in st.session_state:
    st.session_state.text_input = ""
if "source_lang_select" not in st.session_state:
    st.session_state.source_lang_select = "English"


def _apply_example():
    choice = st.session_state.example_choice
    if choice != NO_EXAMPLE:
        idx = example_labels.index(choice) - 1
        _, ex_lang, ex_text = examples[idx]
        st.session_state.text_input = ex_text
        st.session_state.source_lang_select = "English" if ex_lang == "en" else "Kiswahili"


st.selectbox("Try an example PSA (optional)", example_labels,
             key="example_choice", on_change=_apply_example)

model_options = []
if MT5_AVAILABLE:
    model_options.append("mT5-small")
if NLLB_AVAILABLE:
    model_options.append("NLLB-200-distilled-600M")
if MT5_AVAILABLE and NLLB_AVAILABLE:
    model_options.append("Both (compare)")

col1, col2 = st.columns(2)
with col1:
    source_lang_label = st.selectbox("Source language", ["English", "Kiswahili"], key="source_lang_select")
    source_lang = "en" if source_lang_label == "English" else "sw"
with col2:
    model_choice = st.selectbox("Model", model_options)
    if model_choice in ("NLLB-200-distilled-600M", "Both (compare)"):
        st.caption("Note: NLLB may occasionally output Kiswahili instead of Ekegusii — a known limitation, see poster for details.")
text = st.text_area(
    "Sentence to translate",
    key="text_input",
    placeholder="e.g. Farmers are urged to prioritize safe agrochemical usage this season.",
    height=100,
)

if st.button("Translate", type="primary", disabled=not text.strip()):
    results = {}
    with st.spinner("Translating..."):
        if model_choice in ("mT5-small", "Both (compare)") and MT5_AVAILABLE:
            results["mT5-small"] = translate_mt5(text, source_lang)
        if model_choice in ("NLLB-200-distilled-600M", "Both (compare)") and NLLB_AVAILABLE:
            results["NLLB-200-distilled-600M"] = translate_nllb(text, source_lang)
    st.session_state.last_translations = results
    st.session_state.last_source_text = text
    st.session_state.last_source_lang = source_lang_label
    st.session_state.last_model_choice = model_choice

if st.session_state.get("last_translations"):
    for model_name, (out, conf) in st.session_state.last_translations.items():
        st.subheader(model_name)
        st.success(out)
        st.caption(f"Confidence: {conf:.0%}  *(mean token probability -- a rough signal, not a guarantee)*")

    st.divider()
    st.subheader("📝 Feedback")
    with st.form("feedback_form", clear_on_submit=True):
        rating = st.radio("How was this translation?", ["👍 Good", "👎 Needs work"], horizontal=True)
        comment = st.text_area("Optional comment (what was wrong, or what worked well)", height=68)
        submitted = st.form_submit_button("Submit feedback")
        if submitted:
            os.makedirs("logs", exist_ok=True)
            row = pd.DataFrame([{
                "timestamp": pd.Timestamp.now().isoformat(),
                "source_text": st.session_state.last_source_text,
                "source_lang": st.session_state.last_source_lang,
                "model_choice": st.session_state.last_model_choice,
                "rating": rating,
                "comment": comment,
            }])
            row.to_csv(FEEDBACK_PATH, mode="a", header=not os.path.exists(FEEDBACK_PATH), index=False)
            st.success("Thanks — feedback saved.")
    st.caption(f"Feedback is stored locally on this server at `{FEEDBACK_PATH}`, not sent anywhere else.")

st.divider()
st.caption(
    "Note: Ekegusii has no native language code in NLLB-200, so a placeholder target-language "
    "tag is used to steer generation. Both models were fine-tuned with layer freezing on a "
    "low-resource English/Kiswahili → Ekegusii dataset; see the accompanying training notebooks "
    "and results tables for hyperparameters and evaluation metrics."
)
