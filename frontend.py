import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ===================== MODEL LOADING (LAZY) =====================

@st.cache_resource(show_spinner=False)
def load_models():
    """
    Load the trained Random Forest model and scaler from the models/ folder.
    Works both locally and on Streamlit Cloud.
    """
    try:
        # Try local path first
        if os.path.exists("models/rf_suppression_model.pkl"):
            rf_path = "models/rf_suppression_model.pkl"
            scaler_path = "models/scaler.pkl"
        # Try relative to script
        elif os.path.exists(os.path.join(os.path.dirname(__file__), "models", "rf_suppression_model.pkl")):
            rf_path = os.path.join(os.path.dirname(__file__), "models", "rf_suppression_model.pkl")
            scaler_path = os.path.join(os.path.dirname(__file__), "models", "scaler.pkl")
        # Try Streamlit Cloud path
        elif os.path.exists("/mount/src/voiceweave/models/rf_suppression_model.pkl"):
            rf_path = "/mount/src/voiceweave/models/rf_suppression_model.pkl"
            scaler_path = "/mount/src/voiceweave/models/scaler.pkl"
        else:
            return None, None, "Models not found in any expected location."
        
        with open(rf_path, "rb") as f:
            rf_model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        
        return rf_model, scaler, None
    except Exception as e:
        return None, None, f"Error loading models: {str(e)}"


# ===================== PAGE CONFIG & STYLES =====================

st.set_page_config(
    page_title="VoiceWeave – Dialogue Suppression Explorer",
    layout="wide",
)

st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #f8fafc;
        color: #111827;
        font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    h1, h2, h3, h4 {
        color: #0f172a;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1.05rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
    }
    .metric-card {
        padding: 0.8rem 1rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        background-color: #ffffff;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: .06em;
        color: #6b7280;
    }
    .metric-value {
        font-size: 1.2rem;
        font-weight: 600;
        color: #111827;
        margin-top: 0.1rem;
    }
    .section-box {
        padding: 1rem 1.1rem;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        background-color: #ffffff;
    }
    .turn-row {
        padding: 0.45rem 0.2rem;
        border-bottom: 1px solid #e5e7eb;
        font-size: 0.9rem;
    }
    .turn-row:last-child {
        border-bottom: none;
    }
    .turn-speaker {
        font-weight: 500;
        color: #111827;
    }
    .turn-text {
        color: #4b5563;
    }
    .risk-high {
        color: #b91c1c;
        font-weight: 600;
    }
    .risk-moderate {
        color: #c05621;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===================== SHARED LAYOUT HELPERS =====================

def layout_header():
    c1, c2 = st.columns([4, 2])
    with c1:
        st.title("VoiceWeave")
        st.caption(
            "A research prototype for surfacing when contributions in a group dialogue "
            "are at risk of being suppressed."
        )
    with c2:
        st.markdown(
            '<div style="text-align:right;"><span style="font-size:0.8rem;color:#6b7280;">'
            "Models loaded inside app</span></div>",
            unsafe_allow_html=True,
        )


def layout_metrics(summary):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total turns</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value">{summary.get("total_turns", 0)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="metric-label">Unique speakers</div>', unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="metric-value">{summary.get("unique_speakers", 0)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        avg = summary.get("avg_risk", 0.0)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="metric-label">Average suppression risk</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="metric-value">{round(avg * 100, 1)}%</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="metric-label">High-risk turns</div>', unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="metric-value">{summary.get("high_risk_count", 0)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


def build_heatmap(turns):
    if not turns:
        return None
    df = pd.DataFrame(turns)
    df["turn"] = df["turn_index"].astype(int) + 1
    fig = px.imshow(
        df[["suppression_prob"]].T,
        color_continuous_scale="Blues",
        aspect="auto",
        zmin=0.0,
        zmax=1.0,
    )
    fig.update_yaxes(
        tickvals=[0],
        ticktext=["Suppression risk"],
        showgrid=False,
        showline=False,
        tickfont=dict(size=11),
    )
    fig.update_xaxes(
        tickvals=list(range(len(df))),
        ticktext=[str(t) for t in df["turn"]],
        title_text="Turn index",
        showgrid=False,
    )
    fig.update_traces(colorbar_title="Risk")
    fig.update_layout(
        margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    return fig


def render_turn_table(turns):
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    for t in turns:
        prob = t["suppression_prob"]
        label = "suppressed" if prob > 0.5 else "amplified"
        risk_class = ""
        risk_text = ""
        if prob > 0.8:
            risk_class = "risk-high"
            risk_text = "high risk"
        elif prob > 0.65:
            risk_class = "risk-moderate"
            risk_text = "moderate risk"

        st.markdown(
            f"""
            <div class="turn-row">
                <div class="turn-speaker">
                    Turn {t['turn_index'] + 1} · {t['speaker']}
                    <span style="font-weight:400;color:#6b7280;"> · {label}</span>
                    {" · " if risk_text else ""}<span class="{risk_class}">{risk_text}</span>
                </div>
                <div class="turn-text">{t['text']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ===================== ANALYSIS LOGIC =====================

def parse_transcript(text: str):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    rows = []
    for line in lines:
        if "|" not in line:
            continue
        speaker, utt = line.split("|", 1)
        speaker = speaker.strip()
        utt = utt.strip()
        if speaker and utt:
            rows.append({"speaker": speaker, "utterance": utt})
    if len(rows) < 2:
        return None
    df = pd.DataFrame(rows)
    df["dialogue_id"] = 0
    return df


def compute_features(df: pd.DataFrame):
    df["utterance_len"] = df["utterance"].astype(str).str.len()
    df["word_count"] = df["utterance"].astype(str).str.split().str.len()
    df["speakers_in_dialogue"] = df["speaker"].nunique()
    df["speaker_turn_count"] = df.groupby("speaker")["utterance"].transform("count")
    feature_cols = [
        "utterance_len",
        "word_count",
        "speakers_in_dialogue",
        "speaker_turn_count",
    ]
    X = df[feature_cols].fillna(0).astype(float).values
    return X


def run_model_on_transcript(text: str, rf_model, scaler):
    df = parse_transcript(text)
    if df is None:
        return None, "Need at least 2 turns in format Speaker|text."

    X = compute_features(df)

    if X.shape[1] != rf_model.n_features_in_ or X.shape[1] != scaler.n_features_in_:
        return None, (
            f"Feature mismatch: X has {X.shape[1]}, "
            f"model expects {rf_model.n_features_in_}, "
            f"scaler expects {scaler.n_features_in_}."
        )

    X_scaled = scaler.transform(X)
    probs = rf_model.predict_proba(X_scaled)[:, 1]

    turns = []
    recs = []
    for i, row in df.reset_index(drop=True).iterrows():
        p = float(probs[i])
        status = "suppressed" if p > 0.5 else "amplified"
        turns.append(
            {
                "turn_index": int(i),
                "speaker": row["speaker"],
                "text": row["utterance"][:200],
                "suppression_prob": p,
                "status": status,
            }
        )
        if p > 0.65:
            recs.append(
                {
                    "turn_index": int(i),
                    "speaker": row["speaker"],
                    "risk": "high" if p > 0.80 else "moderate",
                    "probability": p,
                    "suggestion": (
                        f"Invite {row['speaker']} to expand on this turn; "
                        "their contribution is at elevated risk of being suppressed."
                    ),
                }
            )

    summary = {
        "total_turns": int(len(df)),
        "unique_speakers": int(df["speaker"].nunique()),
        "avg_risk": float(np.mean(probs)),
        "high_risk_count": int(sum(p > 0.65 for p in probs)),
    }

    return {"summary": summary, "turns": turns, "recommendations": recs}, None

# ===================== PAGES =====================

def page_about():
    st.header("What this prototype does")
    st.write(
        """
        VoiceWeave estimates, for each turn in a group conversation, how likely it is that the turn behaves 
        like a suppressed contribution rather than an amplified one. The underlying model is trained on a 
        labeled corpus of approximately 38,000 dialogue turns, each annotated as suppressed or amplified.
        """
    )
    st.write(
        """
        The current version focuses on structural cues in the transcript: who speaks, how often they appear, 
        and how long their turns are. It does not yet use word meaning or audio tone.
        """
    )
    st.header("How the analysis works")
    st.markdown(
        """
        - Utterance length in characters  
        - Word count  
        - Number of distinct speakers in the dialogue  
        - Number of turns contributed by this speaker in the dialogue  
        """
    )
    st.write(
        """
        These features are standardized and passed into a Random Forest classifier trained on labeled data.
        """
    )


def page_how_to_use():
    st.header("How to use this interface")
    st.subheader("Input format")
    st.write("`SpeakerName|utterance text`")
    st.write(
        """
        Example:
        - `Alex|I think the budget is tight.`  
        - `Priya|I disagree.`  
        - `Facilitator|Who else has thoughts?`
        """
    )
    st.subheader("What the model analyzes")
    st.markdown(
        """
        - Structure only: turn length, speaker frequency, dialogue diversity  
        - Not yet: word meaning, tone, or audio  
        """
    )


def page_analyze():
    st.header("Analyze a transcript")

    if "transcript" not in st.session_state:
        st.session_state["transcript"] = ""

    col_buttons = st.columns([1, 1, 6])
    with col_buttons[0]:
        load_clicked = st.button("Load example", use_container_width=True)
    with col_buttons[1]:
        analyze_clicked = st.button("Analyze transcript", use_container_width=True)

    if load_clicked:
        example = (
            "Facilitator|Thanks everyone for joining. The goal today is to discuss our new community program.\n"
            "Alex|I think the budget is tight, and we should start small.\n"
            "Priya|I am worried that if we start small, some neighborhoods will be left out.\n"
            "Ravi|Maybe we can rotate the pilot locations every few months.\n"
            "Alex|Rotations are fine, but we still do not have clarity on who decides the order.\n"
            "Facilitator|That is a good point. Who feels their perspective has not been heard yet?\n"
            "Sara|I have been quiet because I am new here, but I think residents should nominate pilot sites.\n"
            "Alex|If residents nominate, we might get pressure from louder groups.\n"
            "Priya|That is exactly why we need a clear process so quieter groups are not sidelined.\n"
            "Facilitator|Thank you. Let us slow down and hear from people who have not spoken much yet.\n"
        )
        st.session_state["transcript"] = example

    st.text_area(
        "Transcript",
        value=st.session_state["transcript"],
        key="transcript",
        height=260,
    )

    st.header("Analysis results")

    if analyze_clicked:
        transcript = st.session_state["transcript"].strip()
        if not transcript:
            st.warning("Please paste a transcript or load the example.")
            return

        # Load models only when needed
        rf_model, scaler, error = load_models()
        if error:
            st.error(f"Cannot load models: {error}")
            return

        result, error = run_model_on_transcript(transcript, rf_model, scaler)
        if error is not None:
            st.error(error)
            return

        summary = result["summary"]
        turns = result["turns"]
        recs = result["recommendations"]

        layout_metrics(summary)

        st.subheader("Suppression heatmap")
        fig = build_heatmap(turns)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Facilitation recommendations")
        if recs:
            st.markdown('<div class="section-box">', unsafe_allow_html=True)
            for r_item in recs:
                risk_label = "High" if r_item["risk"] == "high" else "Moderate"
                prob_pct = round(r_item["probability"] * 100, 1)
                st.markdown(
                    f"""
                    <div class="turn-row">
                        <div class="turn-speaker">
                            Turn {r_item['turn_index'] + 1} · {r_item['speaker']}
                            <span style="color:#6b7280;font-weight:400;"> · {risk_label} · {prob_pct}%</span>
                        </div>
                        <div class="turn-text">{r_item['suggestion']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No high-risk turns detected.")

        st.subheader("Turn-by-turn view")
        render_turn_table(turns)
    else:
        st.info("Load the example or paste a transcript, then click Analyze.")


def page_research():
    st.header("End-to-end pipeline")
    st.markdown(
        """
        1. **Data**: ~38,000 labeled dialogue turns  
        2. **Features**: Length, word count, speaker diversity, turn frequency  
        3. **Model**: Random Forest on standardized features  
        4. **App**: Loads model, analyzes new transcripts in real time  
        """
    )
    st.subheader("Current vs. Future")
    st.markdown(
        """
        | Now | Next |
        |-----|------|
        | Structural features | Semantic & prosodic |
        | Transcript analysis | Live transcription |
        | Single turn labels | Session-level insights |
        """
    )

# ===================== MAIN =====================

def main():
    layout_header()
    st.markdown("---")

    pages = {
        "About": page_about,
        "How to use": page_how_to_use,
        "Analyze": page_analyze,
        "Research": page_research,
    }
    choice = st.sidebar.radio("Navigation", list(pages.keys()))
    pages[choice]()


if __name__ == "__main__":
    main()
