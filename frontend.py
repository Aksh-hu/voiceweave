import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ---------- MODEL LOADING (LOCAL, NO FLASK) ----------

@st.cache_resource(show_spinner=True)
def load_models():
    """
    Load the trained Random Forest model and scaler from the models/ folder.
    Assumes the repository structure:
    D:/VoiceWeave_MPDD/
      ├── frontend.py
      ├── models/
      │   ├── rf_suppression_model.pkl
      │   └── scaler.pkl
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")

    rf_path = os.path.join(models_dir, "rf_suppression_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    with open(rf_path, "rb") as f:
        rf_model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return rf_model, scaler


rf_model, scaler = load_models()

# ---------- STREAMLIT PAGE CONFIG / STYLES ----------

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

# ---------- SHARED LAYOUT HELPERS ----------


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
            '<div style="text-align:right;"><span class="metric-label">'
            "Models loaded inside app</span></div>",
            unsafe_allow_html=True,
        )
        st.write("")


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
            '<div class="metric-label">High‑risk turns</div>', unsafe_allow_html=True
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

# ---------- CORE ANALYSIS LOGIC (NO API) ----------


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


def run_model_on_transcript(text: str):
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

# ---------- PAGES ----------


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
        and how long their turns are. It does not yet use word meaning or audio tone. The goal is to make 
        basic participation patterns visible and inspectable before adding more complex signals.
        """
    )
    st.header("How the analysis works")
    st.write(
        """
        For each utterance in a transcript, the app computes four structural features:
        """
    )
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
        These features are standardized and fed into a Random Forest classifier trained on the labeled dataset. 
        The model outputs a suppression probability for each turn, which is then summarized as metrics, a 
        suppression heatmap, and per‑turn recommendations.
        """
    )
    st.write(
        """
        During model development, mechanistic interpretability tools such as feature importance analysis and 
        turn‑level inspection were used offline to understand how these structural features influence predictions. 
        This helps reveal patterns like: conversations where one speaker holds many long turns tend to generate 
        higher suppression scores for shorter, infrequent contributions from others.
        """
    )


def page_how_to_use():
    st.header("How to use this interface")
    st.subheader("Input format")
    st.write(
        """
        Paste a transcript where each line follows:

        `SpeakerName|utterance text`
        """
    )
    st.write(
        """
        For example:

        `Alex|I think the budget is tight and we should start small.`  
        `Priya|I am worried that some neighborhoods will be left out.`  
        `Facilitator|Who has not had a chance to speak yet?`
        """
    )
    st.subheader("What the model analyzes")
    st.markdown(
        """
        - It uses only the structure of the transcript: length of each turn, number of speakers, and how often each speaker appears.  
        - It does not yet use word meaning, topics, or sentiment.  
        - It does not yet use audio information such as tone, pitch, or interruptions.  
        """
    )
    st.write(
        """
        Even with these constraints, structural patterns already capture useful signals about participation balance. 
        The model uses these patterns, learned from the training corpus, to estimate suppression risk per turn.
        """
    )
    st.subheader("Reading the output")
    st.markdown(
        """
        - Metric cards summarize the size and diversity of the dialogue and overall risk.  
        - The heatmap shows suppression risk by turn index, highlighting where risk spikes.  
        - Recommendations point to specific turns where inviting a speaker back in may be helpful.  
        - The turn‑by‑turn view lets you read each utterance with its risk label and risk level.
        """
    )
    st.write(
        """
        These outputs are designed as prompts for reflection and facilitation, not as definitive labels about people or groups.
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
        "Transcript (one line per turn, format: Speaker|text)",
        value=st.session_state["transcript"],
        key="transcript",
        height=260,
    )

    st.header("Analysis results")

    if analyze_clicked:
        transcript = st.session_state["transcript"].strip()
        if not transcript:
            st.warning("Please paste a transcript or load the example before analyzing.")
            return

        result, error = run_model_on_transcript(transcript)
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
        else:
            st.info("No turns available for visualization.")

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
                            <span style="color:#6b7280;font-weight:400;">
                                · {risk_label} risk · {prob_pct}%
                            </span>
                        </div>
                        <div class="turn-text">{r_item['suggestion']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No turns crossed the high‑risk threshold in this transcript.")

        st.subheader("Turn‑by‑turn view")
        render_turn_table(turns)
    else:
        st.info("Load the example or paste a transcript, then click “Analyze transcript”.")
        st.write("Once you analyze, results will appear here: metrics, heatmap, and turn‑by‑turn view.")


def page_research():
    st.header("End‑to‑end pipeline")

    st.write(
        """
        The current prototype follows this pipeline:
        """
    )
    st.markdown(
        """
        1. **Data assembly**  
           - Labeled dialogue corpus with approximately 38,000 utterances.  
           - Each turn includes dialogue id, speaker id, turn index, utterance text, emotion, and a label: suppressed or amplified.  

        2. **Feature engineering (training)**  
           - For each turn, compute:  
             - Utterance length in characters  
             - Word count  
             - Number of distinct speakers in the dialogue  
             - Number of turns contributed by this speaker in the dialogue  
           - Standardize these four features with a `StandardScaler`.  

        3. **Model training**  
           - Train a Random Forest classifier on the four standardized features to predict suppressed vs amplified.  
           - Evaluate on a held‑out test split with metrics such as accuracy and AUC.  
           - Use feature importance and turn‑level inspection offline to understand which structural patterns drive predictions.  

        4. **Deployment in this app**  
           - The trained Random Forest and scaler are loaded directly inside this Streamlit app.  
           - For any new transcript, the app recomputes the same four features, applies the saved scaler, and runs the trained model.  
           - The app then visualizes per‑turn suppression probabilities plus summary statistics and recommendations.  
        """
    )

    st.subheader("What is included now vs. future work")
    st.markdown(
        """
        | Aspect             | Current prototype                                              | Planned extensions                                      |
        |--------------------|----------------------------------------------------------------|---------------------------------------------------------|
        | Data               | Labeled transcript corpus (~38k turns)                        | Additional datasets and domains                         |
        | Features           | Length, word count, speaker diversity, speaker turn count      | Semantic and prosodic descriptors                       |
        | Model              | Random Forest classifier                                      | Alternative models and baselines                        |
        | Interpretability   | Structural feature importance and turn‑level inspection        | Deeper analysis of feature interactions and dynamics    |
        | Interface          | Transcript analysis, heatmap, recommendations, turn table      | Multi‑session views, comparison between conversations   |
        | Deployment         | Streamlit app with embedded model                             | Integration with live transcription for facilitation    |
        """
    )

    st.subheader("Live facilitation vision")
    st.write(
        """
        The same analysis used here for static transcripts can be connected to a streaming transcription service. 
        In that setting, each new utterance would be processed as it arrives, structural features would be updated 
        incrementally, and suppression risk would be computed in near real time. A facilitator could then use a 
        live dashboard to notice which speakers or turns are consistently flagged as high risk and adjust their 
        interventions during the conversation, rather than only after it ends.
        """
    )

# ---------- MAIN ----------


def main():
    layout_header()
    st.markdown("---")

    pages = {
        "About": page_about,
        "How to use": page_how_to_use,
        "Analyze": page_analyze,
        "Research background": page_research,
    }
    choice = st.sidebar.radio("Navigation", list(pages.keys()))
    pages[choice]()


if __name__ == "__main__":
    main()
