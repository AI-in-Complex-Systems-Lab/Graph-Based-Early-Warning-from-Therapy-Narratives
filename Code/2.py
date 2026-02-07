# --- System and Libraries ---
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import joblib

# --- Visualization ---
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from pyvis.network import Network

# --- Streamlit Page Config --------
st.set_page_config(
    page_title="Temporal Graph Mining of Personal Storylines",
    layout="wide"
)


import matplotlib.pyplot as plt
import plotly.io as pio


from pathlib import Path
from io import BytesIO

EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(exist_ok=True)


# background
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# background
pio.templates.default = "plotly_white"

# ------------------------ THEME & CSS ------------------------
BG      = "#FFFFFF"   # app + charts background
TEXT    = "#2E2157"   # deep eggplant text
MUTED   = "#7D6FA2"   # muted eggplant (ticks, borders)
ACCENT  = "#2AB3A6"   # teal buttons
ACCENT_D= "#1F958A"   # teal hover
POS     = "#F1C40F"   # golden (positive)
NEU     = "#3FB0CF"   # cyan (neutral)
NEG     = "#B6465F"   # raspberry (negative)

st.markdown(
    f"""
    <style>
    /* ========= Base app ========= */
    .stApp {{
        background: #FFFFFF !important;
        color: {TEXT} !important;
        font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
    }}

    /* ========= Headings ========= */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    h1, h2, h3 {{
        color: {TEXT} !important;
        letter-spacing: 0.2px;
    }}

    /* ========= labels ========= */
    label, .stMarkdown p, .stRadio > label, .stSelectbox > label, .stMultiSelect > label {{
        background: transparent !important;
        color: {TEXT} !important;
        padding: 0 !important;
        border: none !important;
        box-shadow: none !important;
    }}

    /* ========= Inputs / selects / multiselect chips ========= */
    .stTextInput input, .stTextArea textarea, .stNumberInput input,
    .stDateInput input, .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div {{
        background: #FCFAF6 !important;
        color: {TEXT} !important;
        border: 1px solid #D8CFBF !important;
        border-radius: 12px !important;
        box-shadow: none !important;
    }}

    .stMultiSelect span[data-baseweb="tag"] {{
        background: #EFE9DD !important;
        color: {TEXT} !important;
        border-radius: 12px !important;
        border: 1px solid #D8CFBF !important;
    }}

    /* ========= Buttons ========= */
    .stButton button, .stDownloadButton button {{
        background: {ACCENT} !important;
        color: white !important;
        border-radius: 14px !important;
        border: none !important;
        padding: 0.6rem 1.25rem !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 0 rgba(0,0,0,0.05) !important;
    }}
    .stButton button:hover, .stDownloadButton button:hover {{
        background: {ACCENT_D} !important;
    }}

    /* ========= Sidebar: compact + legible ========= */
    section[data-testid="stSidebar"] {{
        background: #FFFFFF !important;
        border-right: 1px solid #E7DECD !important;
    }}
    section[data-testid="stSidebar"] .block-container {{
        padding-top: 0.75rem !important;
        padding-bottom: 0.75rem !important;
        padding-left: 0.9rem !important;
        padding-right: 0.9rem !important;
    }}
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {{
        margin-top: 0.6rem !important;
        margin-bottom: 0.4rem !important;
        color: {TEXT} !important;
    }}
    section[data-testid="stSidebar"] label {{
        font-size: 0.92rem !important;
        color: {TEXT} !important;
        margin-bottom: 0.25rem !important;
    }}
    /* Sidebar text in all states */
    section[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
    section[data-testid="stSidebar"] [data-baseweb="radio"] label *,
    section[data-testid="stSidebar"] [role="radiogroup"] label *,
    section[data-testid="stSidebar"] [data-baseweb="checkbox"] label *,
    section[data-testid="stSidebar"] [data-baseweb="select"] div,
    section[data-testid="stSidebar"] .stMultiSelect label span,
    section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] * {{
        color: {TEXT} !important;
    }}
    section[data-testid="stSidebar"] [data-baseweb="radio"] label:hover *,
    section[data-testid="stSidebar"] [data-baseweb="radio"] input:checked + div * {{
        color: {TEXT} !important;
    }}
    section[data-testid="stSidebar"] span[data-baseweb="tag"] {{
        background: #EFE9DD !important;
        border: 1px solid #D8CFBF !important;
        color: {TEXT} !important;
    }}

    /* ========= Custom tab headers ========= */
    .tab-header {{
        color: {TEXT} !important;
        font-weight: 700;
        margin-right: 1.5rem;
        cursor: pointer;
        opacity: 0.85;
    }}
    .tab-header.active {{
        color: {NEG} !important;
        border-bottom: 3px solid {NEG};
        padding-bottom: 6px;
        opacity: 1;
    }}

    /* ========= Dataframes ========= */
    .stDataFrame, .stDataFrame table {{ color: {TEXT} !important; }}

    /* ========= Plotly UI & legend text ========= */
    .js-plotly-plot .plotly .modebar-btn path {{ fill: {TEXT}; }}
    .js-plotly-plot .legend text {{ fill: {TEXT} !important; color: {TEXT} !important; }}

    /* ========= File uploader ========= */
    .stFileUploader div[data-testid="stFileDropzone"] {{
        background-color: #FCFAF6 !important;
        border: 1.5px solid #D8CFBF !important;
        border-radius: 12px !important;
        color: {TEXT} !important;
    }}
    .stFileUploader div[data-testid="stFileDropzone"] * {{ color: {TEXT} !important; }}
    .stFileUploader div[data-testid="stFileDropzone"] svg,
    .stFileUploader div[data-testid="stFileDropzone"] svg * {{
        stroke: {TEXT} !important; 
        fill: transparent !important;
    }}
    .stFileUploader button {{
        background-color: {ACCENT} !important;
        color: #FFFFFF !important;
        border: 1px solid #D8CFBF !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }}
    .stFileUploader button:hover {{ background-color: {ACCENT_D} !important; }}

    /* ========= PyVis container ========= */
    div[data-testid="stIFrame"], 
    div[data-testid="stHtml"] {{
        background: #FCFAF6 !important;
        border: 1.5px solid #D8CFBF !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }}
    div[data-testid="stIFrame"] > iframe,
    div[data-testid="stHtml"] > iframe {{
        background: #FFFFFF !important;
    }}

    /* ========= this is to fix the dark focus ring  ..... ========= */
    .stTextInput input,
    .stTextArea textarea,
    .stNumberInput input,
    .stDateInput input {{
        outline: none !important;
        box-shadow: none !important;
    }}

    .stTextInput:focus-within > div > div,
    .stTextArea:focus-within > div > div,
    .stNumberInput:focus-within > div > div,
    .stDateInput:focus-within > div > div {{
        box-shadow: none !important;
        border: 1px solid #D8CFBF !important;
        border-radius: 12px !important;
    }}
    /* Select / Multiselect (BaseWeb) */
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div {{
        box-shadow: none !important;
        border: 1px solid #D8CFBF !important;
        border-radius: 12px !important;
    }}
    .stSelectbox div[data-baseweb="select"] > div:focus,
    .stSelectbox div[data-baseweb="select"] > div:focus-within,
    .stMultiSelect div[data-baseweb="select"] > div:focus,
    .stMultiSelect div[data-baseweb="select"] > div:focus-within {{
        outline: none !important;
        box-shadow: none !important;
        border: 1px solid #D8CFBF !important;
    }}

    div[data-baseweb="input"] > div:focus-within,
    div[role="spinbutton"]:focus {{
        outline: none !important;
        box-shadow: none !important;
    }}

    .stNumberInput div[role="spinbutton"],
    .stNumberInput div[role="spinbutton"]:focus {{
        outline: none !important;
        box-shadow: none !important;
    }}

    .stTextInput input,
    .stTextArea textarea,
    .stNumberInput input,
    .stDateInput input {{
      color: {TEXT} !important;
      caret-color: {TEXT} !important;
    }}

    /* BaseWeb select & multiselect */
    [data-baseweb="select"] div > div,
    [data-baseweb="select"] input {{
      color: {TEXT} !important;
    }}

 
    .stNumberInput div[data-baseweb="base-input"] input {{
      color: {TEXT} !important;
    }}

    /* Streamlit alert boxes */
    div[role="alert"], .stAlert {{
      color: {TEXT} !important;
    }}
    div[role="alert"] *, .stAlert * {{
      color: {TEXT} !important;
    }}
    
    .stAlert[data-baseweb="notification"] {{
      background: #FBEAEA !important;
      border: 1px solid #E7C0C0 !important;
    }}


    </style>
    """,
    unsafe_allow_html=True
)



# --- OpenAI (NVIDIA) API Client ----------
client = OpenAI(
    api_key="nvapi-", # note from Rawan to the user: Use your API Key
    base_url="https://integrate.api.nvidia.com/v1"
)

def get_model_response(model_name, prompt):
    try:
        model_map = {
            "mixtral": "mistralai/mixtral-8x7b-instruct-v0.1",
            "llama3": "meta/llama3-8b-instruct",
            "qwen2": "qwen/qwen2-7b-instruct",
            "llama4": "meta/llama-4-maverick-17b-128e-instruct",
            "phi4": "microsoft/phi-4-mini-instruct",
            "gemma3": "google/gemma-3n-e4b-it",
            "qwen3": "qwen/qwen3-235b-a22b",
            "dbrx": "databricks/dbrx-instruct",
            "roberta-goemotions": "local"
        }
        if model_name == "roberta-goemotions":
            return "[Local model selected for emotion annotation. Use CSV or narrative generation.]"

        response = client.chat.completions.create(
            model=model_map.get(model_name, model_name),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            top_p=0.7,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error from {model_name}: {str(e)}]"

# -------- App Header ---
col1, col2, col3 = st.columns([1.2, 5, 1])
with col1:
    st.image("Code/ai_logo.png", width=140)
with col2:
    st.markdown(
        f"<h1 style='text-align:center; color:{TEXT}; margin-bottom:0.4rem;'>"
        "Temporal Graph Mining of Personal Storylines for Early Detection of Cognitive and Emotional Decline"
        "</h1>",
        unsafe_allow_html=True
    )

# --- Device -------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Encoders for GNN --------
emotion_encoder = joblib.load("Code/narrative_emotion_encoder.pkl")
status_encoder = joblib.load("Code/session_status_encoder.pkl")

# -------- Load Embedding Model for GNN ---
embedder = SentenceTransformer("intfloat/e5-large-v2")

# --- Simplified GNN ---
from torch_geometric.nn import GATv2Conv, GlobalAttention
class GATv2MultiTask_Small(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_node_classes, num_graph_classes, heads=2, dropout=0.35):
        super().__init__()
        self.gat1 = GATv2Conv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.ln1  = nn.LayerNorm(hidden_dim * heads)
        self.gat2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout)
        self.ln2  = nn.LayerNorm(hidden_dim)
        self.global_attention = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        )
        self.dropout = nn.Dropout(dropout)
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_node_classes)
        )
        self.graph_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_graph_classes)
        )
    def forward(self, x, edge_index, batch):
        x = self.gat1(x, edge_index); x = self.ln1(x); x = F.elu(x)
        x = self.gat2(x, edge_index); x = self.ln2(x); x = F.elu(x)
        x = self.dropout(x)
        node_out = self.node_classifier(x)
        graph_emb = self.global_attention(x, batch)
        graph_out = self.graph_classifier(graph_emb)
        return node_out, graph_out

model = GATv2MultiTask_Small(
    input_dim=1025, hidden_dim=64,
    num_node_classes=len(emotion_encoder.classes_),
    num_graph_classes=len(status_encoder.classes_),
    heads=2, dropout=0.35
).to(device)
model.load_state_dict(torch.load("gnn_multitask_best_fold4.pth", map_location=device))
model.eval()

# ---------- Freeze graph head ---------------
for p in model.graph_classifier.parameters():
    p.requires_grad = False

# --- RoBERTa GoEmotions --------
ROBERTA_MODEL_NAME = "SamLowe/roberta-base-go_emotions"
roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
roberta_model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL_NAME).to(device)

EMOTIONS = [
    "Admiration", "Amusement", "Anger", "Annoyance", "Approval", "Caring",
    "Confusion", "Curiosity", "Desire", "Disappointment", "Disapproval", "Disgust",
    "Embarrassment", "Excitement", "Fear", "Gratitude", "Grief", "Joy", "Love",
    "Nervousness", "Optimism", "Pride", "Realization", "Relief", "Remorse",
    "Sadness", "Surprise", "Neutral"
]

# --- Load Data -------------
df = pd.read_csv("Data/narratives.csv")
scores_df = pd.read_csv("Data/scores.csv")

# --- Sidebar ---
st.sidebar.header("Session Filter")
patients = sorted(df["Patient"].unique())
selected_patient = st.sidebar.selectbox("Patient", patients)
patient_sessions = sorted(df[df["Patient"] == selected_patient]["SessionNumber"].unique())
selected_sessions = st.sidebar.multiselect("Sessions", options=patient_sessions, default=patient_sessions)

st.sidebar.subheader("Emotion Cluster")
emotion_cluster = st.sidebar.radio("Show", options=["All", "Positive", "Negative", "Neutral"], index=0, horizontal=False)

st.sidebar.subheader("Agency Filter")
agency_filter = st.sidebar.radio("Agency", options=["All", "Active", "Passive"], index=0, horizontal=False)

# --- Sentiment utils -------------
def emotion_to_polarity(emotion):
    emotion = str(emotion).lower()
    if emotion in {"disapproval","sadness","anger","grief","fear","disgust","remorse",
                   "annoyance","disappointment","embarrassment","nervousness","confusion"}:
        return "Negative"
    elif emotion in {"amusement","joy","gratitude","love","caring","relief","admiration",
                     "approval","optimism","realization","excitement","pride"}:
        return "Positive"
    else:
        return "Neutral"

SENTIMENT_COLOR_MAP = {"Positive": POS, "Negative": NEG, "Neutral": NEU}
def get_sentiment_color(emotion):
    return SENTIMENT_COLOR_MAP.get(emotion_to_polarity(emotion), "#999")

# ------------- Filtered data ---
filtered_df = df[df["Patient"] == selected_patient].copy()
filtered_df["SentimentPolarity"] = filtered_df["Emotion"].apply(emotion_to_polarity)
if emotion_cluster != "All":
    filtered_df = filtered_df[filtered_df["SentimentPolarity"] == emotion_cluster]
if agency_filter != "All":
    filtered_df = filtered_df[filtered_df["Agency"].str.capitalize() == agency_filter]
graph_df = filtered_df[filtered_df["SessionNumber"].isin(selected_sessions)].copy()

# --- Narrative Graph  -------------
import json

G = nx.Graph()
for idx, row in graph_df.iterrows():
    tooltip = (
        f"Session: {row['SessionNumber']}\n"
        f"Date: {row['Date']}\n"
        f"Emotion: {row['Emotion']}\n"
        f"Topic: {row['Topic']}\n\n"
        f"{row['Sentence']}"
    )
    G.add_node(
        idx,
        label=row['Sentence'][:40],
        title=tooltip,
        color=get_sentiment_color(row["Emotion"]),
        shape="dot",
        size=8
    )

for session in selected_sessions:
    indices = list(graph_df[graph_df["SessionNumber"] == session].index)
    for i in range(len(indices) - 1):
        G.add_edge(indices[i], indices[i+1], color="#C9BEAC")

net = Network(
    height="620px",
    width="100%",
    bgcolor="#FFFFFF",            
    font_color=TEXT,
    cdn_resources="in_line"  
)
net.from_nx(G)
net.set_edge_smooth("dynamic")

# ----------------------------------------
options = {
    "interaction": {"hover": True, "dragNodes": True, "zoomView": True},
    "physics": {
        "enabled": True,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
            "gravitationalConstant": -40,
            "centralGravity": 0.02,
            "springLength": 120,
            "springConstant": 0.05,
            "avoidOverlap": 0.6
        },
        "minVelocity": 0.2,
        "timestep": 0.35,
        "stabilization": {"enabled": True, "iterations": 120} 
    },
    "nodes": {"shape": "dot", "borderWidth": 1}
}
net.set_options(json.dumps(options))

net.save_graph("graph.html")

# ------------------------------

with open("graph.html", "r", encoding="utf-8") as f:
    html = f.read()

html = html.replace("border: 1px solid lightgray;", "border: 0;") 
html = html.replace("background-color: white;", f"background-color: #FFFFFF;")
html = html.replace("<body>", f"<body style='margin:0;background:#FFFFFF;'>")

st.markdown(f"<h2 style='margin-top:0.5rem;color:{TEXT};'>Narrative Graph</h2>",
            unsafe_allow_html=True)

components.html(
    f"""
    <div style="
        background:#FFFFFF;
        border:1.5px solid #D8CFBF;
        border-radius:12px;
        overflow:hidden;">
        {html}
    </div>
    """,
    height=650
)


# --- Summary -------------
st.markdown(f"<h2 style='color:{TEXT};'>Patient: {selected_patient} | Sessions: {len(selected_sessions)} | Sentences: {len(graph_df)}</h2>", unsafe_allow_html=True)


def apply_plotly_theme(fig, title=None, height=None):
    fig.update_layout(
        title=title or fig.layout.title.text,

        title_font=dict(color="#000000", size=26),
        font=dict(color="#000000", size=18),
        plot_bgcolor=BG,
        paper_bgcolor=BG,

        legend=dict(bgcolor='rgba(255,255,255,0.7)', bordercolor="#E7DECD", borderwidth=1, font=dict(color="#000000", size=16)),
    )

    fig.update_xaxes(
        showgrid=True, gridcolor="#E7DECD", zeroline=False, linecolor="#000000",
        tickfont=dict(color="#000000", size=14), title_font=dict(color="#000000", size=18)
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="#E7DECD", zeroline=False, linecolor="#000000",
        tickfont=dict(color="#000000", size=14), title_font=dict(color="#000000", size=18)
    )
    if height: fig.update_layout(height=height)
    return fig

# --- Sankey --------------- ---
graph_df["SentimentPolarity"] = graph_df["Emotion"].apply(emotion_to_polarity)
graph_df["Topic"] = graph_df["Topic"].fillna("Unknown")
link_data = graph_df.groupby(["SentimentPolarity", "Topic"]).size().reset_index(name="count")
polarities = ["Negative", "Neutral", "Positive"]
topics = sorted(link_data["Topic"].unique())
all_labels = polarities + topics
label_to_index = {label: i for i, label in enumerate(all_labels)}
sources = link_data["SentimentPolarity"].map(label_to_index)
targets = link_data["Topic"].map(label_to_index)
values = link_data["count"]

# -------------------------

right_side_color = "#0a210f"
sankey_fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=14, thickness=18,
        line=dict(color=MUTED, width=0.5),
        label=all_labels,
        color=[NEG, NEU, POS] + [right_side_color]*len(topics)
    ),
    link=dict(source=sources, target=targets, value=values, color="#D8CFBF")
)])
sankey_fig = apply_plotly_theme(
    sankey_fig,
    title=f"Sentiment Polarity → Topics for Patient {selected_patient}",
    height=560
)
st.plotly_chart(sankey_fig, use_container_width=True)



# --- Sentiment Distribution per Session (Stacked Bar) ------------------
sentiment_counts = graph_df.groupby(["SessionNumber", "SentimentPolarity"]).size().unstack(fill_value=0)
for col in ["Negative", "Neutral", "Positive"]:
    if col not in sentiment_counts.columns: sentiment_counts[col] = 0
sentiment_counts = sentiment_counts[["Negative", "Neutral", "Positive"]].sort_index()

fig = go.Figure(data=[
    go.Bar(name='Negative', x=sentiment_counts.index, y=sentiment_counts["Negative"], marker_color=NEG),
    go.Bar(name='Neutral',  x=sentiment_counts.index, y=sentiment_counts["Neutral"],  marker_color=NEU),
    go.Bar(name='Positive', x=sentiment_counts.index, y=sentiment_counts["Positive"], marker_color=POS),
])
fig.update_layout(barmode='stack', xaxis_title='Session Number', yaxis_title='Sentence Count')
fig = apply_plotly_theme(fig, title="Sentiment Distribution per Session", height=500)

# ---------------------------------------------
fig.update_layout(legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top"))

st.plotly_chart(fig, use_container_width=True)


# -------- Deterioration Analysis -----------------------------
session_stats = filtered_df.groupby(["Patient", "SessionNumber"]).agg({
    "SentimentPolarity": lambda x: (x == "Negative").mean()
}).rename(columns={"SentimentPolarity": "NegativityRatio"}).reset_index()

merged_stats = pd.merge(session_stats, scores_df, on=["Patient", "SessionNumber"], how="left")
merged_stats["Deteriorating"] = merged_stats.apply(
    lambda row: row["NegativityRatio"] > 0.6 and (row.get("GAD-7_Score", 0) > 15 or row.get("PHQ-9_Score", 0) > 15),
    axis=1
)

st.markdown(f"**Deterioration Flags:** {merged_stats['Deteriorating'].sum()}")

flagged_sessions = merged_stats[merged_stats["Deteriorating"]]
if not flagged_sessions.empty:
    st.markdown("**Flagged Sessions:**")
    st.dataframe(flagged_sessions[["SessionNumber", "NegativityRatio", "GAD-7_Score", "PHQ-9_Score"]])

# --- Mental Status Trend (Line) ------------------
merged_stats["MentalStatus"] = merged_stats.apply(
    lambda row: "Deteriorating" if row["NegativityRatio"] > 0.6 and (row.get("GAD-7_Score", 0) > 15 or row.get("PHQ-9_Score", 0) > 15)
    else ("Improving" if row["NegativityRatio"] < 0.2 and row.get("GAD-7_Score", 10) <= 10 and row.get("PHQ-9_Score", 10) <= 10
    else "Stable"),
    axis=1
)

def _session_key(s):
    s = str(s); m = re.search(r"(\d+)$", s)
    return int(m.group(1)) if m else float("inf")

order = sorted(merged_stats["SessionNumber"].astype(str).unique(), key=_session_key)
merged_stats["SessionNumber"] = pd.Categorical(merged_stats["SessionNumber"], categories=order, ordered=True)

fig_trend = px.line(
    merged_stats.sort_values("SessionNumber"),
    x="SessionNumber",
    y="NegativityRatio",
    color="MentalStatus",
    title="Session-Level Negativity Over Time",
    markers=True,
    color_discrete_map={"Improving": POS, "Stable": NEU, "Deteriorating": NEG}
)
fig_trend.update_xaxes(categoryorder="array", categoryarray=order)
fig_trend = apply_plotly_theme(fig_trend, height=420)


fig_trend.update_layout(legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top"))

st.plotly_chart(fig_trend, use_container_width=True)


# --- Live Prediction ------------------
st.markdown("---")
st.markdown(f"<h2 style='color:{TEXT};'>Live Emotion & Mental Status Prediction</h2>", unsafe_allow_html=True)

col_pred_input, col_pred_output = st.columns([1.1, 1])
with col_pred_input:
    user_input = st.text_area("Enter sentence(s):", placeholder="One sentence per line…", height=160)
    predict_clicked = st.button("Predict")

with col_pred_output:
    if 'predict_clicked' in locals() and predict_clicked and user_input.strip():
        sentences_raw = [s.strip() for s in user_input.strip().split("\n") if s.strip()]
        sentences = [f"passage: {s}" for s in sentences_raw]

        if len(sentences) == 1:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        else:
            edge_index = torch.tensor(
                [[i, i+1] for i in range(len(sentences)-1)] + [[i+1, i] for i in range(len(sentences)-1)],
                dtype=torch.long
            ).T

        with torch.no_grad():
            emb = embedder.encode(sentences)
        x = torch.tensor(emb, dtype=torch.float)
        n = x.size(0)
        pos = torch.zeros((n,1), dtype=torch.float) if n == 1 else (torch.arange(n, dtype=torch.float)/max(1,(n-1))).view(-1,1)
        x = torch.cat([x, pos], dim=1)
        batch = torch.zeros(x.size(0), dtype=torch.long)

        x, edge_index, batch = x.to(device), edge_index.to(device), batch.to(device)
        with torch.no_grad():
            node_out, graph_out = model(x, edge_index, batch)
            node_probs = F.softmax(node_out, dim=1).cpu().numpy()
            graph_probs = F.softmax(graph_out, dim=1).cpu().numpy()

            node_preds = emotion_encoder.inverse_transform(node_probs.argmax(axis=1))
            graph_pred = status_encoder.inverse_transform([graph_probs.argmax()])[0]
            graph_conf = graph_probs.max()

        st.subheader("Sentence-Level Emotions")
        st.dataframe(pd.DataFrame({
            "Sentence": sentences_raw,
            "Predicted Emotion": node_preds,
            "Confidence (%)": (node_probs.max(axis=1) * 100).round(2)
        }), use_container_width=True)

        st.subheader("Session-Level Mental Status")
        st.markdown(f"**Predicted Status:** {graph_pred} ({graph_conf * 100:.2f}% confidence)")

# --- AI-Generated Narrative ----------------------------
st.markdown("---")
st.markdown(f"<h2 style='color:{TEXT};'>AI-Generated Patient Narrative</h2>", unsafe_allow_html=True)
st.markdown("Generate a natural therapy session narrative from the patient's perspective.")

col1, col2, col3, col4, col5, col6 = st.columns([1.2, 1.2, 1.2, 1.5, 1.5, 1.2])
with col1:
    patient_id = st.text_input("Patient ID (e.g., P0001)", key="narrative_pid")
with col2:
    session_id = st.text_input("Session ID (e.g., S0001)", key="narrative_sid")
with col3:
    session_date = st.date_input("Session Date", key="narrative_date")
with col4:
    model_option = st.selectbox("Choose Model",
                                ["mixtral", "llama3", "qwen2", "llama4", "phi4", "gemma3", "qwen3", "dbrx"],
                                key="narrative_model")
with col5:
    predefined_topics = ["Anxiety","Relationships","Self-Care","Burnout","Stress-Management",
                         "Coping-Skills","Emotional-Regulation","Family-Dynamics","Life-Transitions"]
    topic_selection = st.selectbox("Select Narrative Topic", predefined_topics + ["Other"],
                                   key="narrative_topic_selectbox")
with col6:
    num_sentences = st.number_input("Number of Sentences", min_value=1, max_value=200, value=5,
                                    key="narrative_sentence_count")

if topic_selection == "Other":
    topic = st.text_input("Enter a new topic:", key="narrative_custom_topic").strip() or "Unknown"
else:
    topic = topic_selection
session_date_str = session_date.strftime("%b %d, %Y") if session_date else "Unknown"

def validate_inputs(patient_id, session_id, topic):
    pid_ok = bool(re.fullmatch(r"^P\d{4}$", patient_id or ""))
    sid_ok = bool(re.fullmatch(r"^S\d{4}$", session_id or ""))
    topic_ok = bool(re.fullmatch(r"^[A-Za-z]+(?:-[A-Za-z]+)?$", topic or ""))

    if not pid_ok:
        st.error("Patient ID must start with 'P' followed by 4 digits (e.g., P0001).")
        return False
    if not sid_ok:
        st.error("Session ID must start with 'S' followed by 4 digits (e.g., S0001).")
        return False
    if not topic_ok:
        st.error("Topic must be one word or two words separated by a dash (e.g., Anxiety or Life-Transitions).")
        return False
    return True


def generate_patient_narrative(model_name, topic, num_sentences):
    prompt = f"""
You are simulating a patient in a therapy session. 
Generate {num_sentences} unique sentences about the topic: {topic}, forming a natural narrative 
from the patient's perspective where each sentence follows logically and tells a coherent story.

Guidelines:
- Each sentence should clearly express an emotion that could map to the GoEmotions [
    "Admiration", "Amusement", "Anger", "Annoyance", "Approval", "Caring",
    "Confusion", "Curiosity", "Desire", "Disappointment", "Disapproval", "Disgust",
    "Embarrassment", "Excitement", "Fear", "Gratitude", "Grief", "Joy", "Love",
    "Nervousness", "Optimism", "Pride", "Realization", "Relief", "Remorse",
    "Sadness", "Surprise", "Neutral"] dataset.
- Do NOT label the emotions or mention them in the output.
- Include subtle emotional and situational variations.
- Avoid repeating sentences or producing list-like outputs.
- Maintain a first-person patient perspective suitable for therapy notes.
- Increase the negativity in the story like a patient is so depressed.

Output Format:
Return in this exact format for each sentence (one block per sentence):

Sentence: <patient's sentence>
Agency: <agency> (Active or Passive)
"""
    response = get_model_response(model_name, prompt)
    sentences = re.findall(r"Sentence:\s*(.*)", response, re.IGNORECASE)
    agencies  = re.findall(r"Agency:\s*(.*)", response, re.IGNORECASE)
    max_len = max(len(sentences), len(agencies))
    sentences += [""] * (max_len - len(sentences))
    agencies  += ["Passive"] * (max_len - len(agencies))
    emotions = [""] * max_len
    return list(zip(sentences, emotions, agencies))

if st.button("Generate Narrative"):
    if validate_inputs(patient_id, session_id, topic):
        conversation = []
        narrative = generate_patient_narrative(model_option, topic, num_sentences)
        for sentence, emotion, agency in narrative:
            conversation.append([patient_id, session_id, session_date_str,
                                 sentence.strip(), emotion.strip(), agency.strip(), topic])
        conv_df = pd.DataFrame(conversation,
                               columns=["Patient","SessionNumber","Date","Sentence","Emotion","Agency","Topic"])
        st.subheader("Generated Narrative")
        st.dataframe(conv_df, use_container_width=True)
        csv = conv_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Narrative CSV", csv,
                           file_name=f"patient_narrative_{patient_id}_{session_id}.csv", mime='text/csv')

# --- CSV Upload for RoBERTa Annotation -----------------------
st.markdown("---")
st.markdown(f"<h2 style='color:{TEXT};'>Therapy Session Emotion Annotator (RoBERTa)</h2>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload your therapy session CSV", type=["csv"])

if uploaded_file:
    df_upload = pd.read_csv(uploaded_file)
    required_cols = ["Patient","SessionNumber","Date","Sentence","Emotion","Agency","Topic"]
    if not all(col in df_upload.columns for col in required_cols):
        st.error(f"CSV must contain these columns: {', '.join(required_cols)}")
    else:
        st.success(f"CSV loaded with {len(df_upload)} rows.")
        if st.button("Annotate Emotions with RoBERTa"):
            st.info("Starting annotation…")
            progress_bar = st.progress(0)
            annotated_emotions = []
            sentences = df_upload["Sentence"].astype(str).fillna("").tolist()
            for i, sentence in enumerate(sentences):
                inputs = roberta_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(device)
                with torch.no_grad():
                    outputs = roberta_model(**inputs)
                    pred_id = torch.argmax(outputs.logits, dim=1).item()
                emotion = EMOTIONS[pred_id % len(EMOTIONS)]
                annotated_emotions.append(emotion)
                progress_bar.progress((i + 1) / len(sentences))
            df_upload["Emotion"] = annotated_emotions
            st.subheader("Sample Annotated Rows")
            st.dataframe(df_upload.head(20))
            csv_out = df_upload.to_csv(index=False).encode('utf-8')
            st.download_button("Download Annotated CSV", csv_out, file_name="annotated_sessions.csv", mime='text/csv')
            st.success("Annotation complete!")

