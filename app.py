"""
DeckForge — Pokémon TCG Archetype Explorer
============================================
Run with:  streamlit run app.py or python -m streamlit run app.py
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Palette (validated categorical / sequential / diverging set)
# ----------------------------------------------------------------------------
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948", "#008300"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
DIVERGING = "RdBu_r"  # blue <-> red, matches the diverging pair (reversed so blue=low, red=high)
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

CSV_PATH = Path(__file__).parent / "pokemon_cards.csv"
FEATURES_CSV_PATH = Path(__file__).parent / "pokemon_cards_features.csv"

CLUSTER_LABELS_K6 = {
    0: "Basic Attackers",
    1: "Bulky Heavy Hitters",
    2: "Ability / Combo",
    3: "Evolved Mid-Range",
    4: "Energy Engines",
    5: "Agile Strikers",
}

TXT_FEATURES = [
    "txt_draw", "txt_search", "txt_heal", "txt_status", "txt_accel",
    "txt_discard", "txt_coin", "txt_counter", "txt_switch", "txt_prevent",
]
NUMERIC_FEATURES = [
    "hp", "retreat_count", "is_basic", "is_stage1", "is_stage2",
    "num_attacks", "max_damage", "min_cost", "max_cost", "avg_dpe",
    "has_ability", "num_abilities",
] + TXT_FEATURES

st.set_page_config(page_title="DeckForge — Archetype Explorer", layout="wide")


# ----------------------------------------------------------------------------
# Feature engineering — mirrors eda.ipynb exactly
# ----------------------------------------------------------------------------
def parse_damage(s):
    m = re.match(r"^(\d+)", str(s).strip())
    return int(m.group(1)) if m else 0


def parse_attacks(s):
    try:
        attacks = json.loads(s) if isinstance(s, str) else []
    except Exception:
        attacks = []
    if not attacks:
        return dict(num_attacks=0, max_damage=0, min_cost=0, max_cost=0, avg_dpe=0.0, attack_text="")
    damages = [parse_damage(a.get("damage", "0")) for a in attacks]
    costs = [len(a.get("cost", [])) for a in attacks]
    text = " ".join(a.get("text", "") or "" for a in attacks)
    dpe = [d / c for d, c in zip(damages, costs) if c > 0]
    return dict(
        num_attacks=len(attacks), max_damage=max(damages), min_cost=min(costs),
        max_cost=max(costs), avg_dpe=float(np.mean(dpe)) if dpe else 0.0, attack_text=text,
    )


def parse_abilities(s):
    try:
        abs_ = json.loads(s) if isinstance(s, str) else []
    except Exception:
        abs_ = []
    if not abs_:
        return dict(has_ability=0, num_abilities=0, ability_text="")
    text = " ".join(((a.get("text") or "") + " " + (a.get("name") or "")).strip() for a in abs_)
    return dict(has_ability=1, num_abilities=len(abs_), ability_text=text)


def retreat_count(s):
    if not isinstance(s, str) or not s.strip():
        return 0
    return len(s.split("|"))


def mine_text(t):
    t = (t or "").lower()
    return dict(
        txt_draw=int(bool(re.search(r"draw", t))),
        txt_search=int(bool(re.search(r"search your deck|look at the top", t))),
        txt_heal=int(bool(re.search(r"remove.{0,10}damage counter|heal", t))),
        txt_status=int(bool(re.search(r"paralyz|confus|poison|burn|asleep|sleep", t))),
        txt_accel=int(bool(re.search(r"attach.{0,10}energy", t))),
        txt_discard=int(bool(re.search(r"discard", t))),
        txt_coin=int(bool(re.search(r"flip a coin", t))),
        txt_counter=int(bool(re.search(r"damage counter", t))),
        txt_switch=int(bool(re.search(r"switch", t))),
        txt_prevent=int(bool(re.search(r"prevent|protect|reduce", t))),
    )


@st.cache_data(show_spinner="Loading & engineering features from pokemon_cards.csv…")
def load_and_engineer(csv_path: str) -> pd.DataFrame:
    """Reproduce the eda.ipynb feature-engineering pipeline and return feat_df."""
    df = pd.read_csv(csv_path)

    atk_feats = df["attacks_json"].apply(parse_attacks).apply(pd.Series)
    abl_feats = df["abilities_json"].apply(parse_abilities).apply(pd.Series)
    combined_text = atk_feats["attack_text"] + " " + abl_feats["ability_text"]
    txt_feats = combined_text.apply(mine_text).apply(pd.Series)

    df["retreat_count"] = df["retreat_cost"].apply(retreat_count)
    df["is_basic"] = df["subtypes"].str.contains("Basic", na=False).astype(int)
    df["is_stage1"] = df["subtypes"].str.contains("Stage 1", na=False).astype(int)
    df["is_stage2"] = df["subtypes"].str.contains("Stage 2", na=False).astype(int)

    feat_df = pd.concat(
        [
            df[["card_id", "name", "set_code", "supertype", "subtypes", "types", "rarity",
                "hp", "retreat_count", "is_basic", "is_stage1", "is_stage2", "flavor_text"]],
            atk_feats.drop(columns=["attack_text"]),
            abl_feats.drop(columns=["ability_text"]),
            txt_feats,
        ],
        axis=1,
    )
    feat_df["attack_text"] = combined_text
    return feat_df


@st.cache_data(show_spinner=False)
def write_features_csv(feat_df: pd.DataFrame, path: str) -> str:
    """Persist the engineered dataframe (the df built during the EDA) to disk."""
    feat_df.to_csv(path, index=False)
    return path


@st.cache_data(show_spinner="Clustering Pokémon cards…")
def cluster_pokemon(feat_df: pd.DataFrame, k: int, seed: int = 42):
    poke = feat_df[feat_df["supertype"].str.contains("Pok", na=False)].copy()
    poke["hp"] = poke["hp"].fillna(0)

    X = poke[NUMERIC_FEATURES].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    poke["cluster"] = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, poke["cluster"])

    pca = PCA(n_components=2, random_state=seed)
    X_pca = pca.fit_transform(X_scaled)
    poke["pc1"], poke["pc2"] = X_pca[:, 0], X_pca[:, 1]

    centroids_scaled = km.cluster_centers_
    centroids = pd.DataFrame(
        scaler.inverse_transform(centroids_scaled), columns=NUMERIC_FEATURES
    )
    return poke, centroids, sil, pca.explained_variance_ratio_


@st.cache_data(show_spinner="Sweeping k=2..8 for silhouette scores…")
def silhouette_sweep(feat_df: pd.DataFrame, seed: int = 42):
    poke = feat_df[feat_df["supertype"].str.contains("Pok", na=False)].copy()
    poke["hp"] = poke["hp"].fillna(0)
    X = poke[NUMERIC_FEATURES].values.astype(float)
    X_scaled = StandardScaler().fit_transform(X)

    scores = {}
    for k in range(2, 9):
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)
    return scores


def cluster_name(k: int, c: int) -> str:
    if k == 6:
        return f"C{c}: {CLUSTER_LABELS_K6[c]}"
    return f"Cluster {c}"


def cluster_color_map(k: int):
    return {i: CATEGORICAL[i % len(CATEGORICAL)] for i in range(k)}


PLOTLY_LAYOUT = dict(
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    font_color=INK, font_family="system-ui, -apple-system, Segoe UI, sans-serif",
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, title_font_color=MUTED, tickfont_color=MUTED),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, title_font_color=MUTED, tickfont_color=MUTED),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(t=50, l=10, r=10, b=10),
)


def style(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# ----------------------------------------------------------------------------
# Load data once
# ----------------------------------------------------------------------------
if not CSV_PATH.exists():
    st.error(f"Could not find `pokemon_cards.csv` next to app.py (looked in {CSV_PATH}).")
    st.stop()

feat_df = load_and_engineer(str(CSV_PATH))
saved_path = write_features_csv(feat_df, str(FEATURES_CSV_PATH))

# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
st.sidebar.title("DeckForge")
st.sidebar.caption("New-player onboarding · Pokémon TCG companion")
page = st.sidebar.radio(
    "Section",
    [
        "Overview",
        "Archetype Explorer",
        "Card Deep-Dive",
        "Feature Correlations",
    ],
)
st.sidebar.divider()
st.sidebar.caption(
    f"Engineered feature table saved to:\n`{FEATURES_CSV_PATH.name}`\n"
    f"({feat_df.shape[0]} rows × {feat_df.shape[1]} cols)"
)

poke_all = feat_df[feat_df["supertype"].str.contains("Pok", na=False)]
trainers = feat_df[feat_df["supertype"] == "Trainer"]
energies = feat_df[feat_df["supertype"] == "Energy"]

# ==============================================================================
# 1. OVERVIEW
# ==============================================================================
if page == "Overview":
    st.title("Dataset Overview")
    st.caption("816 cards spanning Base Set → Neo Genesis (`base1`–`base5`, `gym1`–`gym2`, `neo1`).")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total cards", f"{len(feat_df):,}")
    c2.metric("Pokémon", f"{len(poke_all):,}")
    c3.metric("Trainer", f"{len(trainers):,}")
    c4.metric("Energy", f"{len(energies):,}")

    col1, col2 = st.columns(2)

    with col1:
        counts = feat_df["supertype"].value_counts().reset_index()
        counts.columns = ["supertype", "count"]
        fig = px.pie(
            counts, names="supertype", values="count", hole=0.5,
            color="supertype", color_discrete_sequence=CATEGORICAL,
        )
        fig.update_traces(textinfo="label+percent", marker=dict(line=dict(color=SURFACE, width=2)))
        fig.update_layout(title="Card Supertype Split", showlegend=False)
        st.plotly_chart(style(fig), width="stretch")

    with col2:
        hp = poke_all["hp"].dropna()
        fig = px.histogram(hp, x="hp", nbins=18, color_discrete_sequence=[CATEGORICAL[0]])
        fig.update_traces(marker_line_color=SURFACE, marker_line_width=1)
        fig.update_layout(title="HP Distribution (Pokémon only)", xaxis_title="HP", yaxis_title="Cards", bargap=0.05)
        st.plotly_chart(style(fig), width="stretch")

    col3, col4 = st.columns(2)
    with col3:
        rarity_order = ["Common", "Uncommon", "Rare", "Rare Holo", "Rare Secret"]
        rc = feat_df["rarity"].value_counts().reindex(rarity_order).dropna().reset_index()
        rc.columns = ["rarity", "count"]
        fig = px.bar(rc, x="rarity", y="count", color_discrete_sequence=[CATEGORICAL[1]])
        fig.update_traces(marker_line_color=SURFACE, marker_line_width=1)
        fig.update_layout(title="Rarity Distribution", xaxis_title="", yaxis_title="Cards")
        st.plotly_chart(style(fig), width="stretch")

    with col4:
        types = poke_all["types"].value_counts().head(11).reset_index()
        types.columns = ["type", "count"]
        fig = px.bar(
            types.sort_values("count"), x="count", y="type", orientation="h",
            color_discrete_sequence=[CATEGORICAL[2]],
        )
        fig.update_traces(marker_line_color=SURFACE, marker_line_width=1)
        fig.update_layout(title="Pokémon Count by Type", xaxis_title="Cards", yaxis_title="")
        st.plotly_chart(style(fig), width="stretch")

# ==============================================================================
# 2. ARCHETYPE EXPLORER (clustering)
# ==============================================================================
elif page == "Archetype Explorer":
    st.title("Pokémon Archetype Explorer")
    st.caption("K-Means clustering over engineered attack/ability/text features — pick k and explore.")

    with st.expander("Silhouette score vs. k (why k=6 was chosen)"):
        scores = silhouette_sweep(feat_df)
        sdf = pd.DataFrame({"k": list(scores.keys()), "silhouette": list(scores.values())})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sdf["k"], y=sdf["silhouette"], mode="lines+markers",
                                  line=dict(color=CATEGORICAL[0], width=2), marker=dict(size=9)))
        fig.add_vline(x=6, line_dash="dash", line_color=CATEGORICAL[1],
                      annotation_text="k=6 (EDA's choice)", annotation_font_color=MUTED)
        fig.update_layout(title="Silhouette Score vs. Number of Clusters", xaxis_title="k", yaxis_title="Silhouette score")
        st.plotly_chart(style(fig), width="stretch")
        st.caption(
            "Scores are low across the board (~0.15–0.20) — TCG card mechanics form a "
            "continuum, not tight clusters. Treat archetypes as fuzzy, human-nameable "
            "groups rather than hard boundaries."
        )

    k = st.slider("Number of archetypes (k)", min_value=2, max_value=8, value=6)
    poke, centroids, sil, var_ratio = cluster_pokemon(feat_df, k)
    cmap = cluster_color_map(k)
    poke["cluster_name"] = poke["cluster"].map(lambda c: cluster_name(k, c))

    m1, m2, m3 = st.columns(3)
    m1.metric("Clusters (k)", k)
    m2.metric("Silhouette score", f"{sil:.3f}")
    m3.metric("PCA variance explained (2D)", f"{sum(var_ratio) * 100:.1f}%")

    col1, col2 = st.columns([1.3, 1])
    with col1:
        fig = px.scatter(
            poke, x="pc1", y="pc2", color="cluster_name",
            color_discrete_sequence=[cmap[i] for i in range(k)],
            hover_data={"name": True, "hp": True, "max_damage": True, "pc1": False, "pc2": False},
            opacity=0.7,
        )
        fig.update_traces(marker=dict(size=7, line=dict(width=0)))
        fig.update_layout(
            title=f"Archetypes — PCA Projection (k={k})",
            xaxis_title=f"PC1 ({var_ratio[0]*100:.1f}% var)",
            yaxis_title=f"PC2 ({var_ratio[1]*100:.1f}% var)",
            legend_title="Archetype",
        )
        st.plotly_chart(style(fig), width="stretch")

    with col2:
        size_df = poke["cluster_name"].value_counts().reset_index()
        size_df.columns = ["cluster", "n"]
        fig = px.bar(
            size_df.sort_values("n"), x="n", y="cluster", orientation="h",
            color="cluster", color_discrete_map={cluster_name(k, i): cmap[i] for i in range(k)},
        )
        fig.update_traces(marker_line_color=SURFACE, marker_line_width=1, showlegend=False)
        fig.update_layout(title="Archetype Sizes", xaxis_title="Cards", yaxis_title="")
        st.plotly_chart(style(fig), width="stretch")

    st.subheader("Cluster centroid profiles")
    display_cols = ["hp", "retreat_count", "is_basic", "is_stage1", "is_stage2",
                     "max_damage", "avg_dpe", "has_ability", "txt_status", "txt_accel",
                     "txt_coin", "txt_counter", "txt_heal", "txt_draw", "txt_prevent"]
    cent = centroids[display_cols].copy()
    cent.index = [cluster_name(k, i) for i in range(k)]
    fig = px.imshow(
        cent.T, color_continuous_scale=SEQ_BLUE, aspect="auto",
        labels=dict(color="Mean value"),
    )
    fig.update_layout(title="Cluster Centroid Heatmap", xaxis_title="", yaxis_title="")
    fig.update_xaxes(tickangle=25)
    st.plotly_chart(style(fig), width="stretch")

    st.subheader("Browse cards within an archetype")
    chosen = st.selectbox("Archetype", [cluster_name(k, i) for i in range(k)])
    chosen_idx = [i for i in range(k) if cluster_name(k, i) == chosen][0]
    sample = poke[poke["cluster"] == chosen_idx][
        ["name", "hp", "max_damage", "avg_dpe", "has_ability", "retreat_count", "set_code", "rarity"]
    ].reset_index(drop=True)
    st.dataframe(sample, width="stretch", height=320)

# ==============================================================================
# 3. CARD DEEP-DIVE
# ==============================================================================
elif page == "Card Deep-Dive":
    st.title("Card Deep-Dive")
    st.caption("Inspect one Pokémon card against the average profile of its archetype.")

    poke, centroids, sil, var_ratio = cluster_pokemon(feat_df, 6)
    poke["cluster_name"] = poke["cluster"].map(lambda c: cluster_name(6, c))

    names = poke.sort_values("name")["name"] + " (" + poke.sort_values("name")["card_id"] + ")"
    choice = st.selectbox("Choose a card", names)
    card_id = choice.split("(")[-1].rstrip(")")
    card = poke[poke["card_id"] == card_id].iloc[0]

    col1, col2 = st.columns([1, 1.6])
    with col1:
        st.subheader(card["name"])
        st.markdown(f"**Set:** {card['set_code']} · **Rarity:** {card['rarity']}")
        st.markdown(f"**Type:** {card['types']} · **Subtype:** {card['subtypes']}")
        st.markdown(f"**Archetype:** `{card['cluster_name']}`")
        st.metric("HP", f"{card['hp']:.0f}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Max damage", f"{card['max_damage']:.0f}")
        m2.metric("Dmg / energy", f"{card['avg_dpe']:.1f}")
        m3.metric("Retreat", f"{card['retreat_count']:.0f}")
        if card["flavor_text"] and isinstance(card["flavor_text"], str):
            st.caption(f"_{card['flavor_text']}_")
        if card["attack_text"].strip():
            with st.expander("Attack / ability text"):
                st.write(card["attack_text"])

    with col2:
        radar_cols = ["hp", "max_damage", "avg_dpe", "retreat_count", "num_attacks", "has_ability"]
        pretty = ["HP", "Max Dmg", "Dmg/Energy", "Retreat", "# Attacks", "Has Ability"]
        cluster_avg = centroids.loc[card["cluster"], radar_cols]
        card_vals = card[radar_cols].astype(float)

        # normalize each axis 0-1 against the full Pokémon population for comparability
        pop_max = poke[radar_cols].max().replace(0, 1)
        card_norm = (card_vals / pop_max).tolist()
        avg_norm = (cluster_avg / pop_max).tolist()

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=card_norm + [card_norm[0]], theta=pretty + [pretty[0]],
            fill="toself", name=card["name"], line_color=CATEGORICAL[0],
        ))
        fig.add_trace(go.Scatterpolar(
            r=avg_norm + [avg_norm[0]], theta=pretty + [pretty[0]],
            fill="toself", name=f"{card['cluster_name']} avg", line_color=CATEGORICAL[1], opacity=0.6,
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID), bgcolor=SURFACE),
            title=f"{card['name']} vs. Archetype Average (normalized)",
            showlegend=True,
        )
        st.plotly_chart(style(fig), width="stretch")

    st.subheader(f"Other cards in {card['cluster_name']}")
    peers = poke[(poke["cluster"] == card["cluster"]) & (poke["card_id"] != card_id)][
        ["name", "hp", "max_damage", "avg_dpe", "has_ability", "set_code"]
    ].sample(min(8, max(0, len(poke[poke["cluster"] == card["cluster"]]) - 1)), random_state=1)
    st.dataframe(peers, width="stretch")

# ==============================================================================
# 4. FEATURE CORRELATIONS
# ==============================================================================
elif page == "Feature Correlations":
    st.title("Feature Correlations")
    st.caption("How the engineered features relate to each other, and how any two relate for a chosen pair.")

    poke, centroids, sil, var_ratio = cluster_pokemon(feat_df, 6)
    poke["cluster_name"] = poke["cluster"].map(lambda c: cluster_name(6, c))

    corr = poke[NUMERIC_FEATURES].corr()
    fig = px.imshow(
        corr, color_continuous_scale=DIVERGING, zmin=-1, zmax=1, aspect="auto",
        labels=dict(color="Correlation"),
    )
    fig.update_layout(title="Feature Correlation Matrix (Pokémon only)")
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(style(fig), width="stretch")

    st.subheader("Scatter explorer")
    c1, c2 = st.columns(2)
    with c1:
        x_feat = st.selectbox("X axis", NUMERIC_FEATURES, index=NUMERIC_FEATURES.index("max_damage"))
    with c2:
        y_feat = st.selectbox("Y axis", NUMERIC_FEATURES, index=NUMERIC_FEATURES.index("hp"))

    cmap = cluster_color_map(6)
    fig = px.scatter(
        poke, x=x_feat, y=y_feat, color="cluster_name",
        color_discrete_sequence=[cmap[i] for i in range(6)],
        hover_data=["name"], opacity=0.7,
    )
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(title=f"{y_feat} vs. {x_feat}, colored by archetype", legend_title="Archetype")
    st.plotly_chart(style(fig), width="stretch")
