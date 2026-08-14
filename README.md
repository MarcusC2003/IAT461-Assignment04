# DeckForge — New-Player Onboarding (IAT 461 Final Assignment)

## Problem

DeckForge, a Pokémon TCG companion app, was losing new players in their first week: they'd open the app, see thousands of cards with no structure, and leave. The client proposed three possible onboarding features and asked for a data-driven recommendation on which one to build:

- **Option A** — an archetype browser that groups cards by playstyle.
- **Option B** — a price/investment tracker that flags cards likely to rise in value.
- **Option C** — a "power level" warning for over/under-costed cards.

## What Was Done

1. **Data audit** — cleaned and validated `pokemon_cards.csv` (816 cards), confirming missing values were structural (e.g. Trainer/Energy cards have no HP or attacks) rather than data errors.
2. **Feasibility check** — tested each option against what the data could actually support: Option B needs a price signal, Option C needs a ground-truth power-level label — neither exists in the dataset, so both were eliminated.
3. **Feature engineering** — for Option A, parsed each card's `attacks_json`/`abilities_json` into numeric features (energy cost, damage, ability flags) and mined 10 binary keyword features from attack/ability text (draw, search, heal, status, coin flip, etc.).
4. **Clustering** — ran K-Means (k=2–8, selected k=6 via silhouette score) on the 624 Pokémon cards to discover natural archetypes; grouped Trainer/Energy cards separately by subtype since they have no attack data.
5. **Delivery** — built a Streamlit app to let stakeholders explore the resulting archetypes interactively, plus a presentation and report summarizing the process.

## What Was Found

- **Option A (Archetype Browser) was the only feasible option** — Options B and C were ruled out before any modelling because the required data (prices, power labels) doesn't exist in the dataset.
- Clustering produced **6 named archetypes** for Pokémon cards:

  | Archetype          | Size | Key Traits                         |
  | ------------------ | ---- | ---------------------------------- |
  | Basic Attackers    | ~280 | Low HP, no ability, Basic stage    |
  | Heavy Hitters      | ~59  | High HP + damage, Stage 2          |
  | Ability Disruptors | ~88  | 100% ability rate, status-heavy    |
  | Evolved Mid-Range  | ~156 | Stage 1/2, no ability, solid stats |
  | Energy Engines     | 7    | Attach extra energy, rare          |
  | Agile Strikers     | ~34  | Low retreat, efficient attackers   |

- Trainer and Energy cards were grouped separately by subtype (e.g. Item, Stadium, Basic/Special Energy) since they have no attack/ability stats to cluster on.
- Silhouette scores were low overall (~0.184), so clusters should be treated as **fuzzy, human-interpretable categories** rather than hard boundaries — sufficient for a browsing UI, not for precise classification.

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Explore the analysis
jupyter notebook eda.ipynb

# 3. Launch the app
streamlit run app.py
```

## Files

- **Client_Proposal.md** — The client brief: business context, the three onboarding options considered, and the task definition.
- **eda.ipynb** — Exploratory data analysis and machine learning report notebook (clustering, feature engineering, findings, summary).
- **FitnessPlanet_Client_Evaluation** — Client Evaluation for FitnessPlanet.
- **Milestone2_Progress_Report.pdf** — Milestone 2 progress report writeup.
- **DeckForge_Presentation.pdf** — Slide deck for the final presentation.
- **presentation_video.mp4** — Recorded video of the presentation.
- **app.py** — Streamlit app (DeckForge Archetype Explorer) that visualizes and explores the card clusters.
- **pokemon_cards.csv** — Raw Pokémon TCG card dataset used for the analysis.
- **pokemon_cards_features.csv** — Engineered/derived features built from the raw card dataset for clustering.
- **requirements.txt** — Python dependencies needed to run `app.py` and `eda.ipynb`.
