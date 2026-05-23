# Drop Shot 🎾

A Roland-Garros 2026 tournament predictor powered by a clay-court Elo model and Monte Carlo simulation. Every simulation produces a unique bracket — just like the real thing.

**Live at:** [drop-shot on Railway] *(https://web-production-b3f9e.up.railway.app)*

---

## What it does

| Feature | Description |
|---------|-------------|
| **Bracket** | Simulates the full 128-player draw. R128 and R64 run silently; you see Round of 32 onwards. Each click generates a unique result. |
| **Odds** | Runs 3,000 Monte Carlo simulations and shows title probabilities for every player in the draw. |
| **Head to Head** | Pick any two players from the draw, get a win probability and a one-line clay Elo insight. |
| **Compare** | Runs two independent simulations side by side — shows how much variance a single tournament contains. |
| **Player card** | Click any player in the bracket for their clay Elo rating, global Elo, and recent win rate. |

---

## How the model works

The prediction engine lives in the parent repo: [grand-slam-predictions](https://github.com/maanitmehta/grand-slam-predictions).

In brief:

- **25 years of data** — ATP and WTA match results 2000–2025 (tennis-data.co.uk)
- **Logistic Regression** trained on surface-filtered matches (clay only for Roland-Garros)
- **Features:** surface-specific Elo difference, global Elo difference, rolling win rate, rolling average odds, matches played
- **Elo ratings** built from scratch with K=32 (K=48 for Grand Slams), separate tracks for clay / hard / grass / global
- **Unknown players** (qualifiers, wildcards with no history) fall back to Elo-only prediction; players absent from Elo entirely receive the database median (~1500), giving them ~5% vs a top-10 player

Model accuracy on clay: **64% ATP / 67% WTA** (walk-forward backtest).

---

## Running locally

```bash
git clone https://github.com/maanitmehta/drop-shot.git
cd drop-shot
pip install -r requirements.txt
python app.py
```

Open [http://localhost:5050](http://localhost:5050).

The landing page pre-computes odds in a background thread on startup — takes ~10 seconds before the Home tab populates.

---

## Project structure

```
drop-shot/
├── app.py                        # Flask app — all routes and simulation logic
├── templates/
│   └── index.html                # Single-page frontend (vanilla JS, no framework)
├── scripts/
│   ├── predict_match.py          # Match win probability (model + Elo fallback)
│   └── name_utils.py             # Canonical name formatting
├── config/
│   ├── tournaments.py            # Tournament configs (surface, draw paths, model paths)
│   └── surfaces.py               # Surface training filters
├── models/
│   └── fo26/
│       ├── atp_model.pkl         # Trained logistic regression — ATP clay
│       └── wta_model.pkl         # Trained logistic regression — WTA clay
├── data/
│   ├── draws/fo26/               # Official RG 2026 first-round draw CSVs
│   └── processed/
│       ├── atp/                  # Rolling stats + Elo snapshot
│       └── wta/                  # Rolling stats + Elo snapshot
├── requirements.txt
└── Procfile                      # gunicorn config for Railway/Render
```

---

## Deploying

The app is configured for [Railway](https://railway.app) out of the box.

1. Fork or clone this repo and push to GitHub
2. Create a new Railway project → **Deploy from GitHub repo**
3. Railway auto-detects Python and uses the `Procfile`
4. Go to **Settings → Domains → Generate Domain** for a public URL

The `Procfile` uses gunicorn with a 120-second timeout to handle the longer simulation requests:

```
web: gunicorn app:app --workers 1 --timeout 120 --bind 0.0.0.0:$PORT
```

---

## FO26 predictions (22 May 2026)

| ATP | % | WTA | % |
|-----|---|-----|---|
| J. Sinner | 21.5% | I. Swiatek | 23.5% |
| N. Djokovic | 11.2% | C. Gauff | 13.6% |
| A. Zverev | 10.3% | E. Rybakina | 12.5% |
| C. Ruud | 4.3% | A. Sabalenka | 12.4% |
| D. Medvedev | 3.7% | A. Anisimova | 4.0% |

Based on 10,000 Monte Carlo simulations on the real draw. Tournament begins 24 May 2026.

---

## Tech stack

- **Backend:** Python, Flask, scikit-learn, pandas, numpy
- **Frontend:** Vanilla JS, HTML/CSS — no framework
- **Hosting:** Railway
- **Data pipeline:** [grand-slam-predictions](https://github.com/maanitmehta/grand-slam-predictions)
