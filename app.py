import sys
import os
import threading

# All paths are relative to this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

from flask import Flask, jsonify, render_template, request
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

app = Flask(__name__)

VISIBLE_ROUNDS = ['Round of 32', 'Round of 16', 'Quarter-Finals', 'Semi-Finals', 'Final']

STORYLINES = {
    'atp': [
        {'icon': '🛡', 'tag': 'The Favourite',
         'title': "Sinner's Title Defence",
         'body': "The world No.1 arrives at Roland-Garros as Australian Open champion. A Melbourne–Paris double would cement his place among the greats."},
        {'icon': '👑', 'tag': 'Grand Slam Hunt',
         'title': "Djokovic Still Chasing History",
         'body': "Two-time champion, 38 years old, and still dangerous on clay. Every Grand Slam Djokovic enters feels like a farewell — and he keeps proving people wrong."},
        {'icon': '⚡', 'tag': 'The Challenger',
         'title': "Zverev's Defining Moment",
         'body': "Alexander Zverev has the clay-court game to beat anyone in the draw. Multiple Slam final near-misses have sharpened his hunger. This could be his year."},
        {'icon': '🇫🇷', 'tag': 'Home Favourite',
         'title': "Fils and the Chatrier Effect",
         'body': "Arthur Fils enters as a wildcard carrying the weight of French expectations. A partisan Philippe-Chatrier crowd could lift him to places his ranking shouldn't reach."},
    ],
    'wta': [
        {'icon': '👑', 'tag': 'Comeback Story',
         'title': "Swiatek Hunting a 5th Title",
         'body': "Seeded 3rd — unusual for a four-time champion. Swiatek must navigate a brutal draw to reclaim her Paris throne, and my model still backs her as the favourite."},
        {'icon': '🔥', 'tag': 'World No.1',
         'title': "Sabalenka's Missing Major",
         'body': "Aryna Sabalenka has won at Melbourne and New York. Paris is the one that keeps eluding her. She arrives as the top seed and in career-best form on clay."},
        {'icon': '⭐', 'tag': 'Next Generation',
         'title': "Gauff's Clay Progression",
         'body': "Coco Gauff has quietly built one of the best clay-court records in the draw. The US Open champion is no longer a surprise — she's a genuine title threat."},
        {'icon': '🎯', 'tag': 'Dark Horse',
         'title': "Rybakina the Underrated",
         'body': "Elena Rybakina's flat, penetrating groundstrokes don't care about surface. She's seeded 2nd for a reason, yet somehow still feels underrated by the public."},
    ]
}

# ── Pre-computed landing cache (populated at startup) ─────────────────
_landing_cache = {}


def readable(name):
    if name and '.' in name and ' ' in name:
        last, init = name.split(' ', 1)
        return f"{init.replace('.', '')}. {last}"
    return name


def to_canonical(r):
    if r and '. ' in r:
        init, rest = r.split('. ', 1)
        last = rest.strip().split()[-1]
        return f"{last} {init}."
    return r


def _bootstrap():
    from scripts.predict_match import predict_match
    from scripts.name_utils import canonical_name
    from config.tournaments import TOURNAMENTS
    return predict_match, canonical_name, TOURNAMENTS


def _load_draw(tour, cfg, cn_fn):
    draw = pd.read_csv(cfg['draws'][tour], engine='python')
    draw['player_A'] = draw['player_A'].apply(cn_fn)
    draw['player_B'] = draw['player_B'].apply(cn_fn)
    return draw


def _load_elo(tour, cn_fn):
    path = Path('data/processed') / tour / 'elo_snapshot.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df['player'] = df['player'].apply(cn_fn)
    return df.set_index('player')


# ── Simulations ───────────────────────────────────────────────────────

def run_bracket(tour, tournament='fo26'):
    predict_match, cn_fn, TOURNAMENTS = _bootstrap()
    cfg = TOURNAMENTS[tournament]
    draw = _load_draw(tour, cfg, cn_fn)

    players = []
    for _, row in draw.iterrows():
        players += [row['player_A'], row['player_B']]

    for _ in range(2):
        players = [
            (players[i] if np.random.rand() <
             predict_match(players[i], players[i+1], tour=tour, tournament=tournament)
             else players[i+1])
            for i in range(0, len(players), 2)
        ]

    rounds_out, biggest_upset, min_winner_p = [], None, 1.0

    for rname in VISIBLE_ROUNDS:
        if len(players) <= 1:
            break
        matches, nxt = [], []
        for i in range(0, len(players), 2):
            A, B = players[i], players[i+1]
            p = predict_match(A, B, tour=tour, tournament=tournament)
            win = np.random.rand() < p
            winner, loser = (A, B) if win else (B, A)
            winner_p = p if win else 1 - p
            nxt.append(winner)
            matches.append({
                'a': readable(A), 'b': readable(B),
                'winner': readable(winner),
                'winner_prob': round(winner_p * 100, 1)
            })
            if winner_p < 0.4 and winner_p < min_winner_p:
                min_winner_p = winner_p
                biggest_upset = {
                    'round': rname,
                    'winner': readable(winner), 'loser': readable(loser),
                    'winner_prob': round(winner_p * 100, 1),
                    'loser_prob': round((1 - winner_p) * 100, 1),
                }
        rounds_out.append({'name': rname, 'matches': matches})
        players = nxt

    champion = readable(players[0])
    path = []
    for r in rounds_out:
        for m in r['matches']:
            if m['winner'] == champion:
                beaten = m['b'] if m['a'] == champion else m['a']
                path.append({'round': r['name'], 'beat': beaten})

    return {
        'rounds': rounds_out, 'champion': champion,
        'champion_path': path, 'upset': biggest_upset,
        'tour': tour.upper()
    }


def run_probs(tour, tournament='fo26', n_sims=3_000):
    predict_match, cn_fn, TOURNAMENTS = _bootstrap()
    cfg = TOURNAMENTS[tournament]
    draw = _load_draw(tour, cfg, cn_fn)

    def one():
        ps = []
        for _, row in draw.iterrows():
            ps += [row['player_A'], row['player_B']]
        while len(ps) > 1:
            ps = [
                (ps[i] if np.random.rand() <
                 predict_match(ps[i], ps[i+1], tour=tour, tournament=tournament)
                 else ps[i+1])
                for i in range(0, len(ps), 2)
            ]
        return ps[0]

    counts = Counter(one() for _ in range(n_sims))
    rows = (
        pd.DataFrame.from_dict(counts, orient='index', columns=['wins'])
        .assign(prob=lambda df: (df['wins'] / n_sims * 100).round(1))
        .sort_values('prob', ascending=False).reset_index()
        .rename(columns={'index': 'player'})
    )
    rows['player'] = rows['player'].apply(readable)
    return rows[['player', 'prob']].to_dict(orient='records')


def _warm_cache():
    """Pre-compute landing odds for both tours at startup."""
    for tour in ('atp', 'wta'):
        try:
            print(f'  Pre-computing {tour.upper()} landing odds…')
            _landing_cache[tour] = run_probs(tour, n_sims=1_000)
            print(f'  ✓ {tour.upper()} done')
        except Exception as e:
            print(f'  ✗ {tour.upper()} failed: {e}')


# ── Routes ────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/landing/<tour>')
def landing_route(tour):
    if tour not in ('atp', 'wta'):
        return jsonify({'error': 'Invalid tour'}), 400
    # Serve from cache if ready, otherwise compute on demand
    probs = _landing_cache.get(tour) or run_probs(tour, n_sims=500)
    return jsonify({
        'favourites': probs[:8],
        'storylines': STORYLINES[tour],
        'tour': tour.upper()
    })


@app.route('/bracket/<tour>')
def bracket_route(tour):
    if tour not in ('atp', 'wta'):
        return jsonify({'error': 'Invalid tour'}), 400
    try:
        return jsonify(run_bracket(tour))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/simulate/<tour>')
def simulate_route(tour):
    if tour not in ('atp', 'wta'):
        return jsonify({'error': 'Invalid tour'}), 400
    try:
        return jsonify({'results': run_probs(tour, n_sims=3_000), 'tour': tour.upper()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/compare/<tour>')
def compare_route(tour):
    if tour not in ('atp', 'wta'):
        return jsonify({'error': 'Invalid tour'}), 400
    try:
        def summarise(sim):
            final = sim['rounds'][-1]['matches'][0]
            finalist = final['b'] if sim['champion'] == final['a'] else final['a']
            sf_matches = sim['rounds'][-2]['matches'] if len(sim['rounds']) >= 2 else []
            return {
                'champion': sim['champion'], 'finalist': finalist,
                'semis': [{'winner': m['winner'],
                           'loser': m['b'] if m['a'] == m['winner'] else m['a']}
                          for m in sf_matches],
                'path': sim['champion_path'], 'upset': sim['upset'],
            }
        return jsonify({
            'sim1': summarise(run_bracket(tour)),
            'sim2': summarise(run_bracket(tour)),
            'tour': tour.upper()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/h2h')
def h2h_route():
    tour = request.args.get('tour', 'atp')
    a_r  = request.args.get('a', '')
    b_r  = request.args.get('b', '')
    if not a_r or not b_r:
        return jsonify({'error': 'Two players required'}), 400
    try:
        predict_match, cn_fn, TOURNAMENTS = _bootstrap()
        elo_df  = _load_elo(tour, cn_fn)
        surface = TOURNAMENTS['fo26']['surface']
        cn_a, cn_b = to_canonical(a_r), to_canonical(b_r)
        p = predict_match(cn_a, cn_b, tour=tour, tournament='fo26')

        a_elo = float(elo_df.loc[cn_a, f'elo_{surface}']) if (elo_df is not None and cn_a in elo_df.index) else 1500
        b_elo = float(elo_df.loc[cn_b, f'elo_{surface}']) if (elo_df is not None and cn_b in elo_df.index) else 1500
        diff  = abs(a_elo - b_elo)
        fav   = a_r if p >= 0.5 else b_r

        if diff > 400:
            insight = f"{fav} has a commanding clay Elo advantage — this is a heavy mismatch on paper."
        elif diff > 150:
            insight = f"{fav} leads on clay Elo by {diff:.0f} points — a clear edge, but anything can happen."
        elif diff > 50:
            insight = f"A competitive match. {fav} has a slight clay Elo edge ({diff:.0f} pts) that tilts the model."
        else:
            insight = "These two are essentially dead level on clay Elo. The model calls it a coin flip."

        return jsonify({
            'playerA': a_r, 'playerB': b_r,
            'prob_A': round(p * 100, 1), 'prob_B': round((1 - p) * 100, 1),
            'insight': insight
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


_H2H_CACHE = {}

@app.route('/h2h-history')
def h2h_history_route():
    tour = request.args.get('tour', 'atp')
    a_r  = request.args.get('a', '')
    b_r  = request.args.get('b', '')
    if not a_r or not b_r:
        return jsonify({'error': 'Two players required'}), 400
    try:
        if tour not in _H2H_CACHE:
            p = Path(f'data/h2h_{tour}.json')
            if p.exists():
                import json as _json
                with open(p) as f:
                    _H2H_CACHE[tour] = _json.load(f)
            else:
                _H2H_CACHE[tour] = {}

        db = _H2H_CACHE[tour]
        cn_a, cn_b = to_canonical(a_r), to_canonical(b_r)
        key = f"{min(cn_a, cn_b)}|{max(cn_a, cn_b)}"
        rec = db.get(key)

        if not rec:
            return jsonify({'found': False})

        # Ensure p1/p2 labels match the requested order (a_r first)
        if rec['p1'] == a_r:
            return jsonify({'found': True, **rec})
        else:
            # Swap perspective so player A is always the left player
            return jsonify({
                'found': True,
                'p1': rec['p2'], 'p2': rec['p1'],
                'overall': {'p1': rec['overall']['p2'], 'p2': rec['overall']['p1']},
                'clay':    {'p1': rec['clay']['p2'],    'p2': rec['clay']['p1']},
                'last':    rec['last'],
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/draw-probs/<tour>')
def draw_probs_route(tour):
    """Win probabilities for all R128 matchups — used by My Picks feature."""
    if tour not in ('atp', 'wta'):
        return jsonify({'error': 'Invalid tour'}), 400
    try:
        predict_match, cn_fn, TOURNAMENTS = _bootstrap()
        cfg = TOURNAMENTS['fo26']
        draw = _load_draw(tour, cfg, cn_fn)
        probs = []
        for _, row in draw.iterrows():
            p = predict_match(row['player_A'], row['player_B'], tour=tour, tournament='fo26')
            probs.append({'prob_a': round(p * 100, 1), 'prob_b': round((1 - p) * 100, 1)})
        return jsonify({'probs': probs, 'tour': tour.upper()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/draw/<tour>')
def draw_route(tour):
    if tour not in ('atp', 'wta'):
        return jsonify({'error': 'Invalid tour'}), 400
    try:
        _, cn_fn, TOURNAMENTS = _bootstrap()
        cfg = TOURNAMENTS['fo26']
        draw = _load_draw(tour, cfg, cn_fn)
        matches = [
            {'a': readable(row['player_A']), 'b': readable(row['player_B'])}
            for _, row in draw.iterrows()
        ]
        return jsonify({'matches': matches, 'tour': tour.upper()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/players/<tour>')
def players_route(tour):
    if tour not in ('atp', 'wta'):
        return jsonify({'error': 'Invalid tour'}), 400
    try:
        _, cn_fn, TOURNAMENTS = _bootstrap()
        cfg  = TOURNAMENTS['fo26']
        draw = _load_draw(tour, cfg, cn_fn)
        players = sorted(set(
            draw['player_A'].apply(readable).tolist() +
            draw['player_B'].apply(readable).tolist()
        ))
        return jsonify({'players': players})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/name-map/<tour>')
def name_map_route(tour):
    """Returns {readable_name: full_name} for every player in the draw.
    Used by the frontend to build Wikipedia search queries for any player."""
    if tour not in ('atp', 'wta'):
        return jsonify({'error': 'Invalid tour'}), 400
    try:
        _, cn_fn, TOURNAMENTS = _bootstrap()
        cfg = TOURNAMENTS['fo26']
        raw = pd.read_csv(cfg['draws'][tour], engine='python')
        name_map = {}
        for col in ['player_A', 'player_B']:
            for full_name in raw[col]:
                r = readable(cn_fn(full_name))
                name_map[r] = full_name   # e.g. "J. Sinner" → "Jannik Sinner"
        return jsonify(name_map)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/player-stats/<tour>/<path:name>')
def player_stats_route(tour, name):
    try:
        _, cn_fn, TOURNAMENTS = _bootstrap()
        from config.surfaces import SURFACE_TRAINING_FILTER
        surface = TOURNAMENTS['fo26']['surface']
        base    = Path('data/processed') / tour
        cn      = to_canonical(name)

        elo_global = elo_surface = win_rate = matches_played = None

        elo_path = base / 'elo_snapshot.csv'
        if elo_path.exists():
            elo_df = pd.read_csv(elo_path)
            elo_df['player'] = elo_df['player'].apply(cn_fn)
            elo_df = elo_df.set_index('player')
            if cn in elo_df.index:
                elo_global  = int(elo_df.loc[cn, 'elo_global'])
                elo_surface = int(elo_df.loc[cn, f'elo_{surface}'])

        stats_file = (base / 'rolling_player_stats.csv'
                      if SURFACE_TRAINING_FILTER[surface] == ['hard']
                      else base / 'rolling_player_stats_all_surfaces.csv')
        if stats_file.exists():
            stats = pd.read_csv(stats_file, low_memory=False)
            stats['player'] = stats['player'].apply(cn_fn)
            rows = stats[stats['player'] == cn].sort_values('date')
            if len(rows):
                row = rows.iloc[-1]
                win_rate       = round(float(row.get('winrate_lastN', 0)) * 100, 1)
                matches_played = int(row.get('matches_played_lastN', 0))

        return jsonify({
            'name': name, 'elo_global': elo_global,
            'elo_surface': elo_surface, 'surface': surface.capitalize(),
            'win_rate': win_rate, 'matches_played': matches_played,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Startup ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Warming landing cache in background…')
    threading.Thread(target=_warm_cache, daemon=True).start()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5050)))
else:
    # Gunicorn startup — warm cache in background thread
    print('Warming landing cache in background…')
    threading.Thread(target=_warm_cache, daemon=True).start()
