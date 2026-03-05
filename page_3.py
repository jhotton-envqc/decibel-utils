
# Calculatrice_Lden.py – v5C.1 (table HTML + ligne Ln limitée à la nuit)
import io
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Calculatrice de Lden", layout="wide")

# ============================
# Styles & mise en page
# ============================
st.markdown(
    """
    <style>
    .main .block-container { max-width: 900px; padding-top: 1rem; padding-bottom: 2rem; }

    /* Carte contenant le tableau */
    .table-card { border: 1px solid rgba(49,51,63,0.25); border-radius: 10px; padding: 8px; background: rgba(255,255,255,0.02); }
    .html-table-wrap { overflow-x: auto; }

    /* Tableau HTML pur (centrage garanti) */
    table.lden { border-collapse: collapse; width: 100%; table-layout: fixed; }
    table.lden thead th { position: sticky; top: 0; background: rgba(0,0,0,0.15); }
    table.lden th, table.lden td { border: 1px solid rgba(255,255,255,0.06); padding: 6px 6px; font-size: 0.92rem; text-align: center; white-space: nowrap; }

    /* Contraintes de largeur par colonne */
    table.lden th.col-heure, table.lden td.col-heure { width: 72px; max-width: 72px; }
    table.lden th.col-laeq,  table.lden td.col-laeq  { width: 90px; max-width: 90px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================
# Couleurs
# ============================
COLOR_DAY = "#2ecc71"   # KPI vert
COLOR_EVE = "#f39c12"   # KPI orange
COLOR_NIGHT = "#3498db" # KPI bleu
ZONE_DAY   = "#27ae60"  # zone vert plus foncé
ZONE_EVE   = "#d35400"  # zone orange plus foncé
ZONE_NIGHT = "#2980b9"  # zone bleu plus foncé
ZONE_ALPHA = 0.18
BOUND_COLOR = "#444444"
BOUND_LW = 1.2

# ============================
# Fonctions utilitaires
# ============================

def period_hex(hour: int, ld_start: int, le_start: int, ln_start: int) -> str:
    if hour < ld_start or hour >= ln_start:
        return ZONE_NIGHT
    elif hour < le_start:
        return ZONE_DAY
    else:
        return ZONE_EVE


def db_mean(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return 0.0
    return float(10 * np.log10((10 ** (s / 10)).mean()))


def calc_ld_le_ln(laeq_24: pd.Series, ld_start: int, le_start: int, ln_start: int):
    laeq_24 = pd.to_numeric(laeq_24, errors="coerce")
    laeq_d = laeq_24.iloc[ld_start:le_start]
    laeq_e = laeq_24.iloc[le_start:ln_start]
    laeq_n = pd.concat([laeq_24.iloc[:ld_start], laeq_24.iloc[ln_start:24]])
    ld = db_mean(laeq_d)
    le = db_mean(laeq_e)
    ln = db_mean(laeq_n)
    return ld, le, ln, laeq_d, laeq_e, laeq_n


def calc_lden(ld: float, le: float, ln: float, ld_start: int, le_start: int, ln_start: int) -> float:
    duree_ld = le_start - ld_start
    duree_le = ln_start - le_start
    duree_ln = 24 - duree_ld - duree_le
    if ld == 0 or le == 0 or ln == 0:
        return 0.0
    lden = 10 * np.log10(((duree_ld * 10 ** (ld / 10))
                          + (duree_le * 10 ** ((le + 5) / 10))
                          + (duree_ln * 10 ** ((ln + 10) / 10))) / 24)
    return float(lden)


def format_hour(h: int) -> str:
    return f"{h:02d}h"


def parse_hour_series(time_series: pd.Series) -> pd.Series:
    s = time_series.copy()
    if np.issubdtype(s.dtype, np.datetime64):
        return pd.to_datetime(s, errors="coerce").dt.hour
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().mean() > 0.8:
        return s_num.round().astype("Int64")
    s_dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
    if s_dt.notna().mean() > 0.5:
        return s_dt.dt.hour
    s_str = s.astype(str).str.strip().str.lower().str.replace("h", ":", regex=False)
    s_dt2 = pd.to_datetime(s_str, errors="coerce")
    return s_dt2.dt.hour


def smart_to_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(" ", "", regex=False).str.replace(r"\s+", "", regex=True)
    has_comma = s.str.contains(",", regex=False)
    has_dot = s.str.contains(".", regex=False)
    mask_both = has_comma & has_dot
    s1 = s.where(~mask_both, s.str.replace(",", "", regex=False))
    has_comma_only = s1.str.contains(",", regex=False) & (~s1.str.contains(".", regex=False))
    s1 = s1.where(~has_comma_only, s1.str.replace(",", ".", regex=False))
    return pd.to_numeric(s1, errors="coerce")


def build_laeq_24_from_df(df: pd.DataFrame, time_col: Optional[str], laeq_col: str) -> pd.Series:
    laeq_raw = smart_to_numeric(df[laeq_col])
    if time_col is None:
        laeq = laeq_raw.copy()
        if len(laeq) >= 24:
            laeq = laeq.iloc[:24].reset_index(drop=True)
        else:
            laeq = laeq.reset_index(drop=True)
            laeq = pd.concat([laeq, pd.Series([np.nan] * (24 - len(laeq)))], ignore_index=True)
        return laeq
    hours = parse_hour_series(df[time_col])
    tmp = pd.DataFrame({"hour": hours, "laeq": laeq_raw})
    tmp = tmp.dropna(subset=["hour"]).copy()
    tmp["hour"] = pd.to_numeric(tmp["hour"], errors="coerce").astype("Int64")
    tmp = tmp[(tmp["hour"] >= 0) & (tmp["hour"] <= 23)]
    tmp = tmp.sort_values("hour")
    grouped = tmp.groupby("hour")["laeq"].apply(db_mean)
    laeq_24 = pd.Series([np.nan] * 24, index=range(24), dtype="float64")
    for h, val in grouped.items():
        laeq_24[int(h)] = val
    return laeq_24.reset_index(drop=True)


def _auto_name_columns(n: int) -> List[str]:
    return [f"Col {i}" for i in range(1, n + 1)]


def read_input_file(uploaded_file, has_header: bool) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    header_opt = 0 if has_header else None
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(data), sep=None, engine="python", header=header_opt, encoding="utf-8", dtype=str)
        except Exception:
            try:
                df = pd.read_csv(io.BytesIO(data), sep=None, engine="python", header=header_opt, encoding="latin-1", dtype=str)
            except Exception:
                try:
                    df = pd.read_csv(io.BytesIO(data), sep=";", header=header_opt, encoding="utf-8", dtype=str)
                except Exception:
                    df = pd.read_csv(io.BytesIO(data), sep=",", header=header_opt, encoding="utf-8", dtype=str)
    elif name.endswith(".xlsx"):
        df = pd.read_excel(io.BytesIO(data), engine="openpyxl", header=header_opt, dtype=object)
    elif name.endswith(".xls"):
        try:
            df = pd.read_excel(io.BytesIO(data), engine="xlrd", header=header_opt, dtype=object)
        except Exception:
            df = pd.read_excel(io.BytesIO(data), header=header_opt, dtype=object)
    else:
        raise ValueError("Format non supporté. Utilise CSV, XLS ou XLSX.")
    if not has_header:
        df.columns = _auto_name_columns(df.shape[1])
    return df


# ============================
# Session & Sidebar
# ============================
defaults = {
    "data": pd.DataFrame(),
    "laeq_24": pd.Series([0.0] * 24),
    "time_col": None,
    "laeq_col": None,
    "ld_start": 7,
    "le_start": 19,
    "ln_start": 23,
    "has_header": True,
    "ratio_left": 1.0,
    "ratio_right": 3.0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _on_header_toggle():
    st.session_state.time_col = None
    st.session_state.laeq_col = None

with st.sidebar:
    st.header("Paramètres")
    st.subheader("Périodes")
    st.session_state.ld_start = st.number_input("Heure début Ld", 0, 23, int(st.session_state.ld_start), step=1)
    st.session_state.le_start = st.number_input("Heure début Le", 0, 23, int(st.session_state.le_start), step=1)
    st.session_state.ln_start = st.number_input("Heure début Ln", 0, 23, int(st.session_state.ln_start), step=1)

    st.divider()
    st.subheader("Mise en page")
    st.session_state.ratio_left = st.slider("Largeur relative colonne gauche (tableau)", 0.6, 2.0, float(st.session_state.ratio_left), 0.1)
    st.session_state.ratio_right = st.slider("Largeur relative colonne droite (KPI + graphe)", 1.0, 4.0, float(st.session_state.ratio_right), 0.1)

    st.divider()
    with st.expander("Info ?", expanded=False):
        st.write("Voici les équations utilisées pour les calculs de cette page.")
        img_path = Path("static") / "lden.png"
        if img_path.exists():
            st.image(str(img_path), caption="Équations Lden", use_container_width=True)
        else:
            st.info("Image non trouvée. Place `lden.png` dans `static/lden.png`.")
        st.caption("Lden = 10·log10( (Ld + Le(+5 dB) + Ln(+10 dB)) pondéré par durées / 24 )")

# ============================
# Corps principal
# ============================
st.title("Calculatrice de Lden")

ld_start = int(st.session_state.ld_start)
le_start = int(st.session_state.le_start)
ln_start = int(st.session_state.ln_start)
ratio_left = float(st.session_state.ratio_left)
ratio_right = float(st.session_state.ratio_right)

if not (0 <= ld_start <= le_start <= ln_start <= 23):
    st.warning("⚠️ Assure-toi que 0 ≤ Ld ≤ Le ≤ Ln ≤ 23 (ordre croissant).")

st.subheader("Fichier d’entrées")
st.session_state.has_header = st.checkbox(
    "Le fichier contient une ligne d’entête (titres de colonnes)",
    value=bool(st.session_state.has_header),
    help="Décoche si la première ligne contient des données et non des noms de colonnes.",
    on_change=_on_header_toggle,
)

uploaded = st.file_uploader(
    "Sélectionner un fichier d'entrées (CSV / XLS / XLSX)",
    type=["csv", "xls", "xlsx"],
    accept_multiple_files=False,
)

if uploaded is not None:
    try:
        df = read_input_file(uploaded, has_header=st.session_state.has_header)
        st.session_state.data = df
        if not st.session_state.has_header:
            if st.session_state.time_col not in df.columns:
                st.session_state.time_col = "Col 1" if "Col 1" in df.columns else None
            if st.session_state.laeq_col not in df.columns:
                st.session_state.laeq_col = "Col 2" if "Col 2" in df.columns else df.columns[0]
        st.success("Fichier chargé ✅")
    except Exception as e:
        st.session_state.data = pd.DataFrame()
        st.error(f"Impossible de lire le fichier : {e}")

df = st.session_state.data

if not df.empty:
    cols = list(df.columns)
    st.subheader("Sélection des colonnes")
    c1, c2 = st.columns(2)
    with c1:
        time_options = ["(aucune — utiliser l'ordre des lignes)"] + cols
        desired_time = st.session_state.time_col if st.session_state.time_col in cols else "(aucune — utiliser l'ordre des lignes)"
        default_time = time_options.index(desired_time) if desired_time in time_options else 0
        time_choice = st.selectbox("Colonne temporelle", options=time_options, index=default_time)
        st.session_state.time_col = None if time_choice.startswith("(aucune") else time_choice
    with c2:
        guess = None
        for c in cols:
            if str(c).strip().lower() in ["laeq", "l_eq", "leq", "laeq_db", "laeq (db)"]:
                guess = c; break
        desired_laeq = st.session_state.laeq_col or guess or cols[0]
        default_laeq = cols.index(desired_laeq) if desired_laeq in cols else 0
        laeq_choice = st.selectbox("Colonne LAeq", options=cols, index=default_laeq)
        st.session_state.laeq_col = laeq_choice
    try:
        laeq_24 = build_laeq_24_from_df(df, st.session_state.time_col, st.session_state.laeq_col)
        st.session_state.laeq_24 = pd.to_numeric(laeq_24, errors="coerce")
    except Exception as e:
        st.error(f"Erreur pendant la préparation des données (24h) : {e}")
else:
    st.info("Charge un fichier pour activer la sélection des colonnes et les calculs.")

# ============================
# Affichage tableau (HTML pur) + résultats
# ============================
laeq_24 = st.session_state.laeq_24

# -- Construction du HTML pour le tableau --
rows = []
for h in range(24):
    bg = period_hex(h, ld_start, le_start, ln_start)
    heure_txt = format_hour(h)
    val = laeq_24.iloc[h] if h < len(laeq_24) else np.nan
    val_txt = "" if pd.isna(val) else f"{float(val):.2f}"
    rows.append(f"<tr style='background:{bg};'><td class='col-heure'>{heure_txt}</td><td class='col-laeq'>{val_txt}</td></tr>")

html_table = (
    "<div class='table-card'><div class='html-table-wrap'>"
    "<table class='lden'>"
    "<thead><tr><th class='col-heure'>Heure</th><th class='col-laeq'>LAeq</th></tr></thead>"
    f"<tbody>{''.join(rows)}</tbody>"
    "</table></div></div>"
)

st.subheader("Données horaires (colorées par période)")
left, right = st.columns([ratio_left, ratio_right])

with left:
    st.markdown(html_table, unsafe_allow_html=True)

with right:
    ld, le, ln, _, _, _ = calc_ld_le_ln(laeq_24, ld_start, le_start, ln_start)
    lden = calc_lden(ld, le, ln, ld_start, le_start, ln_start)

    st.subheader("Résultats")
    st.markdown(
        f"""
        <style>
        .kpi-card {{ border: 1px solid rgba(49, 51, 63, 0.2); border-radius: 10px; padding: 14px 16px; text-align: center; background: rgba(255,255,255,0.02); }}
        .kpi-title {{ font-size: 0.95rem; opacity: 0.75; margin-bottom: 6px; }}
        .kpi-value {{ font-size: 1.6rem; font-weight: 700; line-height: 1.2; }}
        .kpi-unit {{ font-size: 0.9rem; opacity: 0.8; margin-left: 4px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
            <div class='kpi-card'><div class='kpi-title'>Ld</div>
            <div class='kpi-value' style='color:{COLOR_DAY};'>{ld:.2f}<span class='kpi-unit'> dB</span></div></div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class='kpi-card'><div class='kpi-title'>Le</div>
            <div class='kpi-value' style='color:{COLOR_EVE};'>{le:.2f}<span class='kpi-unit'> dB</span></div></div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
            <div class='kpi-card'><div class='kpi-title'>Ln</div>
            <div class='kpi-value' style='color:{COLOR_NIGHT};'>{ln:.2f}<span class='kpi-unit'> dB</span></div></div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
        <div class='kpi-card'><div class='kpi-title'>Lden</div>
        <div class='kpi-value'>{lden:.2f}<span class='kpi-unit'> dB</span></div></div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Moyenne énergétique (dB) par période, pondérations +5 dB (soir) et +10 dB (nuit).")

    # ============================
    # Graphique (début à Ld) avec ticks 2 h + ligne Ln sur la nuit
    # ============================
    st.subheader("Graphique des LAeq horaires (début à Ld)")

    y_orig = pd.to_numeric(laeq_24, errors='coerce').to_numpy()
    shift = -ld_start
    y_shift = np.roll(y_orig, shift)

    heures_rel = np.arange(24)

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    duree_day = le_start - ld_start
    duree_eve = ln_start - le_start

    # Bandes de périodes
    ax.axvspan(0, duree_day, color=ZONE_DAY, alpha=ZONE_ALPHA)
    ax.axvspan(duree_day, duree_day + duree_eve, color=ZONE_EVE, alpha=ZONE_ALPHA)
    ax.axvspan(duree_day + duree_eve, 24, color=ZONE_NIGHT, alpha=ZONE_ALPHA)
    for x in [0, duree_day, duree_day + duree_eve, 24]:
        ax.axvline(x, color=BOUND_COLOR, linewidth=BOUND_LW, alpha=0.6)

    # Courbe horaire
    ax.plot(heures_rel, y_shift, marker='o', markersize=4, color='#1f2937', linewidth=1.6, label='LAeq horaire')

    # Ligne Lden globale (optionnelle)
    lden_val = calc_lden(ld, le, ln, ld_start, le_start, ln_start)
    if np.isfinite(lden_val):
        ax.axhline(lden_val, color='#b00020', linestyle='--', linewidth=2.0, label=f'Lden = {lden_val:.2f} dB')

    # ======= NOUVEAU: Ligne Ln limitée à la portion nuit =======
    night_start_rel = duree_day + duree_eve   # position x (relative) où commence la nuit sur l'axe recadré à Ld
    if np.isfinite(ln):
        ax.hlines(y=ln, xmin=night_start_rel, xmax=24, colors=ZONE_NIGHT, linestyles='-', linewidth=2.4, label=f'Ln = {ln:.2f} dB (nuit)')

    # Axes & ticks
    major_step = 2
    xticks_major = list(range(0, 24, major_step))
    ax.set_xticks(xticks_major)
    ax.set_xticklabels([f"{((h + ld_start) % 24):02d}h" for h in xticks_major], rotation=0)
    ax.set_xticks(range(24), minor=True)
    ax.tick_params(axis='x', which='minor', length=3, color='#777', labelbottom=False)
    ax.tick_params(axis='x', labelsize=9)

    ax.set_title("LAeq par heure (axe démarrant à Ld)")
    ax.set_xlabel("Heure")
    ax.set_ylabel("LAeq (dB)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout(pad=0.6)

    st.pyplot(fig, clear_figure=True)
