
# Calculatrice_Lden.py
from pathlib import Path
import io

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Calculatrice de Lden", layout="wide")

# --- Couleurs (proches de ton RIO)
COLOR_DAY = "#2ecc71"      # vert
COLOR_EVE = "#f39c12"      # orange
COLOR_NIGHT = "#3498db"    # bleu


# -----------------------------
# Helpers calcul & parsing
# -----------------------------
def period_color(hour: int, ld_start: int, le_start: int, ln_start: int) -> str:
    if hour < ld_start:
        return COLOR_NIGHT
    elif hour >= ln_start:
        return COLOR_NIGHT
    elif ld_start <= hour < le_start:
        return COLOR_DAY
    else:
        return COLOR_EVE


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

    lden = 10 * np.log10(
        (
            (duree_ld * 10 ** (ld / 10)) +
            (duree_le * 10 ** ((le + 5) / 10)) +
            (duree_ln * 10 ** ((ln + 10) / 10))
        ) / 24
    )
    return float(lden)


def format_hour(h: int) -> str:
    return f"{h:02d}h00"


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

    s_str = s.astype(str).str.strip().str.lower()
    s_str = s_str.str.replace("h", ":", regex=False)
    s_dt2 = pd.to_datetime(s_str, errors="coerce")
    return s_dt2.dt.hour


def build_laeq_24_from_df(df: pd.DataFrame, time_col: str | None, laeq_col: str) -> pd.Series:
    laeq_raw = pd.to_numeric(df[laeq_col], errors="coerce")

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

    tmp = tmp.dropna(subset=["hour"])
    tmp["hour"] = pd.to_numeric(tmp["hour"], errors="coerce").astype("Int64")
    tmp = tmp[(tmp["hour"] >= 0) & (tmp["hour"] <= 23)]

    grouped = tmp.groupby("hour")["laeq"].apply(db_mean)

    laeq_24 = pd.Series([np.nan] * 24, index=range(24), dtype="float64")
    for h, val in grouped.items():
        laeq_24[int(h)] = val

    return laeq_24.reset_index(drop=True)


def _auto_name_columns(n: int) -> list[str]:
    return [f"Col {i}" for i in range(1, n + 1)]


def read_input_file(uploaded_file, has_header: bool) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    header_opt = 0 if has_header else None

    if name.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(data), sep=";", header=header_opt, encoding="utf-8")
        except Exception:
            df = pd.read_csv(io.BytesIO(data), sep=",", header=header_opt, encoding="utf-8")
    elif name.endswith(".xlsx"):
        df = pd.read_excel(io.BytesIO(data), engine="openpyxl", header=header_opt)
    elif name.endswith(".xls"):
        try:
            df = pd.read_excel(io.BytesIO(data), engine="xlrd", header=header_opt)
        except Exception:
            df = pd.read_excel(io.BytesIO(data), header=header_opt)
    else:
        raise ValueError("Format non supporté. Utilise CSV, XLS ou XLSX.")

    if not has_header:
        df.columns = _auto_name_columns(df.shape[1])

    return df


# -----------------------------
# Session state
# -----------------------------
defaults = {
    "data": pd.DataFrame(),
    "laeq_24": pd.Series([0.0] * 24),
    "time_col": None,
    "laeq_col": None,
    "ld_start": 7,
    "le_start": 19,
    "ln_start": 23,
    "has_header": True,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _on_header_toggle():
    st.session_state.time_col = None
    st.session_state.laeq_col = None


# -----------------------------
# Sidebar : Paramètres + Info
# -----------------------------
with st.sidebar:
    st.header("Paramètres")

    st.subheader("Périodes")
    st.session_state.ld_start = st.number_input(
        "Heure début Ld", min_value=0, max_value=23,
        value=int(st.session_state.ld_start), step=1
    )
    st.session_state.le_start = st.number_input(
        "Heure début Le", min_value=0, max_value=23,
        value=int(st.session_state.le_start), step=1
    )
    st.session_state.ln_start = st.number_input(
        "Heure début Ln", min_value=0, max_value=23,
        value=int(st.session_state.ln_start), step=1
    )

    st.divider()

    with st.expander("Info ?", expanded=False):
        st.write("Voici les équations utilisées pour les calculs de cette page.")
        img_path = Path("static") / "lden.png"
        if img_path.exists():
            st.image(str(img_path), caption="Équations Lden", use_container_width=True)
        else:
            st.info("Image non trouvée. Place `lden.png` dans `static/lden.png`.")
        st.caption("Lden = 10·log10( (Ld + Le(+5 dB) + Ln(+10 dB)) pondéré par durées / 24 )")


# -----------------------------
# UI principal
# -----------------------------
st.title("Calculatrice de Lden")

ld_start = int(st.session_state.ld_start)
le_start = int(st.session_state.le_start)
ln_start = int(st.session_state.ln_start)

if not (0 <= ld_start <= le_start <= ln_start <= 23):
    st.warning("⚠️ Assure-toi que 0 ≤ Ld ≤ Le ≤ Ln ≤ 23 (ordre croissant).")

st.subheader("Fichier d’entrées")

st.session_state.has_header = st.checkbox(
    "Le fichier contient une ligne d’entête (titres de colonnes)",
    value=bool(st.session_state.has_header),
    help="Décoche si la première ligne contient des données et non des noms de colonnes.",
    on_change=_on_header_toggle
)

uploaded = st.file_uploader(
    "Sélectionner un fichier d'entrées (CSV / XLS / XLSX)",
    type=["csv", "xls", "xlsx"],
    accept_multiple_files=False
)

if uploaded is not None:
    try:
        df = read_input_file(uploaded, has_header=st.session_state.has_header)
        st.session_state.data = df

        # Defaults Cas A (sans entête) : Temps=Col 1, LAeq=Col 2
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

        if not st.session_state.has_header and "Col 1" in cols:
            desired_time = "Col 1"
        else:
            desired_time = st.session_state.time_col

        default_time = time_options.index(desired_time) if desired_time in cols else 0

        time_choice = st.selectbox(
            "Colonne temporelle",
            options=time_options,
            index=default_time,
            help="Heure (0-23), date/heure, ou texte '07h00'. (Sans entête: Cas A = Col 1)"
        )
        st.session_state.time_col = None if time_choice.startswith("(aucune") else time_choice

    with c2:
        if not st.session_state.has_header and "Col 2" in cols:
            desired_laeq = "Col 2"
        else:
            guess = None
            for c in cols:
                if str(c).strip().lower() in ["laeq", "l_eq", "leq", "laeq_db", "laeq (db)"]:
                    guess = c
                    break
            desired_laeq = st.session_state.laeq_col or guess or cols[0]

        default_laeq = cols.index(desired_laeq) if desired_laeq in cols else 0

        laeq_choice = st.selectbox(
            "Colonne LAeq",
            options=cols,
            index=default_laeq,
            help="La colonne contenant les valeurs LAeq (dB). (Sans entête: Cas A = Col 2)"
        )
        st.session_state.laeq_col = laeq_choice

    # Construire laeq_24
    try:
        laeq_24 = build_laeq_24_from_df(df, st.session_state.time_col, st.session_state.laeq_col)
        st.session_state.laeq_24 = pd.to_numeric(laeq_24, errors="coerce")
    except Exception as e:
        st.error(f"Erreur pendant la préparation des données (24h) : {e}")
else:
    st.info("Charge un fichier pour activer la sélection des colonnes et les calculs.")


# -----------------------------
# Affichage tableau + résultats
# -----------------------------
laeq_24 = st.session_state.laeq_24

table = pd.DataFrame({
    "Heure": [format_hour(h) for h in range(24)],
    "LAeq": laeq_24.values
})

def style_period(row):
    h = int(row.name)
    color = period_color(h, ld_start, le_start, ln_start)
    return [f"background-color: {color}; color: white; text-align: center;"] * len(row)

# ✅ (2) Centrer titres + valeurs
table_styler = (
    table.style
    .apply(style_period, axis=1)
    .format({"LAeq": "{:.2f}"})
    .set_properties(**{"text-align": "center"})  # cellules
    .set_table_styles([  # entêtes
        {"selector": "th", "props": [("text-align", "center")]}
    ])
)

# ✅ (1) Hauteur suffisante pour voir 24 lignes sans scroll
# Ajuste si tu changes le thème / densité : 28-35 px/ligne selon environnements.
ROW_PX = 35
HEADER_PX = 38
PADDING_PX = 12
TABLE_HEIGHT = HEADER_PX + (24 * ROW_PX) + PADDING_PX

st.subheader("Données horaires (colorées par période)")

left, right = st.columns([1.2, 1.0])

with left:
    st.dataframe(
        table_styler,
        use_container_width=True,
        height=TABLE_HEIGHT  # pas de scroll pour 24 lignes
    )

with right:
    ld, le, ln, _, _, _ = calc_ld_le_ln(laeq_24, ld_start, le_start, ln_start)
    lden = calc_lden(ld, le, ln, ld_start, le_start, ln_start)

    st.subheader("Résultats")

    # ✅ (3) Valeurs colorées comme le tableau (Ld vert, Le orange, Ln bleu)
    # On fait des "cartes" HTML car st.metric ne supporte pas la couleur de texte.
    st.markdown(
        f"""
        <style>
        .kpi-card {{
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 10px;
            padding: 14px 16px;
            text-align: center;
            background: rgba(255,255,255,0.02);
        }}
        .kpi-title {{
            font-size: 0.95rem;
            opacity: 0.75;
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 1.6rem;
            font-weight: 700;
            line-height: 1.2;
        }}
        .kpi-unit {{
            font-size: 0.9rem;
            opacity: 0.8;
            margin-left: 4px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Ld</div>
                <div class="kpi-value" style="color:{COLOR_DAY};">{ld:.2f}<span class="kpi-unit"> dB</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Le</div>
                <div class="kpi-value" style="color:{COLOR_EVE};">{le:.2f}<span class="kpi-unit"> dB</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Ln</div>
                <div class="kpi-value" style="color:{COLOR_NIGHT};">{ln:.2f}<span class="kpi-unit"> dB</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Lden</div>
            <div class="kpi-value">{lden:.2f}<span class="kpi-unit"> dB</span></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.caption("Moyenne énergétique (dB) par période, pondérations +5 dB (soir) et +10 dB (nuit).")

with st.expander("Aperçu du fichier importé", expanded=False):
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.write("Aucun fichier chargé.")
