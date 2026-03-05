# -*- coding: utf-8 -*-
"""
Rose des vents — version complète (2026-03-05)
Fonctionnalités :
- Bouton Démo (chargement de static/DemoBruitAmbiant.xlsx) + bouton Désactiver
- Téléversement CSV/XLSX
- Auto-plot (tracé automatique) ou mode manuel par bouton
- Palette de couleurs
- Fond transparent
- Export PNG/SVG (affiché seulement si figure existante)
- Filtre temporel
- Bins (classes de vitesse) personnalisés
- Détection automatique des colonnes avec sélecteurs visibles et pré‑sélection
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from windrose import WindroseAxes
import io

# -----------------------------------------------------------------------------
# Titre
# -----------------------------------------------------------------------------
st.title("Rose des vents")

# -----------------------------------------------------------------------------
# Sidebar — Fichier de démonstration (en premier)
# -----------------------------------------------------------------------------
if "use_demo" not in st.session_state:
    st.session_state["use_demo"] = False
if "demo_df" not in st.session_state:
    st.session_state["demo_df"] = None

st.sidebar.subheader("Fichier de démonstration")

col_demo_1, col_demo_2 = st.sidebar.columns(2)
if col_demo_1.button("Charger le fichier démo"):
    demo_path = "static/DemoBruitAmbiant.xlsx"
    try:
        st.session_state["demo_df"] = pd.read_excel(demo_path, engine="openpyxl")
        st.session_state["use_demo"] = True
        st.success("Fichier démo chargé avec succès !")
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier démo : {e}")

if col_demo_2.button("Désactiver le démo"):
    st.session_state["use_demo"] = False
    st.session_state["demo_df"] = None
    st.info("Mode démo désactivé. Téléversez un fichier pour continuer.")

st.sidebar.markdown("---")

# -----------------------------------------------------------------------------
# Sidebar — Options
# -----------------------------------------------------------------------------
st.sidebar.write("Options")
kmh = st.sidebar.checkbox("Vitesse en km/h")
titre_on = st.sidebar.checkbox("Inscrire le titre du graphique")
transparent_bg = st.sidebar.checkbox("Fond transparent")

# Auto-plot
st.sidebar.markdown("---")
auto_plot = st.sidebar.checkbox("Tracer automatiquement (sans bouton)", value=True)

# Palette
st.sidebar.markdown("---")
st.sidebar.subheader("Palette de couleurs")
palette_name = st.sidebar.selectbox(
    "Colormap",
    [
        "viridis", "plasma", "magma", "inferno", "cividis",
        "tab20", "tab10", "Set2", "Set3", "Pastel1", "Pastel2",
        "Accent", "Dark2", "Paired"
    ],
)
palette_reverse = st.sidebar.checkbox("Palette inversée", value=False)

# Bins personnalisés
st.sidebar.markdown("---")
st.sidebar.subheader("Classes de vitesse (bins)")
use_custom_bins = st.sidebar.checkbox("Classes personnalisées", value=False)

custom_bins = None
if use_custom_bins:
    bins_str = st.sidebar.text_input(
        "Valeurs des classes (séparées par des virgules)",
        value="0,1,2,3,5,10,20"
    )
    try:
        parsed = [float(x.strip()) for x in bins_str.split(",") if x.strip() != ""]
        # Nettoyage: tri + uniques
        parsed = sorted(set(parsed))
        if len(parsed) < 2:
            st.sidebar.error("Il faut au moins deux bornes pour définir des classes.")
            custom_bins = None
        else:
            custom_bins = parsed
            unit_label = "km/h" if kmh else "m/s"
            st.sidebar.caption(f"Les classes sont interprétées en {unit_label} (selon l'option d'unité ci-dessus).")
    except Exception:
        st.sidebar.error("Format invalide. Exemple : 0,1,2,5,10,20")
        custom_bins = None

# Export
st.sidebar.markdown("---")
st.sidebar.subheader("Export")
png_dpi = st.sidebar.slider("DPI (PNG)", 72, 600, 200, step=10)
export_png = st.sidebar.checkbox("Générer PNG", value=True)
export_svg = st.sidebar.checkbox("Générer SVG (vectoriel)", value=True)

# -----------------------------------------------------------------------------
# Chargement des données (démo prioritaire si activée)
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("Téléversez un fichier CSV ou Excel", type=["csv", "xlsx"])

df = None

if st.session_state.get("use_demo", False) and st.session_state.get("demo_df") is not None:
    df = st.session_state["demo_df"]
    st.write("Aperçu des données (fichier démo) :")
    st.dataframe(df.head())
elif uploaded_file is not None:
    # bascule automatique vers le fichier uploadé
    st.session_state["use_demo"] = False
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        st.write("Aperçu des données :")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")

if df is None:
    st.info("Veuillez téléverser un fichier ou charger le fichier démo.")
    st.stop()

# -----------------------------------------------------------------------------
# Sélection des colonnes — sélecteurs visibles + pré-sélection auto
# -----------------------------------------------------------------------------
cols = list(df.columns)

def _default_index(colname: str, columns: list):
    try:
        return columns.index(colname)
    except ValueError:
        return None

# Pré-sélection intelligente
time_default_idx = _default_index("Start Time", cols)
ws_default_idx = _default_index("Wind Speed avg", cols)
wd_default_idx = _default_index("Wind Dir. avg", cols)

# Sélecteurs (toujours visibles)
time_col = st.selectbox(
    "Sélectionnez la colonne de temps",
    cols,
    index=time_default_idx if time_default_idx is not None else 0,
)
wind_speed_col = st.selectbox(
    "Sélectionnez la colonne de vitesse du vent",
    cols,
    index=ws_default_idx if ws_default_idx is not None else 0,
)
wind_dir_col = st.selectbox(
    "Sélectionnez la colonne de direction du vent",
    cols,
    index=wd_default_idx if wd_default_idx is not None else 0,
)

# -----------------------------------------------------------------------------
# Conversion de la colonne de temps + filtre temporel
# -----------------------------------------------------------------------------
time_series = pd.to_datetime(df[time_col], errors="coerce")

if time_series.notna().any():
    tmin = time_series.min()
    tmax = time_series.max()
else:
    st.warning("Impossible d'interpréter la colonne de temps en datetime. Le filtre temporel sera désactivé.")
    tmin = None
    tmax = None

if tmin is not None and tmax is not None and tmin < tmax:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Période d'affichage")

    start_dt = st.sidebar.datetime_input(
        "Date-heure début",
        value=tmin.to_pydatetime(),
        min_value=tmin.to_pydatetime(),
        max_value=tmax.to_pydatetime(),
    )
    end_dt = st.sidebar.datetime_input(
        "Date-heure fin",
        value=tmax.to_pydatetime(),
        min_value=tmin.to_pydatetime(),
        max_value=tmax.to_pydatetime(),
    )

    if start_dt > end_dt:
        st.sidebar.error("La date de début est après la date de fin. Filtre ignoré.")
        df_plot = df.copy()
        time_series_plot = time_series.copy()
    else:
        mask_time = (time_series >= pd.to_datetime(start_dt)) & (time_series <= pd.to_datetime(end_dt))
        df_plot = df.loc[mask_time].copy()
        time_series_plot = time_series.loc[mask_time]
else:
    df_plot = df.copy()
    time_series_plot = time_series.copy()

# -----------------------------------------------------------------------------
# Fonction de tracé (réutilisable)
# -----------------------------------------------------------------------------

def build_windrose_figure(
    df_plot,
    wind_speed_col,
    wind_dir_col,
    time_series_plot,
    palette_name,
    palette_reverse,
    kmh,
    titre_on,
    transparent_bg,
    custom_bins=None,
):
    """Construit et retourne la figure de rose des vents."""
    fig = plt.figure(figsize=(8, 8))
    ax = WindroseAxes.from_ax(fig=fig)

    wind_speed = pd.to_numeric(df_plot[wind_speed_col], errors="coerce")
    wind_dir = pd.to_numeric(df_plot[wind_dir_col], errors="coerce")

    # Palette
    cmap_name = palette_name + ("_r" if palette_reverse else "")
    cmap = plt.get_cmap(cmap_name)

    # Unité + classes (bins) — les bins sont interprétés dans l'unité affichée
    if kmh:
        speed_vals = wind_speed * 3.6
        bins_to_use = custom_bins if custom_bins else None
        legend_title = "Vitesse du vent\n km/h"
    else:
        speed_vals = wind_speed
        bins_to_use = custom_bins if custom_bins else None
        legend_title = "Vitesse du vent\n m/s"

    # Tracé
    ax.bar(
        wind_dir,
        speed_vals,
        bins=bins_to_use,       # <= classes personnalisées
        normed=True,
        opening=0.8,
        edgecolor="white",
        cmap=cmap,
    )
    ax.set_legend(title=legend_title, loc="best")

    # Axe radial en %
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))

    # Titre
    if titre_on:
        if time_series_plot.notna().any():
            tmin_plot = time_series_plot.min()
            tmax_plot = time_series_plot.max()
            titre = f"Direction des vents mesurées de {tmin_plot} à {tmax_plot}"
        else:
            titre = "Direction des vents"
        ax.set_title(titre)

    # Fond transparent
    if transparent_bg:
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")

    return fig

# -----------------------------------------------------------------------------
# Tracé — Auto-plot ou bouton manuel
# -----------------------------------------------------------------------------
fig = None

if auto_plot:
    fig = build_windrose_figure(
        df_plot=df_plot,
        wind_speed_col=wind_speed_col,
        wind_dir_col=wind_dir_col,
        time_series_plot=time_series_plot,
        palette_name=palette_name,
        palette_reverse=palette_reverse,
        kmh=kmh,
        titre_on=titre_on,
        transparent_bg=transparent_bg,
        custom_bins=custom_bins,
    )
    st.pyplot(fig)
else:
    if st.button("Tracer la rose des vents"):
        fig = build_windrose_figure(
            df_plot=df_plot,
            wind_speed_col=wind_speed_col,
            wind_dir_col=wind_dir_col,
            time_series_plot=time_series_plot,
            palette_name=palette_name,
            palette_reverse=palette_reverse,
            kmh=kmh,
            titre_on=titre_on,
            transparent_bg=transparent_bg,
            custom_bins=custom_bins,
        )
        st.pyplot(fig)

# -----------------------------------------------------------------------------
# Export — seulement si un graphique a été généré
# -----------------------------------------------------------------------------
if fig is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Télécharger")

    if export_png:
        png_buf = io.BytesIO()
        fig.savefig(
            png_buf,
            format='png',
            dpi=png_dpi,
            transparent=transparent_bg,
            bbox_inches='tight',
            facecolor='none' if transparent_bg else 'white'
        )
        png_buf.seek(0)
        st.sidebar.download_button(
            label=f"Télécharger PNG ({png_dpi} DPI)",
            data=png_buf,
            file_name="rose_des_vents.png",
            mime="image/png"
        )

    if export_svg:
        svg_buf = io.BytesIO()
        fig.savefig(
            svg_buf,
            format='svg',
            transparent=transparent_bg,
            bbox_inches='tight',
            facecolor='none' if transparent_bg else 'white'
        )
        svg_buf.seek(0)
        st.sidebar.download_button(
            label="Télécharger SVG (vectoriel)",
            data=svg_buf,
            file_name="rose_des_vents.svg",
            mime="image/svg+xml"
        )
