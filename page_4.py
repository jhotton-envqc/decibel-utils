# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 10:28:06 2025

@author: hotju02
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from windrose import WindroseAxes
import io

# Titre de la page
st.title("Rose des vents")

# Options supplémentaires (barre latérale)
st.sidebar.write("Options")
kmh = st.sidebar.checkbox("Vitesse en km/h")
titre_on = st.sidebar.checkbox("Inscrire le titre du graphique")
transparent_bg = st.sidebar.checkbox("Fond transparent")

# --- Palette de couleurs (nouveau)
st.sidebar.markdown("---")
st.sidebar.subheader("Palette de couleurs")
palette_name = st.sidebar.selectbox(
    "Colormap",
    options=[
        "viridis", "plasma", "magma", "inferno", "cividis",
        "tab20", "tab10", "Set2", "Set3", "Pastel1", "Pastel2",
        "Accent", "Dark2", "Paired"
    ],
    index=0
)
palette_reverse = st.sidebar.checkbox("Palette inversée", value=False)

# --- Section Export (dans la sidebar)
st.sidebar.markdown("---")
st.sidebar.subheader("Export")
png_dpi = st.sidebar.slider("DPI (PNG)", min_value=72, max_value=600, value=200, step=10)
export_png = st.sidebar.checkbox("Générer PNG", value=True)
export_svg = st.sidebar.checkbox("Générer SVG (vectoriel)", value=True)

# Chargement du fichier de données
uploaded_file = st.file_uploader("Téléversez un fichier CSV ou Excel", type=["csv", "xlsx"])

if uploaded_file:
    # Détection du type de fichier et lecture
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, engine='openpyxl')

    st.write("Aperçu des données :")
    st.dataframe(df.head())

    # Sélection des colonnes
    time_col = st.selectbox("Sélectionnez la colonne de temps", df.columns)
    wind_speed_col = st.selectbox("Sélectionnez la colonne de vitesse du vent", df.columns)
    wind_dir_col = st.selectbox("Sélectionnez la colonne de direction du vent", df.columns)

    # Conversion robuste de la colonne temps en datetime
    time_series = pd.to_datetime(df[time_col], errors="coerce", infer_datetime_format=True)

    # Déterminer les bornes min/max valides
    if time_series.notna().any():
        tmin = time_series.min()
        tmax = time_series.max()
    else:
        st.warning("Impossible d'interpréter la colonne de temps en datetime. Le filtre temporel sera désactivé.")
        tmin = None
        tmax = None

    # ---------------------------
    # 🗓️  Sélection de période dans la sidebar (deux champs date-heure)
    #     Valeurs par défaut = plage complète
    # ---------------------------
    if tmin is not None and tmax is not None and tmin < tmax:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Période d'affichage")

        # Deux champs datetime_input dans la sidebar (sans format=)
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

        # Validation basique : inversions de bornes
        if start_dt > end_dt:
            st.sidebar.error("La date de début est après la date de fin. Veuillez corriger.")
            # On n’applique pas de filtre si invalides
            df_plot = df.copy()
            time_series_plot = time_series.copy()
        else:
            mask_time = (time_series >= pd.to_datetime(start_dt)) & (time_series <= pd.to_datetime(end_dt))
            df_plot = df.loc[mask_time].copy()
            time_series_plot = time_series.loc[mask_time]
    else:
        # Pas de filtre si non valide
        df_plot = df.copy()
        time_series_plot = time_series.copy()

    # Tracer la rose des vents
    if st.button("Tracer la rose des vents"):
        fig = plt.figure(figsize=(8, 8))
        ax = WindroseAxes.from_ax(fig=fig)

        # Choix des données vitesse/direction
        wind_speed = pd.to_numeric(df_plot[wind_speed_col], errors="coerce")
        wind_dir = pd.to_numeric(df_plot[wind_dir_col], errors="coerce")

        # Déterminer la palette
        cmap_name = palette_name + ("_r" if palette_reverse else "")
        cmap = plt.get_cmap(cmap_name)

        # Conversion km/h si nécessaire + tracé avec palette
        if kmh:
            ax.bar(
                wind_dir, wind_speed * 3.6,
                normed=True, opening=0.8, edgecolor='white',
                cmap=cmap
            )
            ax.set_legend(title="Vitesse du vent\n km/h", loc="best")
        else:
            ax.bar(
                wind_dir, wind_speed,
                normed=True, opening=0.8, edgecolor='white',
                cmap=cmap
            )
            ax.set_legend(title="Vitesse du vent\n m/s", loc="best")

        # Axe radial en pourcentage
        fmt = '%.0f%%'
        yticks = mtick.FormatStrFormatter(fmt)  # utilisation de mtick (Option A validée)
        ax.yaxis.set_major_formatter(yticks)

        # Titre (si demandé) : utiliser min/max après filtre (ou fallback)
        if titre_on:
            if time_series_plot.notna().any():
                tmin_plot = time_series_plot.min()
                tmax_plot = time_series_plot.max()
                titre = f"Direction des vents mesurées de {tmin_plot} à {tmax_plot}"
            else:
                titre = "Direction des vents"
            ax.set_title(titre)

        # Fond transparent si coché
        if transparent_bg:
            fig.patch.set_alpha(0.0)
            ax.set_facecolor("none")

        # Affichage du graphique
        st.pyplot(fig)

        # ---------------------------
        # ⬇️  Boutons de téléchargement dans la sidebar
        # ---------------------------
        st.sidebar.markdown("---")
        st.sidebar.subheader("Télécharger")

        # PNG
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

        # SVG (vectoriel)
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
        
        

