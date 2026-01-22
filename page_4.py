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
# Options supplémentaires
st.sidebar.write("Options")
kmh = st.sidebar.checkbox("Vitesse en km/h")
titre_on = st.sidebar.checkbox("inscrire titre du graphique")
transparent_bg = st.sidebar.checkbox("Fond transparent")
download_image = st.sidebar.checkbox("Télécharger l'image")

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

    # Options supplémentaires
    ##kmh = st.sidebar.checkbox("Vitesse en km/h")
    ##titre_on = st.checkbox("inscrire titre du graphique")
    ##transparent_bg = st.checkbox("Fond transparent")
    ##download_image = st.checkbox("Télécharger l'image")

    # Tracer la rose des vents
    if st.button("Tracer la rose des vents"):
        fig = plt.figure(figsize=(8, 8))
        ax = WindroseAxes.from_ax(fig=fig)
        if kmh:
           ax.bar(df[wind_dir_col], df[wind_speed_col]*3.6, normed=True, opening=0.8, edgecolor='white')
           ax.set_legend(title="Vitesse du vent\n km/h", loc="best")
        else:
           ax.bar(df[wind_dir_col], df[wind_speed_col], normed=True, opening=0.8, edgecolor='white')
           ax.set_legend(title="Vitesse du vent\n m/s", loc="best")            
        # Format radius axis to percentages
        fmt = '%.0f%%' 
        yticks = mtick.FormatStrFormatter(fmt)
        ax.yaxis.set_major_formatter(yticks)
        
        if titre_on: ax.set_title("Direction des vents mesurées de {0} à {1}".format(str(df[time_col][0]),str(df[time_col].iloc[-1])))

        # Affichage du graphique
        st.pyplot(fig)

        # Sauvegarde de l'image si demandé
        if download_image:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', transparent=transparent_bg)
            buf.seek(0)
            st.download_button(
                label="Télécharger l'image",
                data=buf,
                file_name="rose_des_vents.png",
                mime="image/png"
            )
