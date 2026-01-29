# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 09:59:43 2025

@author: hotju02
"""

import streamlit as st
import pandas as pd
import numpy as np
import openpyxl

#t.title("This is the title page 3")

st.markdown("# Calculateur de l'indice acoustique Lden")
st.sidebar.markdown("# Lden")
st.sidebar.image("static/lden.png", use_container_width=True)

# Titre et description de l'application
##st.title("Calculateur de l'indice acoustique Lden")
st.write("""
Cette application calcule l'indice Lden à partir d'un fichier Excel. 
Votre fichier doit contenir deux colonnes: une pour l'heure (format hh:mm:ss) et une pour le niveau acoustique LAeq (en dB). Les données doivent commencer à la deuxième ligne.
""")

# Section pour le téléchargement du fichier
uploaded_file = st.file_uploader("Veuillez choisir un fichier Excel", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Lecture du fichier Excel en ignorant la première ligne
        df = pd.read_excel(uploaded_file, header=0)
        
        # Renommer les colonnes pour la clarté si nécessaire
        ##df.columns = ["Heure", "LAeq"]
        df.columns.values[0] = 'Heure'
        df.columns.values[1] = 'LAeq'
        
        st.write("Aperçu des données chargées:")
        st.dataframe(df)#.head()

        
        # Vérification des colonnes nécessaires
        if "Heure" in df.columns and "LAeq" in df.columns:
            # Assurer que la colonne 'Heure' est de type datetime pour extraire les heures
            df["Heure"] = pd.to_datetime(df["Heure"], format="%H:%M:%S").dt.hour

            # Séparer les données en périodes Jours/Soir/Nuit
            # Jour: 7h à 19h
            # Soir: 19h à 23h
            # Nuit: 23h à 7h
            
            laeq_jour = df[(df["Heure"] >= 7) & (df["Heure"] < 19)]["LAeq"]
            laeq_soir = df[(df["Heure"] >= 19) & (df["Heure"] < 23)]["LAeq"]
            laeq_nuit_1 = df[df["Heure"] >= 23]["LAeq"]
            laeq_nuit_2 = df[df["Heure"] < 7]["LAeq"]
            laeq_nuit = pd.concat([laeq_nuit_1, laeq_nuit_2])
            
            # Calcul des niveaux LAeq moyens par période
            L_jour = 10 * np.log10(np.mean(10**(laeq_jour/10))) if not laeq_jour.empty else -np.inf
            L_soir = 10 * np.log10(np.mean(10**(laeq_soir/10))) if not laeq_soir.empty else -np.inf
            L_nuit = 10 * np.log10(np.mean(10**(laeq_nuit/10))) if not laeq_nuit.empty else -np.inf

            # L'indice acoustique Lden est un moyenne énergétique pondérée sur une période de 24h
            # Période de jour : 12h
            # Période de soir : 4h (avec une pénalité de +5 dB)
            # Période de nuit : 8h (avec une pénalité de +10 dB)
            
            num_j = 12 * 10**(L_jour/10)
            num_s = 4 * 10**((L_soir+5)/10)
            num_n = 8 * 10**((L_nuit+10)/10)

            # Calcul final du Lden
            if (num_j + num_s + num_n) > 0:
                lden = 10 * np.log10((num_j + num_s + num_n) / 24)
                st.subheader("Résultat du calcul Lden")
                st.metric(label="Lden", value=f"{lden:.2f} dB")
            else:
                st.warning("Impossible de calculer le Lden. Assurez-vous que vos données couvrent les trois périodes (jour, soir, nuit).")

        else:
            st.error("Erreur : Le fichier ne contient pas les colonnes requises ou leur format est incorrect.")
        
    except Exception as e:
        st.error(f"Une erreur s'est produite lors du traitement du fichier : {e}")

