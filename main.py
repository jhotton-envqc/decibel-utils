
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 09:53:36 2025
@author: hotju02
"""
import streamlit as st

st.markdown("# 🎧 décibel-utils")

st.subheader("Outils d’analyse, de calcul et de visualisation pour des données acoustiques et météo")

st.markdown(
    """
pour info: julien.hotton@environnement.gouv.qc.ca
    """
)

st.divider()

st.subheader("Aller directement à :")

# Single column list of page links (no icons)
st.page_link("page_2.py", label="Calculatrice décibels")
st.page_link("page_3.py", label="Calculatrice Lden")
st.page_link("page_4.py", label="Rose des vents")
st.page_link("page_5.py", label="Multi-traces")


