
import streamlit as st



st.set_page_config(page_title="Décibel-Utils - test", layout="wide")

# Définition des pages
main_page = st.Page("main.py", title="Accueil")
page_2 = st.Page("page_2.py", title="Calculatrice décibels")
page_3 = st.Page("page_3.py", title="Calcul Lden")
page_4 = st.Page("page_4.py", title="Rose des vents")
page_5 = st.Page("page_5.py", title="Multi-traces")

# Navigation
pg = st.navigation(
    [main_page, page_2, page_3, page_4, page_5],
    position="sidebar"  # optionnel, mais améliore l'UI
)



# --- Sidebar ---
st.sidebar.title("Décibel-Utils")
st.sidebar.divider()  # optionnel

# Reste du contenu
st.sidebar.header("Paramètres")
option = st.sidebar.selectbox("Choix :", ["A", "B", "C"])



# Exécution de la page choisie
pg.run()
