
import streamlit as st



st.set_page_config(page_title="Décibel-Utils", layout="wide")

# Définition des pages
main_page = st.Page("main.py", title="Accueil")
page_2 = st.Page("page_2.py", title="Calculatrice décibels")
page_3 = st.Page("page_3.py", title="Calculatrice Lden")
page_4 = st.Page("page_4.py", title="Rose des vents")
page_5 = st.Page("page_5.py", title="Multi-traces")

# Navigation
pg = st.navigation(
    [main_page, page_2, page_3, page_4, page_5],
    position="sidebar"  # optionnel, mais améliore l'UI
)



import streamlit as st

# --- Sidebar ---
with st.sidebar:
    # Titre centré (HTML autorisé)
    st.markdown(
        "<h1 style='text-align:center; margin-bottom:0;'>🎧 Décibel-Utils</h1>",
        unsafe_allow_html=True
    )

    # Centrer un bouton dans la sidebar avec des colonnes
    left, center, right = st.columns([0.5, 2, 0.5])
    with center:
        if st.button("🎈 Célébration ! 🎈"):
            st.balloons()

    st.divider()

# Reste du contenu
#st.sidebar.header("Paramètres")
#option = st.sidebar.selectbox("Choix :", ["A", "B", "C"])



# Exécution de la page choisie
pg.run()
