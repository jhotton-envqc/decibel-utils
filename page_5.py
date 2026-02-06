import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cycler  # 🔧 pour définir la palette
from io import BytesIO
from datetime import datetime

# st.set_page_config(page_title="Multi-Trace", layout="wide")

# ------------------------------------------------------------
# FONCTIONS
# ------------------------------------------------------------

def calculate_mean_direction_and_sigma_theta(wind_directions):
    wind_directions_rad = np.radians(wind_directions - 270)
    mean_sin = np.mean(np.sin(wind_directions_rad))
    mean_cos = np.mean(np.cos(wind_directions_rad))
    mean_direction = np.degrees(np.arctan2(mean_sin, mean_cos))
    sigma_theta = np.degrees(np.sqrt(-2 * np.log(np.sqrt(mean_sin**2 + mean_cos**2))))
    return mean_direction, sigma_theta


def compute_wind_vectors(df):
    results = []
    for minute, group in df.groupby(pd.Grouper(key="Start Time", freq="5Min")):
        mean_speed = group["Wind Speed avg"].mean()
        mean_dir, sigma = calculate_mean_direction_and_sigma_theta(group["Wind Dir. avg"])
        results.append([minute, mean_speed, mean_dir, sigma])
    return pd.DataFrame(results, columns=["Start Time", "MeanWindSpeed", "MeanWindDirection", "SigmaTheta"])


# 🔧 Palette: fonction utilitaire
def get_color_cycle(palette_name: str):
    """
    Retourne une liste de couleurs pour le cycle des axes Matplotlib selon la palette choisie.
    """
    palette_name = palette_name.lower()
    if palette_name == "matplotlib (défaut)":
        # Récupérer les couleurs par défaut de la version courante de Matplotlib
        return plt.rcParamsDefault["axes.prop_cycle"].by_key().get("color", ["#1f77b4", "#ff7f0e", "#2ca02c",
                                                                            "#d62728", "#9467bd", "#8c564b",
                                                                            "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"])
    if palette_name == "tableau 10":
        return list(plt.get_cmap("tab10").colors)
    if palette_name == "colorblind (cbf)":
        # Palette adaptée daltoniens (type Okabe–Ito approx.)
        return ["#0072B2", "#E69F00", "#009E73", "#D55E00",
                "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
    if palette_name == "viridis":
        cmap = plt.get_cmap("viridis")
        return [cmap(x) for x in np.linspace(0.05, 0.95, 10)]
    if palette_name == "plasma":
        cmap = plt.get_cmap("plasma")
        return [cmap(x) for x in np.linspace(0.05, 0.95, 10)]
    if palette_name == "deep (seaborn-like)":
        # Proche de seaborn deep
        return ["#4C72B0", "#55A868", "#C44E52", "#8172B3",
                "#CCB974", "#64B5CD", "#937860", "#DA8BC3"]
    # Fallback
    return plt.rcParamsDefault["axes.prop_cycle"].by_key().get("color", ["#1f77b4", "#ff7f0e", "#2ca02c",
                                                                        "#d62728", "#9467bd", "#8c564b",
                                                                        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"])


# ------------------------------------------------------------
# INTERFACE PRINCIPALE
# ------------------------------------------------------------

st.title("Multi-Traces")

uploaded_file = st.file_uploader("Sélectionner un fichier Excel", type=["xlsx"])

# ------------------------------------------------------------
# SIDEBAR : OPTIONS D’AFFICHAGE
# ------------------------------------------------------------
with st.sidebar.expander("⚙️ Options d’affichage"):
    wind = st.checkbox("Afficher vitesse du vent", True)
    kmh = st.checkbox("Afficher vent en km/h ?", True)
    direction = st.checkbox("Afficher direction du vent", True)
    dirlabel = st.checkbox("Afficher étiquettes direction", False)
    celcius = st.checkbox("Afficher Température", True)
    HR = st.checkbox("Afficher Humidité relative", True)

# 🔧 Choix de la palette
with st.sidebar.expander("🎨 Couleurs"):
    palette_choice = st.selectbox(
        "Palette de couleurs",
        [
            "Matplotlib (défaut)",
            "Tableau 10",
            "Colorblind (CBF)",
            "Viridis",
            "Plasma",
            "Deep (seaborn-like)",
        ],
        index=0
    )

# ------------------------------------------------------------
# SI FICHIER CHARGÉ
# ------------------------------------------------------------
if uploaded_file:

    df = pd.read_excel(uploaded_file)
    df["Start Time"] = pd.to_datetime(df["Start Time"], errors="coerce")
    df = df.dropna(subset=["Start Time"])

    # Vecteurs de vent
    results_df = compute_wind_vectors(df)

    # VALEURS AUTOMATIQUES POUR INFO
    laeq_min_auto = float(df["LAeq"].min())
    laeq_max_auto = float(df["LAeq"].max())

    wind_series = df["Wind Speed avg"] * 3.6 if kmh else df["Wind Speed avg"]
    wind_min_auto = float(wind_series.min())
    wind_max_auto = float(wind_series.max())

    hr_min_auto = float(df["Amb. Humidity"].min())
    hr_max_auto = float(df["Amb. Humidity"].max())

    temp_min_auto = float(df["Amb. Temperature"].min())
    temp_max_auto = float(df["Amb. Temperature"].max())

    # ------------------------------------------------------------
    # PERIODE D’AFFICHAGE (option B = limite affichage seulement)
    # ------------------------------------------------------------
    with st.sidebar.expander("🕒 Période d’affichage"):
        debut_global = df["Start Time"].min()
        fin_global = df["Start Time"].max()

        reset_time = st.button("🔄 Réinitialiser période d'affichage")

        if reset_time:
            date_debut = debut_global
            date_fin = fin_global
        else:
            date_debut = st.datetime_input(
                "Date-heure début",
                value=debut_global,
                min_value=debut_global,
                max_value=fin_global
            )

            date_fin = st.datetime_input(
                "Date-heure fin",
                value=fin_global,
                min_value=debut_global,
                max_value=fin_global
            )

        if date_debut >= date_fin:
            st.warning("⚠️ La date de début doit être antérieure à la date de fin.")
            date_debut = debut_global
            date_fin = fin_global

    # ------------------------------------------------------------
    # TITRE PERSONNALISÉ
    # ------------------------------------------------------------
    default_title = f"Données mesurées de {date_debut} à {date_fin}"

    with st.sidebar.expander("📝 Titre du graphique"):
        titre_graphique = st.text_input(
            "Titre du graphique",
            value=default_title
        )

    # ------------------------------------------------------------
    # CONTROLE DES ECHELLES
    # ------------------------------------------------------------
    with st.sidebar.expander("📏 Contrôle manuel des échelles"):
        # Valeurs fixes par défaut
        DEFAULT_LAEQ_MIN = 30
        DEFAULT_LAEQ_MAX = 70
        DEFAULT_WIND_MIN = 0
        DEFAULT_WIND_MAX = 90
        DEFAULT_HR_MIN = 0
        DEFAULT_HR_MAX = 100
        DEFAULT_TEMP_MIN = -10
        DEFAULT_TEMP_MAX = 35

        reset_scales = st.button("🔄 Réinitialiser échelles")

        if reset_scales:
            laeq_min = DEFAULT_LAEQ_MIN
            laeq_max = DEFAULT_LAEQ_MAX
            wind_min = DEFAULT_WIND_MIN
            wind_max = DEFAULT_WIND_MAX
            hr_min = DEFAULT_HR_MIN
            hr_max = DEFAULT_HR_MAX
            temp_min = DEFAULT_TEMP_MIN
            temp_max = DEFAULT_TEMP_MAX
        else:
            laeq_min = DEFAULT_LAEQ_MIN
            laeq_max = DEFAULT_LAEQ_MAX
            wind_min = DEFAULT_WIND_MIN
            wind_max = DEFAULT_WIND_MAX
            hr_min = DEFAULT_HR_MIN
            hr_max = DEFAULT_HR_MAX
            temp_min = DEFAULT_TEMP_MIN
            temp_max = DEFAULT_TEMP_MAX

        # LAeq
        st.markdown(f"### LAeq (données : {laeq_min_auto:.1f} → {laeq_max_auto:.1f})")
        laeq_min = st.number_input("LAeq Min", value=float(laeq_min))
        laeq_max = st.number_input("LAeq Max", value=float(laeq_max))

        # Vent
        st.markdown(f"### Vent ({'km/h' if kmh else 'm/s'}) – données : {wind_min_auto:.1f} → {wind_max_auto:.1f}")
        wind_min = st.number_input("Vent Min", value=float(wind_min))
        wind_max = st.number_input("Vent Max", value=float(wind_max))

        # HR
        st.markdown(f"### HR – données : {hr_min_auto:.1f} → {hr_max_auto:.1f}")
        hr_min = st.number_input("HR Min", value=float(hr_min))
        hr_max = st.number_input("HR Max", value=float(hr_max))

        # Température
        st.markdown(f"### Température – données : {temp_min_auto:.1f} → {temp_max_auto:.1f}")
        temp_min = st.number_input("Température Min", value=float(temp_min))
        temp_max = st.number_input("Température Max", value=float(temp_max))

        def validate(name, vmin, vmax, default_min, default_max):
            if vmin >= vmax:
                st.warning(f"{name}: min ≥ max → valeurs par défaut restaurées.")
                return default_min, default_max
            return vmin, vmax

        laeq_min, laeq_max = validate("LAeq", laeq_min, laeq_max, DEFAULT_LAEQ_MIN, DEFAULT_LAEQ_MAX)
        wind_min, wind_max = validate("Vent", wind_min, wind_max, DEFAULT_WIND_MIN, DEFAULT_WIND_MAX)
        hr_min, hr_max = validate("HR", hr_min, hr_max, DEFAULT_HR_MIN, DEFAULT_HR_MAX)
        temp_min, temp_max = validate("Température", temp_min, temp_max, DEFAULT_TEMP_MIN, DEFAULT_TEMP_MAX)

    # ------------------------------------------------------------
    # APPLIQUER LA PALETTE 🔧
    # ------------------------------------------------------------
    colors = get_color_cycle(palette_choice)
    plt.rcParams["axes.prop_cycle"] = cycler(color=colors)

    # ------------------------------------------------------------
    # GRAPHIQUE
    # ------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(18, 10))
    ax1.grid(True)

    # LAeq (C0)
    ax1.plot(df["Start Time"], df["LAeq"], color="C0")
    ax1.set_ylabel("LAeq", color="C0")
    ax1.tick_params(axis="x", rotation=55)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
    ax1.set_ylim(laeq_min, laeq_max)
    ax1.set_title(titre_graphique)
    ax1.set_xlim(date_debut, date_fin)

    # Vent (C1)
    if wind:
        ax2 = ax1.twinx()
        if kmh:
            ax2.plot(df["Start Time"], df["Wind Speed avg"] * 3.6, color="C1")
            ax2.set_ylabel("Vent vitesse (km/h)", color="C1")
        else:
            ax2.plot(df["Start Time"], df["Wind Speed avg"], color="C1")
            ax2.set_ylabel("Vent vitesse (m/s)", color="C1")
        ax2.set_ylim(wind_min, wind_max)

    # HR (C2)
    if HR:
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 40))
        ax3.plot(df["Start Time"], df["Amb. Humidity"], color="C2")
        ax3.set_ylabel("%HR", color="C2")
        ax3.set_ylim(hr_min, hr_max)

    # Température (C3)
    if celcius:
        ax4 = ax1.twinx()
        ax4.spines["right"].set_position(("outward", 100))
        ax4.plot(df["Start Time"], df["Amb. Temperature"], color="C3")
        ax4.set_ylabel("Température (°C)", color="C3")
        ax4.set_ylim(temp_min, temp_max)

    # Direction du vent
    if direction:
        results_df["row"] = results_df.index
        ax_top = ax1.twiny()

        wind_rad = np.radians(results_df["MeanWindDirection"])

        y_arrow = laeq_max - (laeq_max - laeq_min) * 0.05

        ax_top.quiver(
            results_df["row"],
            y_arrow,
            np.cos(wind_rad),
            np.sin(-wind_rad),
            scale_units="xy",
            scale=1,
            width=0.003
        )

        if dirlabel:
            y_label = y_arrow - (laeq_max - laeq_min) * 0.03
            for _, row in results_df.iterrows():
                ax_top.text(
                    row["row"],
                    y_label,
                    f'{row["MeanWindDirection"]+270:.1f}\n({row["SigmaTheta"]:.1f})',
                    color="red",
                    ha="center",
                    fontsize=8
                )

    st.pyplot(fig)

    # ------------------------------------------------------------
    # Téléchargements PNG & SVG 🔧
    # ------------------------------------------------------------
    from zoneinfo import ZoneInfo

    # PNG
    png_buffer = BytesIO()
    fig.savefig(png_buffer, format="png", dpi=300, bbox_inches="tight")
    # SVG
    svg_buffer = BytesIO()
    fig.savefig(svg_buffer, format="svg", bbox_inches="tight")

    # Horodatage (Québec)
    now_local = datetime.now(ZoneInfo("America/Toronto"))
    timestamp = now_local.strftime('%Y-%m-%d_%Hh%Mm%Ss')

    st.sidebar.download_button(
        label="📥 Télécharger l’image (.png)",
        data=png_buffer.getvalue(),
        file_name=f"traces_{timestamp}.png",
        mime="image/png"
    )
    st.sidebar.download_button(
        label="📥 Télécharger l’image (.svg)",
        data=svg_buffer.getvalue(),
        file_name=f"traces_{timestamp}.svg",
        mime="image/svg+xml"
    )
