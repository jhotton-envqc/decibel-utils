import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cycler
from io import BytesIO
from datetime import datetime
from zoneinfo import ZoneInfo

# st.set_page_config(page_title="Multi-Trace", layout="wide")

# ------------------------------------------------------------
# FONCTIONS D'ORIGINE
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


# ------------------------------------------------------------
# PALETTES
# ------------------------------------------------------------
def get_color_cycle(palette_name: str):
    palette_name = palette_name.lower()
    if palette_name == "matplotlib (défaut)":
        return plt.rcParamsDefault["axes.prop_cycle"].by_key().get("color", [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ])
    if palette_name == "tableau 10":
        return list(plt.get_cmap("tab10").colors)
    if palette_name == "colorblind (cbf)":
        return ["#0072B2", "#E69F00", "#009E73", "#D55E00",
                "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
    if palette_name == "viridis":
        cmap = plt.get_cmap("viridis")
        return [cmap(x) for x in np.linspace(0.05, 0.95, 10)]
    if palette_name == "plasma":
        cmap = plt.get_cmap("plasma")
        return [cmap(x) for x in np.linspace(0.05, 0.95, 10)]
    if palette_name == "deep (seaborn-like)":
        return ["#4C72B0", "#55A868", "#C44E52", "#8172B3",
                "#CCB974", "#64B5CD", "#937860", "#DA8BC3"]
    return plt.rcParamsDefault["axes.prop_cycle"].by_key().get("color", [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ])


# ------------------------------------------------------------
# OUTILS TEMPS
# ------------------------------------------------------------
def ceil_to_hour(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.to_datetime(ts).ceil("H")

def make_edges_1h(anchor_start: pd.Timestamp, last_time: pd.Timestamp) -> pd.DatetimeIndex:
    """
    Crée les bords d'intervalles 1h : [anchor, anchor+1h, ...] jusqu'à > last_time.
    """
    edges = []
    cur = pd.to_datetime(anchor_start)
    if cur > last_time:
        return pd.DatetimeIndex([])
    while cur <= last_time:
        edges.append(cur)
        cur = cur + pd.Timedelta(hours=1)
    edges.append(cur)  # dernier bord pour fermer
    return pd.DatetimeIndex(edges)


# ------------------------------------------------------------
# CALCUL STRICT LAeq_1h (équation fournie) — VERSION ROBUSTE
# ------------------------------------------------------------
def laeq_1h_blocks_strict(times: pd.Series,
                          levels_db: pd.Series,
                          anchor: pd.Timestamp,
                          T_seconds: float = 3600.0) -> pd.DataFrame:
    """
    LAeq,T = 10*log10( (1/T) * sum_i ( t_i * 10^(L_i/10) ) )
    - Périodes d'1h non chevauchées, à partir de 'anchor'
    - t_i = chevauchement (s) entre [t_i, t_{i+1}) et la période d'1 h (clipping)
    - T = 3600 s (constant)
    Retourne DataFrame: left, right, center, laeq, coverage_s
    """
    # Données propres
    d = pd.DataFrame({
        "t": pd.to_datetime(times, errors="coerce"),
        "L": pd.to_numeric(levels_db, errors="coerce")
    }).dropna().sort_values("t")

    if len(d) < 2:
        return pd.DataFrame(columns=["left", "right", "center", "laeq", "coverage_s"])

    # Intervalles [t_i, t_{i+1}) & niveau L_i pris au début
    ts = np.asarray(d["t"].values, dtype="datetime64[ns]")        # np.ndarray
    t_start = ts[:-1]
    t_end   = ts[1:]
    L_i     = np.asarray(d["L"].values, dtype=float)[:-1]

    # Garder intervalles valides
    keep = t_end > t_start
    t_start = t_start[keep]
    t_end   = t_end[keep]
    L_i     = L_i[keep]

    if t_start.size == 0:
        return pd.DataFrame(columns=["left", "right", "center", "laeq", "coverage_s"])

    data_last = pd.Timestamp(t_end[-1])
    edges = make_edges_1h(anchor, data_last)
    if len(edges) < 2:
        return pd.DataFrame(columns=["left", "right", "center", "laeq", "coverage_s"])

    n_bins = len(edges) - 1
    energy_sum = np.zeros(n_bins, dtype=float)
    coverage_s = np.zeros(n_bins, dtype=float)

    # Convertir tout en ns (int64)
    ts_ns    = t_start.astype("datetime64[ns]").astype("int64")
    te_ns    = t_end.astype("datetime64[ns]").astype("int64")
    edges_ns = np.asarray(edges.values, dtype="datetime64[ns]").astype("int64")
    lin      = np.power(10.0, L_i / 10.0)

    for k in range(n_bins):
        a_ns, b_ns = edges_ns[k], edges_ns[k + 1]
        start_ns   = np.maximum(ts_ns, a_ns)
        end_ns     = np.minimum(te_ns, b_ns)
        overlap_ns = np.maximum(0, end_ns - start_ns)
        overlap_s  = overlap_ns / 1e9
        if np.any(overlap_s > 0):
            energy_sum[k] = float(np.sum(overlap_s * lin))
            coverage_s[k] = float(np.sum(overlap_s))

    laeq_vals = np.full(n_bins, np.nan, dtype=float)
    mask_nz = energy_sum > 0
    laeq_vals[mask_nz] = 10.0 * np.log10(energy_sum[mask_nz] / T_seconds)

    return pd.DataFrame({
        "left":   edges[:-1],
        "right":  edges[1:],
        "center": edges[:-1] + pd.Timedelta(minutes=30),
        "laeq":   laeq_vals,
        "coverage_s": coverage_s
    })


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("Multi-Traces")

uploaded_file = st.file_uploader("Sélectionner un fichier Excel", type=["xlsx"])

with st.sidebar.expander("⚙️ Options d’affichage"):
    wind = st.checkbox("Afficher vitesse du vent", True)
    kmh = st.checkbox("Afficher vent en km/h ?", True)
    direction = st.checkbox("Afficher direction du vent", True)
    dirlabel = st.checkbox("Afficher étiquettes direction", False)
    celcius = st.checkbox("Afficher Température", True)
    HR = st.checkbox("Afficher Humidité relative", True)

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
    transparent_bg = st.checkbox("Fond transparent", value=False)

# Bloc LAeq 1h + options d’annotation
with st.sidebar.expander("📊 LAeq 1h (équation stricte)"):
    show_laeq1h = st.checkbox("Afficher LAeq 1h", value=False)
    laeq1h_mode = st.radio(
        "Mode (périodes 1h non chevauchées)",
        ["Instantané", "Synchronisé", "Personnalisé"],
        index=0
    ) if show_laeq1h else None
    require_full = st.checkbox(
        "N’afficher que les périodes complètes (60 min)",
        value=True,
        help="Si décoché, les périodes partielles sont calculées avec T=3600 s et affichées."
    ) if show_laeq1h else None

    # 🆕 Options d’annotation au-dessus des segments
    show_labels = st.checkbox(
        "Afficher la valeur au-dessus de chaque segment",
        value=False
    ) if show_laeq1h else False

    if show_labels:
        decimals = st.selectbox("Décimales", [0, 1, 2], index=1)
        label_offset = st.number_input("Décalage vertical (dB)", value=0.5, step=0.1, format="%.1f")
        label_fontsize = st.slider("Taille du texte", min_value=6, max_value=20, value=10, step=1)
        label_color = st.color_picker("Couleur du texte", value="#222222")
        label_box = st.checkbox("Fond blanc derrière l’étiquette", value=not transparent_bg)

# ------------------------------------------------------------
# LOGIQUE PRINCIPALE
# ------------------------------------------------------------
if uploaded_file:

    # Lecture & préparation
    df = pd.read_excel(uploaded_file)
    df["Start Time"] = pd.to_datetime(df["Start Time"], errors="coerce")
    df = df.dropna(subset=["Start Time"]).sort_values("Start Time")

    # Vecteurs de vent pour les flèches
    results_df = compute_wind_vectors(df)

    # Stats auto
    laeq_min_auto = float(df["LAeq"].min())
    laeq_max_auto = float(df["LAeq"].max())

    wind_series = df["Wind Speed avg"] * 3.6 if kmh else df["Wind Speed avg"]
    wind_min_auto = float(wind_series.min()); wind_max_auto = float(wind_series.max())
    hr_min_auto = float(df["Amb. Humidity"].min()); hr_max_auto = float(df["Amb. Humidity"].max())
    temp_min_auto = float(df["Amb. Temperature"].min()); temp_max_auto = float(df["Amb. Temperature"].max())

    # Période d’affichage
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
                value=debut_global, min_value=debut_global, max_value=fin_global
            )
            date_fin = st.datetime_input(
                "Date-heure fin",
                value=fin_global, min_value=debut_global, max_value=fin_global
            )

        if date_debut >= date_fin:
            st.warning("⚠️ La date de début doit être antérieure à la date de fin.")
            date_debut, date_fin = debut_global, fin_global

    # Début personnalisé LAeq 1h si requis
    custom_anchor = None
    if show_laeq1h and laeq1h_mode == "Personnalisé":
        with st.sidebar.expander("⚙️ Paramètres LAeq 1h"):
            custom_anchor = st.datetime_input(
                "Début personnalisé des périodes 1h",
                value=debut_global, min_value=debut_global, max_value=fin_global
            )

    # Titre
    default_title = f"Données mesurées de {date_debut} à {date_fin}"
    with st.sidebar.expander("📝 Titre du graphique"):
        titre_graphique = st.text_input("Titre du graphique", value=default_title)

    # Échelles
    with st.sidebar.expander("📏 Contrôle manuel des échelles"):
        DEFAULT_LAEQ_MIN, DEFAULT_LAEQ_MAX = 30, 70
        DEFAULT_WIND_MIN, DEFAULT_WIND_MAX = 0, 90
        DEFAULT_HR_MIN, DEFAULT_HR_MAX = 0, 100
        DEFAULT_TEMP_MIN, DEFAULT_TEMP_MAX = -10, 35

        # Valeurs initiales
        laeq_min = DEFAULT_LAEQ_MIN; laeq_max = DEFAULT_LAEQ_MAX
        wind_min = DEFAULT_WIND_MIN; wind_max = DEFAULT_WIND_MAX
        hr_min = DEFAULT_HR_MIN; hr_max = DEFAULT_HR_MAX
        temp_min = DEFAULT_TEMP_MIN; temp_max = DEFAULT_TEMP_MAX

        st.markdown(f"### LAeq (données : {laeq_min_auto:.1f} → {laeq_max_auto:.1f})")
        laeq_min = st.number_input("LAeq Min", value=float(laeq_min))
        laeq_max = st.number_input("LAeq Max", value=float(laeq_max))

        st.markdown(f"### Vent ({'km/h' if kmh else 'm/s'}) – données : {wind_min_auto:.1f} → {wind_max_auto:.1f}")
        wind_min = st.number_input("Vent Min", value=float(wind_min))
        wind_max = st.number_input("Vent Max", value=float(wind_max))

        st.markdown(f"### HR – données : {hr_min_auto:.1f} → {hr_max_auto:.1f}")
        hr_min = st.number_input("HR Min", value=float(hr_min))
        hr_max = st.number_input("HR Max", value=float(hr_max))

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
    # APPLIQUER LA PALETTE
    # ------------------------------------------------------------
    colors = get_color_cycle(palette_choice)
    plt.rcParams["axes.prop_cycle"] = cycler(color=colors)

    # ------------------------------------------------------------
    # GRAPHIQUE
    # ------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(18, 10))
    ax1.grid(True)

    if transparent_bg:
        fig.patch.set_alpha(0.0)
        ax1.set_facecolor("none")

    # LAeq (C0)
    ax1.plot(df["Start Time"], df["LAeq"], color="C0", label="LAeq")
    ax1.set_ylabel("LAeq", color="C0")
    ax1.tick_params(axis="x", rotation=55)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
    ax1.set_ylim(laeq_min, laeq_max)
    ax1.set_title(titre_graphique)
    ax1.set_xlim(date_debut, date_fin)

    # Vent (C1)
    ax2 = None
    if wind:
        ax2 = ax1.twinx()
        if kmh:
            ax2.plot(df["Start Time"], df["Wind Speed avg"] * 3.6, color="C1", label="Vent")
            ax2.set_ylabel("Vent vitesse (km/h)", color="C1")
        else:
            ax2.plot(df["Start Time"], df["Wind Speed avg"], color="C1", label="Vent")
            ax2.set_ylabel("Vent vitesse (m/s)", color="C1")
        ax2.set_ylim(wind_min, wind_max)

    # HR (C2)
    ax3 = None
    if HR:
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 40))
        ax3.plot(df["Start Time"], df["Amb. Humidity"], color="C2", label="HR")
        ax3.set_ylabel("%HR", color="C2")
        ax3.set_ylim(hr_min, hr_max)

    # Température (C3)
    ax4 = None
    if celcius:
        ax4 = ax1.twinx()
        ax4.spines["right"].set_position(("outward", 100))
        ax4.plot(df["Start Time"], df["Amb. Temperature"], color="C3", label="Temp.")
        ax4.set_ylabel("Température (°C)", color="C3")
        ax4.set_ylim(temp_min, temp_max)

    # Direction du vent (axes supérieurs)
    ax_top = None
    if direction:
        results_df["row"] = results_df.index
        ax_top = ax1.twiny()
        wind_rad = np.radians(results_df["MeanWindDirection"])
        y_arrow = laeq_max - (laeq_max - laeq_min) * 0.05
        ax_top.quiver(
            results_df["row"], y_arrow,
            np.cos(wind_rad), np.sin(-wind_rad),
            scale_units="xy", scale=1, width=0.003
        )
        if dirlabel:
            y_label = y_arrow - (laeq_max - laeq_min) * 0.03
            for _, row in results_df.iterrows():
                ax_top.text(
                    row["row"], y_label,
                    f'{row["MeanWindDirection"]+270:.1f}\n({row["SigmaTheta"]:.1f})',
                    color="red", ha="center", fontsize=8
                )

    # ------------------------------------------------------------
    # CALCUL & TRACÉ LAeq 1h (paliers horizontaux + étiquettes)
    # ------------------------------------------------------------
    la1h_all = pd.DataFrame(columns=["left", "right", "center", "laeq", "coverage_s"])
    if show_laeq1h and laeq1h_mode:
        t0 = df["Start Time"].min()

        if laeq1h_mode == "Instantané":
            anchor = t0
        elif laeq1h_mode == "Synchronisé":
            anchor = ceil_to_hour(t0)  # première heure pile >= t0
        else:  # Personnalisé
            anchor = custom_anchor if custom_anchor is not None else t0

        # Calcul strict par bins 1h (T = 3600 s)
        la1h_all = laeq_1h_blocks_strict(
            df["Start Time"], df["LAeq"], anchor=anchor, T_seconds=3600.0
        )

        # Filtre optionnel : n’afficher que les périodes complètes
        if require_full:
            la1h_plot = la1h_all[la1h_all["coverage_s"] >= 3600.0 - 1e-6].copy()
        else:
            la1h_plot = la1h_all.copy()

        # Tracé limité à la fenêtre affichée
        color_la1h = "C4"; lw = 2.4
        for _, r in la1h_plot.iterrows():
            L, R, v = r["left"], r["right"], r["laeq"]
            if not np.isfinite(v):
                continue
            if (R <= date_debut) or (L >= date_fin):
                continue  # hors affichage
            Lp = max(L, date_debut)
            Rp = min(R, date_fin)
            # Segment LAeq 1h
            ax1.plot([Lp, Rp], [v, v], color=color_la1h, linewidth=lw, solid_capstyle="butt")

            # 🆕 ÉTIQUETTE AU-DESSUS DU SEGMENT
            if show_labels:
                x_mid = L + (R - L) / 2  # centre temporel du bin complet (plus stable)
                # Si tu préfères le centre tronqué à l’affichage : x_mid = Lp + (Rp - Lp) / 2
                fmt = f"{{:.{decimals}f}}"
                text_val = fmt.format(v)
                bbox = dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2) if label_box else None
                ax1.text(
                    x_mid, v + label_offset,
                    text_val,
                    ha="center", va="bottom",
                    fontsize=label_fontsize,
                    color=label_color,
                    bbox=bbox,
                    clip_on=True  # évite que ça dépasse la zone tracée
                )

        # Légende
        ax1.plot([], [], color=color_la1h, linewidth=lw, label="LAeq 1h")

    ax1.legend(loc="upper left")

    # Transparence pour tous les axes
    if transparent_bg:
        for ax in [a for a in [ax1, ax2, ax3, ax4, ax_top] if a is not None]:
            ax.set_facecolor("none")

    # Afficher le graphique
    st.pyplot(fig)

    # ------------------------------------------------------------
    # TABLEAU & EXPORTS CSV DES LAeq 1h
    # ------------------------------------------------------------
    if show_laeq1h and not la1h_all.empty:
        st.markdown("### 📋 LAeq 1h calculés (périodes visibles)")

        # Périodes qui chevauchent la fenêtre affichée
        visible_mask = ~((la1h_all["right"] <= date_debut) | (la1h_all["left"] >= date_fin))
        la1h_visible = la1h_all[visible_mask].copy()

        # Appliquer le filtre "périodes complètes" pour rester cohérent avec le tracé
        if require_full:
            la1h_visible = la1h_visible[la1h_visible["coverage_s"] >= 3600.0 - 1e-6]

        # Mise en forme
        la1h_visible_fmt = la1h_visible.assign(
            LAeq_1h=lambda d: d["laeq"].round(1),
            Couverture_s=lambda d: d["coverage_s"].round(0).astype(int),
            Début=lambda d: d["left"],
            Fin=lambda d: d["right"],
            Centre=lambda d: d["center"]
        )[["Début", "Fin", "Centre", "LAeq_1h", "Couverture_s"]].reset_index(drop=True)

        st.dataframe(la1h_visible_fmt, use_container_width=True, hide_index=True)

        # Exports CSV : visible et complet (après filtre require_full)
        timestamp = datetime.now(ZoneInfo("America/Toronto")).strftime('%Y-%m-%d_%Hh%Mm%Ss')

        csv_visible = la1h_visible_fmt.to_csv(index=False).encode("utf-8")

        la1h_full = la1h_all.copy()
        if require_full:
            la1h_full = la1h_full[la1h_full["coverage_s"] >= 3600.0 - 1e-6]
        la1h_full_fmt = la1h_full.assign(
            LAeq_1h=lambda d: d["laeq"].round(1),
            Couverture_s=lambda d: d["coverage_s"].round(0).astype(int),
            Début=lambda d: d["left"],
            Fin=lambda d: d["right"],
            Centre=lambda d: d["center"]
        )[["Début", "Fin", "Centre", "LAeq_1h", "Couverture_s"]].reset_index(drop=True)

        csv_full = la1h_full_fmt.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="💾 Télécharger CSV — périodes visibles",
            data=csv_visible,
            file_name=f"LAeq1h_visibles_{timestamp}.csv",
            mime="text/csv"
        )
        st.download_button(
            label="💾 Télécharger CSV — toutes les périodes",
            data=csv_full,
            file_name=f"LAeq1h_complet_{timestamp}.csv",
            mime="text/csv"
        )
    else:
        if show_laeq1h:
            st.info("Aucune période LAeq 1h à afficher pour la fenêtre courante (ou données insuffisantes).")

    # ------------------------------------------------------------
    # TÉLÉCHARGEMENTS PNG & SVG
    # ------------------------------------------------------------
    png_buffer = BytesIO()
    fig.savefig(png_buffer, format="png", dpi=300, bbox_inches="tight", transparent=transparent_bg)

    svg_buffer = BytesIO()
    fig.savefig(svg_buffer, format="svg", bbox_inches="tight", transparent=transparent_bg)

    timestamp = datetime.now(ZoneInfo("America/Toronto")).strftime('%Y-%m-%d_%Hh%Mm%Ss')

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