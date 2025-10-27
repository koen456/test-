# ======================================================
# DASHBOARD: Laadpalen & Elektrische Voertuigen
# ======================================================

# ------------------- Imports --------------------------
# ------------------------------------------------------
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import folium
import requests
import re
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster, FastMarkerCluster
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import warnings
import pickle
import io
import plotly.graph_objects as go

# ------------------- Sidebar ---------------------------
# ------------------------------------------------------
with st.sidebar:
    st.markdown("## Laadpalen & Elektrische Voertuigen")
    st.markdown("---")

    page = st.selectbox(
        "Selecteer een pagina",
        [
            "⚡️ Laadpalen",
            "🚘 Voertuigen",
            "📊 Voorspellend model"
        ]
    )

    st.write("")
    st.info("🔋 Data afkomstig van OpenChargeMap & RDW")
    st.markdown("---")
    st.write("Voor het laatst geüpdatet op:")
    st.write("*09 okt 2025*")


# ------------------- Data inladen -----------------------
# -------------------------------------------------------
@st.cache_data
def load_data():
    df_auto = pd.read_csv("duitse_automerken_JA.csv")
    return df_auto

@st.cache_data(ttl=86400)
def get_laadpalen_data(lat: float, lon: float, radius: float) -> pd.DataFrame:
    """Haalt laadpalen binnen een straal op."""
    url = "https://api.openchargemap.io/v3/poi/"
    params = {
        "output": "json",
        "countrycode": "NL",
        "latitude": lat,
        "longitude": lon,
        "distance": radius,
        "maxresults": 5000,
        "compact": True,
        "verbose": False,
        "key": "bbc1c977-6228-42fc-b6af-5e5f71be11a5"
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    df = pd.json_normalize(data)
    df = df.dropna(subset=['AddressInfo.Latitude', 'AddressInfo.Longitude'])
    return df

@st.cache_data(ttl=86400)
def get_all_laadpalen_nederland() -> pd.DataFrame:
    """Haalt laadpalen van heel Nederland op (voor grafieken)."""
    url = "https://api.openchargemap.io/v3/poi/"
    params = {
        "output": "json",
        "countrycode": "NL",
        "maxresults": 10000,
        "compact": True,
        "verbose": False,
        "key": "bbc1c977-6228-42fc-b6af-5e5f71be11a5"
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    df = pd.json_normalize(data)
    return df

df_auto = load_data()


# ======================================================
#                   PAGINA-INDELING
# ======================================================

# ------------------- Pagina 1 --------------------------
if page == "⚡️ Laadpalen":
    st.markdown("## Kaart Laadpalen Nederland")
    st.markdown("---")

    provincies = {
        "Heel Nederland": [52.1, 5.3, 200],
        "Groningen": [53.2194, 6.5665, 60],
        "Friesland": [53.1642, 5.7818, 60],
        "Drenthe": [52.9476, 6.6231, 60],
        "Overijssel": [52.4380, 6.5010, 60],
        "Flevoland": [52.5270, 5.5953, 60],
        "Gelderland": [52.0452, 5.8712, 60],
        "Utrecht": [52.0907, 5.1214, 60],
        "Noord-Holland": [52.5206, 4.7885, 60],
        "Zuid-Holland": [52.0116, 4.3571, 60],
        "Zeeland": [51.4940, 3.8497, 60],
        "Noord-Brabant": [51.5730, 5.0670, 60],
        "Limburg": [51.2490, 5.9330, 60],
    }

    provincie_keuze = st.selectbox("📍 Kies een provincie", provincies.keys(), index=0)
    center_lat, center_lon, radius_km = provincies[provincie_keuze]

    with st.spinner(f" Laad laadpalen voor {provincie_keuze}..."):
        df = get_laadpalen_data(center_lat, center_lon, radius_km)
        df_all = get_all_laadpalen_nederland()

        if provincie_keuze != "Heel Nederland":
            Laadpalen = df[df["AddressInfo.StateOrProvince"].str.contains(provincie_keuze, case=False, na=False)]
        else:
            Laadpalen = df

    MAX_DEFAULT = 300  
    st.write(f"Provincie: **{provincie_keuze}**")
    laad_alle = st.checkbox("Laad alle laadpalen (geen popups)", value=False)

    if len(Laadpalen) == 0:
        st.warning("Geen laadpalen gevonden voor deze locatie/provincie.")
        m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles="OpenStreetMap")
        st_folium(m, width=900, height=650)
    else:
        start_zoom = 8 if provincie_keuze == "Heel Nederland" else 10
        m = folium.Map(location=[center_lat, center_lon], zoom_start=start_zoom, tiles="OpenStreetMap")

        if laad_alle:
            coords = list(zip(Laadpalen["AddressInfo.Latitude"], Laadpalen["AddressInfo.Longitude"]))
            FastMarkerCluster(data=coords).add_to(m)
            st.info(f"Snelmodus: alle laadpalen geladen (geen popups).")
        else:
            subset_df = Laadpalen.sample(n=min(len(Laadpalen), MAX_DEFAULT), random_state=1).reset_index(drop=True)
            marker_cluster = MarkerCluster().add_to(m)
            for _, row in subset_df.iterrows():
                lat, lon = row["AddressInfo.Latitude"], row["AddressInfo.Longitude"]
                popup = f"""
                <b>{row.get('AddressInfo.Title', 'Onbekend')}</b><br>
                {row.get('AddressInfo.AddressLine1', '')}<br>
                {row.get('AddressInfo.Town', '')}<br>
                Kosten: {row.get('UsageCost', 'N/B')}<br>
                Vermogen: {row.get('PowerKW', 'N/B')} kW
                """
                icon = folium.Icon(color="green", icon="bolt", prefix="fa")
                folium.Marker(location=[lat, lon], popup=folium.Popup(popup, max_width=300), icon=icon).add_to(marker_cluster)

            st.success(f"{len(subset_df)} laadpalen met popups geladen.")
        st_folium(m, width=900, height=650, returned_objects=["center", "zoom"])

    st.markdown("<small>**Bron: openchargemap.org**</small>", unsafe_allow_html=True)
    #Grafiek verdeling laadpalen in nederland 
    st.markdown("---")
    st.markdown("## 📊 Verdeling laadpalen in Nederland")

    if len(df_all) > 0:
        def parse_cost(value):
            if isinstance(value, str):
                if "free" in value.lower() or "gratis" in value.lower():
                    return 0.0
                match = re.search(r"(\d+[\.,]?\d*)", value.replace(",", "."))
                return float(match.group(1)) if match else np.nan
            return np.nan

        df_all["UsageCostClean"] = df_all["UsageCost"].apply(parse_cost)

        df_all.loc[
            (df_all["UsageCostClean"] < 0) | (df_all["UsageCostClean"] > 2),
            "UsageCostClean"
        ] = np.nan

        if "PowerKW" in df_all.columns:
            df_all["PowerKW_clean"] = pd.to_numeric(df_all["PowerKW"], errors="coerce")
        elif "Connections.PowerKW" in df_all.columns:
            df_all["PowerKW_clean"] = pd.to_numeric(df_all["Connections.PowerKW"], errors="coerce")
        elif "Connections[0].PowerKW" in df_all.columns:
            df_all["PowerKW_clean"] = pd.to_numeric(df_all["Connections[0].PowerKW"], errors="coerce")
        else:
            df_all["PowerKW_clean"] = np.nan

        provincie_mapping = {
            "Groningen": "Groningen",
            "Friesland": "Friesland",
            "Fryslân": "Friesland",
            "Drenthe": "Drenthe",
            "Overijssel": "Overijssel",
            "Flevoland": "Flevoland",
            "Gelderland": "Gelderland",
            "Utrecht": "Utrecht",
            "Noord-Holland": "Noord-Holland",
            "North Holland": "Noord-Holland",
            "Zuid-Holland": "Zuid-Holland",
            "South Holland": "Zuid-Holland",
            "Zeeland": "Zeeland",
            "Noord-Brabant": "Noord-Brabant",
            "North Brabant": "Noord-Brabant",
            "Limburg": "Limburg"
        }

        df_all["Provincie"] = df_all["AddressInfo.StateOrProvince"].map(provincie_mapping)
        df_all = df_all[df_all["Provincie"].isin(list(provincies.keys()))]

        df_agg = (
            df_all.groupby("Provincie")
            .agg(
                Aantal_palen=("ID", "count"),
                Gemiddelde_kosten=("UsageCostClean", "mean"),
            )
            .reset_index()
        )

        totaal = df_agg["Aantal_palen"].sum()
        df_agg["Percentage"] = (df_agg["Aantal_palen"] / totaal) * 100
        df_agg = df_agg.sort_values("Percentage", ascending=False)

        keuze = st.selectbox(
            "📈 Kies welke verdeling je wilt zien:",
            ["Verdeling laadpalen per provincie (%)", "Gemiddelde kosten per provincie"]
        )

        if keuze == "Verdeling laadpalen per provincie (%)":
            fig = px.bar(
                df_agg,
                x="Provincie",
                y="Percentage",
                title="Verdeling laadpalen per provincie (%)",
                text=df_agg["Percentage"].apply(lambda x: f"{x:.1f}%")
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(yaxis_title="Percentage van totaal (%)")
        elif keuze == "Gemiddelde kosten per provincie":
            fig = px.bar(
                df_agg,
                x="Provincie",
                y="Gemiddelde_kosten",
                title="Gemiddelde kosten per provincie (€ per kWh)"
            )
            fig.update_layout(yaxis_title="€ per kWh")

        fig.update_layout(xaxis_title="Provincie", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Kon geen landelijke data laden voor de grafiek.")
# ------------------- Pagina 2 --------------------------
elif page == "🚘 Voertuigen":
    st.markdown("## Elektrische Voertuigen & laadtijden")
    st.markdown("---")

    #-----Grafiek Lieke------


    # --- Functie om brandstoftype te bepalen ---
    def bepaal_type(merk, uitvoering):
        u = str(uitvoering).upper()
        m = str(merk).upper()

        elektrische_prefixen = [
            "FA1FA1CZ", "3EER", "3EDF", "3EDE", "2EER", "2EDF", "2EDE",
            "E11", "0AW5", "QE2QE2G1", "QE1QE1G1", "HE1HE1G1", "FA1FA1MD"
        ]

        # Elektrisch
        if "BMW I" in m or "PORSCHE" in m or any(u.startswith(pref) for pref in elektrische_prefixen) or "EV" in u:
            return "Elektrisch"

        # Diesel
        if "DIESEL" in u or "TDI" in u or "CDI" in u or "DPE" in u or u.startswith("D"):
            return "Diesel"

        # Benzine (default)
        return "Benzine"


    # --- Data inladen ---
    data = pd.read_csv("duitse_automerken_JA.csv")

    # --- Merknamen normaliseren ---
    merk_mapping = {
        "VW": "VOLKSWAGEN",
        "FAW-VOLKSWAGEN": "VOLKSWAGEN",
        "VOLKSWAGEN/ZIMNY": "VOLKSWAGEN",
        "BMW I": "BMW",
        "FORD-CNG-TECHNIK": "FORD"
    }
    data["Merk"] = data["Merk"].str.upper().replace(merk_mapping)

    # --- Type bepalen ---
    data["Type"] = data.apply(lambda row: bepaal_type(row["Merk"], row["Uitvoering"]), axis=1)

    # --- Datumverwerking ---
    data["Datum eerste toelating"] = (
        data["Datum eerste toelating"].astype(str).str.split(".").str[0]
    )
    data["Datum eerste toelating"] = pd.to_datetime(
        data["Datum eerste toelating"], format="%Y%m%d", errors="coerce"
    )
    data = data.dropna(subset=["Datum eerste toelating"])
    data = data[data["Datum eerste toelating"].dt.year > 2010]
    data["Maand"] = data["Datum eerste toelating"].dt.to_period("M").dt.to_timestamp()

    # ---  Keuzemenu voor merken ---
    alle_merknamen = sorted(data["Merk"].unique())
    geselecteerde_merknamen = st.multiselect(
        "*Selecteer automerken om te tonen:*",
        options=alle_merknamen,
        default=[]  # begin met geen selectie
    )

    # ---  Als geen merken geselecteerd: waarschuwing + alle merken gebruiken ---
    if not geselecteerde_merknamen:
        st.warning("⚠️ Geen merken geselecteerd. Alle merken worden getoond!")
        geselecteerde_merknamen = alle_merknamen

    # Filter data op geselecteerde merken
    data = data[data["Merk"].isin(geselecteerde_merknamen)]

    # --- Aggregatie ---
    maand_aantal = data.groupby(["Maand", "Type"]).size().unstack(fill_value=0)
    cumulatief = maand_aantal.cumsum()

    # --- 📈 Titel + Grafiek ---
    st.subheader("Cumulatief aantal voertuigen per maand")
    st.line_chart(cumulatief)

   #-------------Grafiek Ann---------


    # ---- Bestand vast instellen ----
    file_path = "Charging_data.pkl"

    # ---- FUNCTIE: x-as als hele getallen zetten ----
    def force_integer_xaxis(fig):
        fig.update_xaxes(dtick=1)
        return fig

    # ---- DATA INLADEN ----
    try:
        ev_data = pd.read_pickle(file_path)
        ev_data.columns = (
            ev_data.columns.astype(str)
            .str.strip()
            .str.replace("\u200b", "", regex=False)
            .str.lower()
        )

        # ---- DATUMCONVERSIE EN KOLOMMEN TOEVOEGEN ----
        ev_data["start_time"] = pd.to_datetime(ev_data["start_time"], errors="coerce")
        ev_data["exit_time"] = pd.to_datetime(ev_data["exit_time"], errors="coerce")
        ev_data["hour"] = ev_data["start_time"].dt.hour
        ev_data["month"] = ev_data["start_time"].dt.to_period("M").astype(str)
        ev_data["year"] = ev_data["start_time"].dt.year
        ev_data = ev_data[ev_data["year"].notna()]
        ev_data["year"] = ev_data["year"].astype(int)
        ev_data["weekday"] = ev_data["start_time"].dt.day_name()

        energy_col = "energy_delivered [kwh]"

        # ---- WEEKDAGFILTER ----
        st.subheader("🔍 Filter op weekdagen")
        weekdays_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        selected_days = st.multiselect(
            "Selecteer één of meerdere weekdagen:",
            weekdays_order,
            default=weekdays_order,
        )
        ev_data = ev_data[ev_data["weekday"].isin(selected_days)]

        # ---- HEATMAP: Laadpatronen per dag en uur ----
        st.subheader("Laadpatronen per dag en uur")
        heatmap_data = ev_data.groupby(["weekday", "hour"]).size().reset_index(name="count")
        heatmap_data["weekday"] = pd.Categorical(heatmap_data["weekday"], categories=weekdays_order, ordered=True)

        fig_hm = px.density_heatmap(
            heatmap_data,
            x="hour",
            y="weekday",
            z="count",
            color_continuous_scale=[[0.0, "#317595"], [0.5, "#fcffbf"], [1.0, "#c2242f"]],
        )
        fig_hm = force_integer_xaxis(fig_hm)
        fig_hm.update_coloraxes(colorbar_title="Aantal sessies", colorbar_title_side="top")
        st.plotly_chart(fig_hm, use_container_width=True)

        # ---- FILTER OP AANTAL FASEN ----
        phase_options = ["Alle"] + [x for x in sorted(ev_data["n_phases"].dropna().unique()) if 0 <= x <= 6]
        phase_choice = st.selectbox("**Filter op aantal fasen**", phase_options)

        ev_filtered = ev_data.copy()
        if phase_choice != "Alle":
            ev_filtered = ev_filtered[ev_filtered["n_phases"] == phase_choice]

        # ---- GRAFIEK 1: Laadsessies per uur van de dag ----
        st.subheader("Laadsessies per uur van de dag")
        hourly_counts = ev_filtered.groupby("hour").size().reset_index(name="Aantal laadsessies")
        fig1 = px.bar(hourly_counts, x="hour", y="Aantal laadsessies")
        fig1 = force_integer_xaxis(fig1)
        st.plotly_chart(fig1, use_container_width=True)

        # ---- GRAFIEK 2: Totaal geladen energie per maand ----
        st.subheader("Totaal geladen energie per maand")
        energy_by_month = ev_filtered.groupby("month")[energy_col].sum().reset_index().sort_values("month")

        # Controleer aantal unieke maanden
        unique_months = energy_by_month["month"].nunique()

        fig2 = px.bar(energy_by_month, x="month", y=energy_col)
        fig2.update_xaxes(type="category", title_text="Maand")
        fig2.update_yaxes(title_text="Totaal geladen energie (kWh)")
        st.plotly_chart(fig2, use_container_width=True)

        # ---- GRAFIEK 3: Gemiddelde sessieduur per maand ----
        st.subheader("Gemiddelde sessieduur per maand (uren)")
        ev_filtered["session_duration"] = (ev_filtered["exit_time"] - ev_filtered["start_time"]).dt.total_seconds() / 3600
        avg_duration = (
            ev_filtered.groupby("month")["session_duration"].mean().reset_index().sort_values("month")
        )
        fig3 = px.line(avg_duration, x="month", y="session_duration", markers=True)
        fig3.update_xaxes(type="category", title_text="Maand")
        fig3.update_yaxes(title_text="Gemiddelde sessieduur (uren)")
        st.plotly_chart(fig3, use_container_width=True)

        # ---- GRAFIEK 4: Boxplot energie per sessie per maand ----
        st.subheader("Verdeling van geladen energie per sessie per maand")
        fig4 = px.box(ev_filtered, x="month", y=energy_col, points="all")
        fig4.update_xaxes(type="category", title_text="Maand")
        fig4.update_yaxes(title_text="Energie per sessie (kWh)")
        st.plotly_chart(fig4, use_container_width=True)

        # ---- DATA BEKIJKEN ----
        with st.expander("📊 Bekijk gebruikte data (Charging_data.pkl)"):
            st.dataframe(ev_filtered)

    except Exception as e:
        st.error(f"Er is een fout opgetreden bij het inlezen van `{file_path}`: {e}")


# ------------------- Pagina 3 --------------------------
elif page == "📊 Voorspellend model":
    st.markdown("## Voorspellend Model")
    st.markdown("---")
    st.subheader("Voorspelling auto's in Nederland per brandstofcategorie")

    warnings.filterwarnings("ignore")

    # ---------- Interactieve instellingen ----------
    eindjaar = st.slider("Voorspellen tot jaar", 2025, 2050, 2030)
    EINDDATUM = pd.Timestamp(f"{eindjaar}-12-01")

    # ---------- Kopie gebruiken ----------
    df_auto_kopie = df_auto.copy()

    # ---------- Type bepalen ----------
    def bepaal_type(merk, uitvoering):
        u = str(uitvoering).upper()
        m = str(merk).upper()
        if ("BMW I" in m or "PORSCHE" in m or
            u.startswith(("FA1FA1CZ","3EER","3EDF","3EDE","2EER","2EDF","2EDE",
                          "E11","0AW5","QE2QE2G1","QE1QE1G1","HE1HE1G1")) or
            "EV" in u or "FA1FA1MD" in u):
            return "Elektrisch"
        if "DIESEL" in u or "TDI" in u or "CDI" in u or "DPE" in u or u.startswith("D"):
            return "Diesel"
        return "Benzine"

    df_auto_kopie["Type"] = df_auto_kopie.apply(
        lambda r: bepaal_type(r.get("Merk",""), r.get("Uitvoering","")), axis=1
    )

    # ---------- Datums opschonen ----------
    df_auto_kopie["Datum eerste toelating"] = df_auto_kopie["Datum eerste toelating"].astype(str).str.split(".").str[0]
    df_auto_kopie["Datum eerste toelating"] = pd.to_datetime(
        df_auto_kopie["Datum eerste toelating"], format="%Y%m%d", errors="coerce"
    )

    # ---------- Filteren en groeperen ----------
    df_auto_kopie2 = df_auto_kopie.dropna(subset=["Datum eerste toelating"])
    df_auto_kopie2 = df_auto_kopie2[df_auto_kopie2["Datum eerste toelating"].dt.year > 2010]
    df_auto_kopie2["Maand"] = df_auto_kopie2["Datum eerste toelating"].dt.to_period("M").dt.to_timestamp()

    maand_counts_charging = df_auto_kopie2.groupby(["Maand", "Type"]).size().unstack(fill_value=0).sort_index()
    if maand_counts_charging.empty:
        st.error("⚠ Geen bruikbare data gevonden in dataset na 2010.")
        st.stop()

    # ---------- Historische cumulatieven ----------
    cumul_hist_charging = maand_counts_charging.cumsum()
    laatste_hist_maand = cumul_hist_charging.index.max()
    forecast_start = laatste_hist_maand + pd.DateOffset(months=1)
    if forecast_start > EINDDATUM:
        st.error("⚠ Het gekozen eindjaar ligt vóór de laatste beschikbare data. Kies een later jaar.")
        st.stop()

    forecast_index = pd.date_range(start=forecast_start, end=EINDDATUM, freq="MS")
    h = len(forecast_index)
    if h <= 0:
        st.error("⚠ Geen forecast-horizon (controleer eindjaar).")
        st.stop()

    # ---------- Voorspel totaal aantal maandelijkse registraties ----------
    totale_maand = maand_counts_charging.sum(axis=1).astype(float)

    # probeer SARIMAX op totaal
    try:
        if len(totale_maand) >= 24:
            model_tot = SARIMAX(totale_maand, order=(1,1,1), seasonal_order=(1,1,0,12),
                                enforce_stationarity=False, enforce_invertibility=False)
            fit_tot = model_tot.fit(disp=False)
            pred_tot = fit_tot.get_forecast(steps=h).predicted_mean.values
        else:

            x = np.arange(len(totale_maand))
            m_tot, b_tot = np.polyfit(x, totale_maand, 1)
            future_x = np.arange(len(totale_maand), len(totale_maand) + h)
            pred_tot = b_tot + m_tot * future_x
    except Exception:

        x = np.arange(len(totale_maand))
        m_tot, b_tot = np.polyfit(x, totale_maand, 1)
        future_x = np.arange(len(totale_maand), len(totale_maand) + h)
        pred_tot = b_tot + m_tot * future_x

    pred_tot = np.maximum(pred_tot, 0.0)  
    pred_tot_series = pd.Series(pred_tot, index=forecast_index)

    types = maand_counts_charging.columns.tolist()


    last_counts = maand_counts_charging.iloc[-1].astype(float)
    last_total = last_counts.sum()
    if last_total <= 0:
        last_12 = maand_counts_charging.tail(12).sum().astype(float)
        if last_12.sum() > 0:
            current_shares = (last_12 / last_12.sum()).to_dict()
        else:

            current_shares = {t: 1.0/len(types) for t in types}
    else:
        current_shares = (last_counts / last_total).to_dict()

    non_ev_targets = {}
    for t in types:
        if t == "Elektrisch":
            continue
        cur = current_shares.get(t, 0.0)
        if t == "Benzine":
            non_ev_targets[t] = cur * 0.15   
        elif t == "Diesel":
            non_ev_targets[t] = cur * 0.10  
        else:
            non_ev_targets[t] = cur * 0.25   

    sum_non_ev_targets = sum(non_ev_targets.values())

    ev_target = max(0.75, min(0.98, 1.0 - sum_non_ev_targets))

    if "Elektrisch" not in types:

        scale = 1.0 / sum_non_ev_targets if sum_non_ev_targets > 0 else 1.0 / max(1, len(types))
        for t in non_ev_targets:
            non_ev_targets[t] = non_ev_targets[t] * scale
        ev_target = 0.0


    targets = {}
    for t in types:
        if t == "Elektrisch":
            targets[t] = ev_target
        else:
            targets[t] = non_ev_targets.get(t, 0.0)

    total_target_sum = sum(targets.values())
    if total_target_sum <= 0:

        targets = {t: 1.0/len(types) for t in types}
    else:
        targets = {t: targets[t]/total_target_sum for t in types}


    t_frac = np.linspace(0, 1, h)
    k = 7.0  
    sigmoid = 1.0 / (1.0 + np.exp(-k*(t_frac - 0.5)))  

    share_dict = {}
    for t in types:
        cur = current_shares.get(t, 0.0)
        targ = targets.get(t, 0.0)

        share_dict[t] = cur + (targ - cur) * sigmoid

    shares_df = pd.DataFrame(share_dict, index=forecast_index)


    row_sums = shares_df.sum(axis=1)
    zero_rows = row_sums == 0
    if zero_rows.any():

        fallback = pd.Series(current_shares)
        fallback = fallback / fallback.sum() if fallback.sum() > 0 else fallback.fillna(1.0/len(types))
        shares_df.loc[zero_rows, :] = fallback.values
        row_sums = shares_df.sum(axis=1)
    shares_df = shares_df.div(row_sums, axis=0)


    future_alloc = shares_df.multiply(pred_tot_series, axis=0)  # per maand counts per type


    forecast_median_charging = pd.DataFrame(index=forecast_index, columns=types)
    for col in types:
        future_monthly = future_alloc[col].fillna(0).values
        last_cumul = cumul_hist_charging[col].iloc[-1] if col in cumul_hist_charging.columns else 0
        cumul_forecast = last_cumul + np.cumsum(np.maximum(future_monthly, 0.0))
        forecast_median_charging[col] = cumul_forecast

    # ----------  selectie categorieën ----------
    categorieen = st.multiselect(
        "Kies brandstoftypes om te tonen",
        options=maand_counts_charging.columns.tolist(),
        default=maand_counts_charging.columns.tolist()
    )

    # ---------- Plotly grafiek  ----------
    fig = go.Figure()

    for col in categorieen:
        # Historisch 
        fig.add_trace(go.Scatter(
            x=cumul_hist_charging.index,
            y=cumul_hist_charging[col],
            mode="lines",
            name=f"{col} (historisch)",
            line=dict(width=2)
        ))
        # Voorspelling 
        fig.add_trace(go.Scatter(
            x=forecast_index,
            y=forecast_median_charging[col].astype(float),
            mode="lines",
            line=dict(dash="dash", width=3),
            name=f"{col} (voorspelling)"
        ))

    fig.update_layout(
        title=f"Voertuigregistraties per brandstoftype — Historisch + voorspelling tot {eindjaar}",
        xaxis_title="Jaar",
        yaxis_title="Aantal voertuigen (cumulatief)",
        hovermode="x unified",
        height=720  
    )

    st.plotly_chart(fig, use_container_width=True)

