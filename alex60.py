import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import numpy as np
from pvlib import location, irradiance, atmosphere, temperature
import datetime
import pytz

# 1. APP-KONFIGURATION
st.set_page_config(page_title="PV Alex Balkonkraftwerk 60 grad ", layout="centered")

# --- PASSWORT ABFRAGE ---
def check_password():
    if "password_correct" not in st.session_state:
        st.markdown("<h2 style='text-align: center; color: #f1c40f;'>☀️ PV Alex Balkonkraftwerk 60° >", unsafe_allow_html=True)
        pwd = st.text_input("Passwort:", type="password", key="password_input")
        if pwd:
            if pwd == st.secrets.get("password", "admin"): # Fallback auf 'admin' falls secrets fehlen
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("😕 Passwort falsch.")
        return False
    return True

if not check_password():
    st.stop()

# --- PARAMETER ---
ALBEDO = 0.2
TURBIDITY_MONTHLY = [2.1, 2.2, 2.5, 2.9, 3.2, 3.4, 3.5, 3.3, 2.9, 2.6, 2.3, 2.1]

configs = [
    {"name": "Balkon", "lat": 49.482869333, "lon": 8.2741404808, "wp": 450, "num": 2, "tilt": 60, "azi": 185, "color": "#f1c40f", "shade": None}
]

@st.cache_data(ttl=3600)
def get_weather_dwd(lat, lon, start, end):
    try:
        url = (f"https://api.open-meteo.com/v1/dwd-icon?latitude={lat}&longitude={lon}"
               f"&hourly=cloudcover,temperature_2m,windspeed_10m&start_date={start}&end_date={end}&timezone=Europe%2FBerlin")
        res = requests.get(url, timeout=15).json()
        return pd.DataFrame({
            'cloud': np.array(res['hourly']['cloudcover']),
            'temp_air': np.array(res['hourly']['temperature_2m']),
            'wind': np.array(res['hourly']['windspeed_10m']) / 3.6
        })
    except: return None

# --- UI & LOGIK ---
START_DATE = st.date_input("Startdatum", datetime.date.today())
if START_DATE:
    tz = pytz.timezone('Europe/Berlin')
    times = pd.date_range(start=pd.Timestamp(START_DATE).tz_localize(tz), periods=72, freq='h')

    weather = get_weather_dwd(configs[0]['lat'], configs[0]['lon'], START_DATE, START_DATE + datetime.timedelta(days=2))
    
    if weather is not None:
        weather = weather.iloc[:len(times)]
        site = location.Location(configs[0]['lat'], configs[0]['lon'], tz='Europe/Berlin', altitude=100)
        solpos = site.get_solarposition(times)
        dni_extra = irradiance.get_extra_radiation(times)
        
        rel_airmass = atmosphere.get_relative_airmass(solpos['zenith'])
        am_abs = atmosphere.get_absolute_airmass(rel_airmass)
        linke_turbidity = TURBIDITY_MONTHLY[START_DATE.month - 1]

        ergebnisse = {}
        for f in configs:
            cs = site.get_clearsky(times, model='ineichen', linke_turbidity=linke_turbidity)
            cloud_factor = weather['cloud'].values / 100
            ghi_adj = cs['ghi'].values * (1 - 0.75 * (cloud_factor ** 3.4))
            dni_adj = cs['dni'].values * (1 - cloud_factor**2)
            dhi_adj = np.maximum(ghi_adj - (dni_adj * np.cos(np.radians(solpos['zenith'].values))), 
                                 cs['dhi'].values * (0.3 + 0.7 * cloud_factor))

            if f['shade']:
                s = f['shade']
                mask = (solpos['azimuth'] > s['azi_min']) & (solpos['azimuth'] < s['azi_max']) & (solpos['elevation'] < s['elev_limit'])
                dni_adj[mask] = 0

            poa = irradiance.get_total_irradiance(
                f['tilt'], f['azi'], solpos['zenith'], solpos['azimuth'],
                dni_adj, ghi_adj, dhi_adj, dni_extra=dni_extra, model='perez', albedo=ALBEDO
            )

            # Korrektur für math. Stabilität
            t_cell = temperature.faiman(poa['poa_global'], weather['temp_air'].values, weather['wind'].values)
            f_temp = 1 + -0.0035 * (t_cell.values - 25)
            f_spectral = np.maximum(0.8, 1 - (am_abs.values / 150))
            f_lowlight = np.where(poa['poa_global'].values < 50, 0.85, 1.0)

            prod = (poa['poa_global'].values / 1000) * ((f['wp'] * f['num']) / 1000) * 0.85 * f_temp * f_spectral * f_lowlight
            ergebnisse[f['name']] = np.nan_to_num(prod)

        df_results = pd.DataFrame(ergebnisse, index=times)
        tages_summen = df_results.sum(axis=1).groupby(df_results.index.date).sum()

        # --- HEADER ANZEIGE ---
        header_str = " | ".join([f"{d.strftime('%d.%m.')}: {s:.1f} kWh" for d, s in tages_summen.items()])
        st.markdown(f"### ☀️ {header_str}")

        # --- INTERAKTIVE PLOTLY GRAFIK ---
        fig = go.Figure()
        for f in configs:
            fig.add_trace(go.Scatter(
                x=df_results.index, y=df_results[f['name']],
                name=f['name'], mode='lines', stackgroup='one',
                line=dict(width=0.5, color=f['color']), fillcolor=f['color'],
                hovertemplate='%{y:.2f} kW'
            ))

        # JETZT-Linie Fix: Zeitstempel explizit konvertieren
        now = datetime.datetime.now(tz)
        if df_results.index.min() <= now <= df_results.index.max():
            fig.add_vline(x=now.timestamp() * 1000, line_width=2, line_dash="dash", line_color="red")
            # Beschriftung separat hinzufügen, um den Fehler in 'annotation_text' zu umgehen
            fig.add_annotation(x=now, y=df_results.sum(axis=1).max(), text="JETZT", showarrow=False, font=dict(color="red"))

        fig.update_layout(
            template="plotly_dark", height=450, margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified", xaxis=dict(rangeslider=dict(visible=True), type="date")
        )
        st.plotly_chart(fig, use_container_width=True)

        ertrag_heute = tages_summen.get(START_DATE, 0.0)
        st.success(f"Prognostizierter Ertrag für {START_DATE.strftime('%d.%m.')}: {ertrag_heute:.1f} kWh")
    else:
        st.error("Wetterdaten nicht verfügbar.")
