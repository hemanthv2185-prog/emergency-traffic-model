import math
from pathlib import Path

import pandas as pd
import streamlit as st
import requests
import joblib
import numpy as np
import folium
from streamlit_folium import st_folium

DATA_PATH = Path(__file__).parent / "Final dp.xlsx"
MODEL_PATH = Path(__file__).parent / "model_outputs" / "best_congestion_model.joblib"
OSRM_URL = "https://router.project-osrm.org/route/v1/driving"

# Approximate demonstration coordinates in Bengaluru. Confirm exact coordinates before any real deployment.
LOCATIONS = {
    "Majestic / KSR Bengaluru": (12.9767, 77.5713),
    "Vidhana Soudha": (12.9796, 77.5907),
    "Indiranagar Metro": (12.9784, 77.6408),
    "Koramangala": (12.9352, 77.6245),
    "M.G. Road Metro": (12.9756, 77.6066),
    "Silk Board Junction": (12.9177, 77.6233),
    "Hebbal Flyover": (13.0358, 77.5970),
    "Whitefield (Hope Farm)": (12.9870, 77.7501),
    "Jayanagar 4th Block": (12.9250, 77.5838),
    "Electronic City": (12.8452, 77.6602),
    "Victoria Hospital": (12.9634, 77.5748),
    "St. John's Medical College Hospital": (12.9279, 77.6271),
    "Manipal Hospital, Old Airport Road": (12.9592, 77.6484),
    "Aster CMI Hospital": (13.0547, 77.5920),
}
AREA_CENTERS = {
    "Indiranagar": (12.9784, 77.6408),
    "Koramangala": (12.9352, 77.6245),
    "M.G. Road": (12.9756, 77.6066),
    "Whitefield": (12.9870, 77.7501),
    "Hebbal": (13.0358, 77.5970),
    "Silk Board": (12.9177, 77.6233),
    "Electronic City": (12.8452, 77.6602),
    "Jayanagar": (12.9250, 77.5838),
}
VEHICLE_PROFILES = {
    "Ambulance": {"multiplier": 2.0, "color": "#16a34a"},
    "Fire engine": {"multiplier": 1.8, "color": "#dc2626"},
    "Emergency response vehicle": {"multiplier": 1.5, "color": "#7c3aed"},
}
ROUTE_COLORS = ["#2563eb", "#f97316", "#7c3aed", "#0891b2", "#db2777", "#a16207"]

st.set_page_config(page_title="Emergency Route Optimizer", page_icon="🚑", layout="wide")
st.title("🚑 Emergency Congestion-Aware Route Optimizer")
st.caption("Same destination, road-following route alternatives, ML-based next-record congestion estimates, and estimated travel times.")

@st.cache_data(show_spinner=False)
def load_dataset(path, modified):
    df = pd.read_excel(path, sheet_name="Connected_Routes")
    for col in ["Congestion Level", "Average Speed", "Traffic Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def build_profiles(df):
    key_cols = [c for c in ["Source_Record_ID", "Date", "Area Name", "Road/Intersection Name"] if c in df.columns]
    unique = df.drop_duplicates(subset=key_cols) if key_cols else df.copy()
    profiles = {}
    for area, group in unique.groupby("Area Name"):
        congestion = group["Congestion Level"].dropna()
        speed = group["Average Speed"].dropna()
        profiles[str(area)] = {
            "congestion": float(congestion.mean()) if len(congestion) else 50.0,
            "speed": float(speed.mean()) if len(speed) else 30.0,
            "rows": len(group),
        }
    return profiles, unique

@st.cache_resource(show_spinner=False)
def load_congestion_model(path, modified):
    if not Path(path).exists():
        return None
    return joblib.load(path)

def build_predicted_profiles(unique_df, model_bundle):
    """Predict the next recorded congestion from each area's latest available historical rows.

    Important: these are predictions from the dataset's latest observations, not live traffic.
    """
    if model_bundle is None:
        return {}
    model = model_bundle["pipeline"]
    features = model_bundle["features"]
    work = unique_df.copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    work = work.dropna(subset=["Date", "Area Name", "Road/Intersection Name"])
    work = work.sort_values("Date").drop_duplicates(
        subset=["Area Name", "Road/Intersection Name"], keep="last"
    ).copy()
    work["DayOfWeek"] = work["Date"].dt.dayofweek
    work["Month"] = work["Date"].dt.month
    work["DayOfYear"] = work["Date"].dt.dayofyear
    for col in features:
        if col not in work.columns:
            work[col] = np.nan if col in model_bundle["numeric_features"] else "Unknown"
    try:
        work["_predicted_congestion"] = np.clip(model.predict(work[features]), 0, 100)
    except Exception as exc:
        st.warning(f"Could not apply the saved congestion model; using historical averages instead. Details: {exc}")
        return {}
    profiles = {}
    for area, group in work.groupby("Area Name"):
        profiles[str(area)] = float(group["_predicted_congestion"].mean())
    return profiles

def nearest_area_congestion(route_coords, profiles):
    centers = [(area, coord) for area, coord in AREA_CENTERS.items() if area in profiles]
    if not centers:
        return 50.0
    sampled = route_coords[::max(1, len(route_coords)//30)]
    scores = []
    for lon, lat in sampled:
        closest_area, closest_dist = None, float("inf")
        for area, (alat, alon) in centers:
            dist = ((lat-alat)*111.0)**2 + ((lon-alon)*111.0*math.cos(math.radians(lat)))**2
            if dist < closest_dist:
                closest_area, closest_dist = area, dist
        if closest_area:
            scores.append(profiles[closest_area]["congestion"])
    return sum(scores)/len(scores) if scores else 50.0

def fetch_routes(origin, destination):
    o = LOCATIONS[origin]
    d = LOCATIONS[destination]
    # Request multiple alternative routes from OSRM. Service may return fewer than requested.
    url = f"{OSRM_URL}/{o[1]},{o[0]};{d[1]},{d[0]}"
    params = {
        "alternatives": "3",
        "steps": "false",
        "overview": "full",
        "geometries": "geojson",
    }
    response = requests.get(url, params=params, timeout=(10, 35))
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != "Ok" or not payload.get("routes"):
        raise RuntimeError(f"Routing service did not return a route: {payload.get('code', 'unknown')}")
    return payload["routes"]

if not DATA_PATH.exists():
    st.error("Dataset file not found. Place Final dp.xlsx in the same folder as app.py.")
    st.stop()

try:
    df = load_dataset(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    profiles, unique_df = build_profiles(df)
    model_bundle = load_congestion_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime) if MODEL_PATH.exists() else None
    predicted_profiles = build_predicted_profiles(unique_df, model_bundle) if model_bundle else {}
except Exception as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()

with st.sidebar:
    st.header("Emergency request")
    vehicle = st.selectbox("Emergency vehicle", list(VEHICLE_PROFILES.keys()))
    leg = st.selectbox(
        "Journey stage",
        ["Vehicle → incident", "Incident → hospital", "Fire station → incident", "Custom route"],
        help="For ambulance pickup, use Vehicle → incident. After pickup, select Incident → hospital."
    )
    if leg == "Vehicle → incident":
        start_label, end_label = "Vehicle starting location", "Incident location"
        start_default, end_default = "Majestic / KSR Bengaluru", "Vidhana Soudha"
    elif leg == "Incident → hospital":
        start_label, end_label = "Incident / patient pickup", "Hospital destination"
        start_default, end_default = "Vidhana Soudha", "Victoria Hospital"
    elif leg == "Fire station → incident":
        start_label, end_label = "Fire engine starting location", "Fire incident location"
        start_default, end_default = "Majestic / KSR Bengaluru", "Vidhana Soudha"
    else:
        start_label, end_label = "Starting location", "Destination"
        start_default, end_default = "Majestic / KSR Bengaluru", "Vidhana Soudha"

    names = list(LOCATIONS.keys())
    start_index = names.index(start_default) if start_default in names else 0
    origin = st.selectbox(start_label, names, index=start_index)
    dest_names = [name for name in names if name != origin]
    dest_default = end_default if end_default in dest_names else dest_names[0]
    destination = st.selectbox(end_label, dest_names, index=dest_names.index(dest_default))
    congestion_weight = st.slider("Congestion penalty", 0.0, 3.0, 1.5, 0.1,
                                  help="Higher values make the ranking penalize routes passing through areas with higher congestion estimates.")
    model_available = MODEL_PATH.exists()
    use_ml_prediction = st.checkbox(
        "Use ML-predicted next-record congestion", value=model_available, disabled=not model_available,
        help="Uses the trained model on the latest historical record per road/area. This is not live traffic."
    )

st.subheader("How route selection works")
st.write("All route options share the same start and destination. Each option is drawn in a different colour. The recommended route is selected using estimated duration plus a congestion penalty. When the ML option is enabled, the penalty uses the model predicted next-record congestion for each area; otherwise it uses historical averages.")

m1, m2, m3 = st.columns(3)
m1.metric("Dataset rows", f"{len(df):,}")
m2.metric("Unique source records", f"{len(unique_df):,}")
m3.metric("Historical areas", f"{len(profiles)}")
if use_ml_prediction and predicted_profiles:
    st.caption(f"ML mode: {model_bundle.get('model_name', 'saved model')} predicts the next recorded congestion from the latest available historical observations. The workbook ends in 2024; this is not a live/current traffic forecast. OSRM does not provide live congestion.")
else:
    st.caption("Historical mode: congestion is based on dataset averages, not live traffic. Public OSRM returns road-following routes but does not provide live congestion.")

if st.button("🚨 Compare routes", type="primary", use_container_width=True):
    try:
        with st.spinner("Requesting road-following alternatives for the same destination..."):
            routes = fetch_routes(origin, destination)

        congestion_profiles = profiles
        congestion_label = "Historical congestion proxy"
        if use_ml_prediction and predicted_profiles:
            congestion_profiles = {area: {**profile, "congestion": predicted_profiles.get(area, profile["congestion"])}
                                  for area, profile in profiles.items()}
            congestion_label = "ML-predicted next-record congestion"
        for route in routes:
            coords = route["geometry"]["coordinates"]
            congestion = nearest_area_congestion(coords, congestion_profiles)
            duration_min = route["duration"] / 60.0
            distance_km = route["distance"] / 1000.0
            penalty_factor = 1 + (max(0, min(100, congestion))/100.0) * congestion_weight * VEHICLE_PROFILES[vehicle]["multiplier"]
            route["_congestion"] = congestion
            route["_duration_min"] = duration_min
            route["_distance_km"] = distance_km
            route["_score"] = duration_min * penalty_factor

        recommended = min(routes, key=lambda route: route["_score"])
        center = [
            (LOCATIONS[origin][0] + LOCATIONS[destination][0])/2,
            (LOCATIONS[origin][1] + LOCATIONS[destination][1])/2
        ]
        fmap = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap", control_scale=True)

        route_rows = []
        for i, route in enumerate(routes):
            color = "#16a34a" if route is recommended else ROUTE_COLORS[i % len(ROUTE_COLORS)]
            coords = [(lat, lon) for lon, lat in route["geometry"]["coordinates"]]
            title = "RECOMMENDED" if route is recommended else f"Alternative {i+1}"
            folium.PolyLine(
                coords, color=color, weight=8 if route is recommended else 5,
                opacity=0.95 if route is recommended else 0.8,
                tooltip=f"{title} | {route['_distance_km']:.1f} km | {route['_duration_min']:.1f} min | congestion proxy {route['_congestion']:.0f}/100"
            ).add_to(fmap)
            route_rows.append({
                "Route": title if route is recommended else f"Alternative {i+1}",
                "Colour": "Green (recommended)" if route is recommended else color,
                "Distance (km)": round(route["_distance_km"], 2),
                "Estimated time (min)": round(route["_duration_min"], 1),
                f"{congestion_label} / 100": round(route["_congestion"], 1),
                "Ranking score": round(route["_score"], 2),
                "_selected": route is recommended
            })

        folium.Marker(LOCATIONS[origin], tooltip=f"START: {origin}",
                      popup=f"START: {origin}",
                      icon=folium.Icon(color="green", icon="play")).add_to(fmap)
        folium.Marker(LOCATIONS[destination], tooltip=f"DESTINATION: {destination}",
                      popup=f"DESTINATION: {destination}",
                      icon=folium.Icon(color="red", icon="flag")).add_to(fmap)
        with st.container():
            st_folium(fmap, use_container_width=True, height=570, returned_objects=[])

        st.markdown(f"### Recommended route: {origin} → {destination}")
        a, b, c = st.columns(3)
        a.metric("Estimated distance", f"{recommended['_distance_km']:.2f} km")
        b.metric("Estimated travel time", f"{recommended['_duration_min']:.1f} min")
        c.metric("Predicted congestion" if (use_ml_prediction and predicted_profiles) else "Historical congestion", f"{recommended['_congestion']:.1f}/100")

        st.markdown("### Route comparison")
        display_rows = [{k:v for k,v in row.items() if not k.startswith("_")} for row in route_rows]
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)
        if len(routes) == 1:
            st.warning("The routing service returned only one route for this pair. The map cannot compare multiple paths unless the service provides alternatives. Try a different start/destination pair.")
        st.download_button(
            "Download route comparison CSV",
            pd.DataFrame(display_rows).to_csv(index=False).encode("utf-8"),
            file_name="emergency_route_comparison.csv",
            mime="text/csv"
        )
        if use_ml_prediction and predicted_profiles:
            st.info("The recommended route is the lowest-scoring option among alternatives returned. The congestion penalty uses the trained model's estimate of the next recorded congestion based on the latest historical row per road/area. Because the workbook ends in 2024, this is not live traffic and is not a reliable current ETA.")
        else:
            st.info("The recommended route is the lowest-scoring option among alternatives returned. Estimated time comes from OSRM; congestion is a historical area-level proxy, not live road traffic.")
    except requests.exceptions.Timeout:
        st.error("The routing service timed out. Wait briefly and retry once. Public demo routing services may be busy.")
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not reach the routing service. Check internet access. Details: {exc}")
    except Exception as exc:
        st.error(f"Route comparison failed: {exc}")

with st.expander("Dataset profile by area"):
    st.dataframe(pd.DataFrame([
        {"Area": area, "Average historical congestion": round(profile["congestion"], 2),
         "Average speed (dataset units)": round(profile["speed"], 2), "Unique records": profile["rows"]}
        for area, profile in profiles.items()
    ]), use_container_width=True, hide_index=True)

st.divider()
st.caption("Academic demonstration only. This is not connected to the 108 dispatch system and must not be used for real emergency navigation. Real deployment requires verified locations, live traffic, road restrictions, emergency-service validation and appropriate permissions.")
