"""
SIH26009: Manganese Mine Production Shortfall Forecaster

"""

import sys
import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Load only the best model selected from training benchmark (GBR)
try:
    gbr_model = joblib.load(os.path.join(MODELS_DIR, 'best_model_gbr.pkl'))
    feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure models/best_model_gbr.pkl and feature_names.pkl exist.")
    sys.exit(1)

MINES = {
    1: {'name': 'Balaghat', 'state': 'Madhya Pradesh', 'method': 'Underground', 'planned': 3074.0, 'is_opencast': 0, 'temp': 26.5, 'humidity': 55.0},
    2: {'name': 'Gumgaon', 'state': 'Maharashtra', 'method': 'Underground', 'planned': 1627.4, 'is_opencast': 0, 'temp': 27.5, 'humidity': 51.0},
    3: {'name': 'Kandri', 'state': 'Maharashtra', 'method': 'Underground', 'planned': 1175.3, 'is_opencast': 0, 'temp': 27.2, 'humidity': 53.0},
    4: {'name': 'Munsar', 'state': 'Maharashtra', 'method': 'Underground', 'planned': 1084.9, 'is_opencast': 0, 'temp': 27.2, 'humidity': 52.0},
    5: {'name': 'Ukwa', 'state': 'Madhya Pradesh', 'method': 'Underground', 'planned': 904.1, 'is_opencast': 0, 'temp': 25.8, 'humidity': 58.0},
    6: {'name': 'Chikla', 'state': 'Maharashtra', 'method': 'Underground', 'planned': 723.3, 'is_opencast': 0, 'temp': 27.1, 'humidity': 52.0},
    7: {'name': 'Dongri Buzurg', 'state': 'Maharashtra', 'method': 'Opencast', 'planned': 452.1, 'is_opencast': 1, 'temp': 27.3, 'humidity': 54.0},
}

def make_feature_vector(mine_info, planned, dt, trans, rain, blast, lab):
    row = {f: 0.0 for f in feature_names}
    row['planned_production_tonnes'] = float(planned)
    row['equipment_downtime_hrs'] = float(dt)
    row['transport_availability_pct'] = float(trans)
    row['labour_attendance_pct'] = float(lab)
    row['blasting_delay_hrs'] = float(blast)
    row['rainfall_mm'] = float(rain)
    row['rainfall_lag1'] = float(rain) * 0.7
    row['rainfall_3d_sum'] = float(rain) * 1.5
    row['temp_mean_c'] = mine_info['temp']
    row['temp_max_c'] = mine_info['temp'] + 6.0
    row['relative_humidity_pct'] = mine_info['humidity']
    row['wind_speed_m_s'] = 2.5
    row['solar_radiation_mj_m2_day'] = 18.0
    row['is_opencast'] = mine_info['is_opencast']
    row['month'] = 7 if rain > 20 else 2
    row['day_of_week'] = 2
    
    col_name = f"mine_{mine_info['name']}"
    if col_name in row:
        row[col_name] = 1.0
        
    return pd.DataFrame([row])[feature_names]

def evaluate_shift(mine_info, planned, dt, trans, rain, blast, lab):
    X = make_feature_vector(mine_info, planned, dt, trans, rain, blast, lab)
    
    # Prediction from the selected best model (GBR)
    pred_gbr = float(gbr_model.predict(X)[0])
    
    # Bound physical constraints
    pred_gbr = max(-0.05 * planned, pred_gbr)
    pred_actual = max(0.0, planned - pred_gbr)
    shortfall_pct = (pred_gbr / planned) * 100.0
    
    # Risk calculation
    if shortfall_pct <= 2.5:
        risk = "LOW RISK"
    elif shortfall_pct <= 7.0:
        risk = "MODERATE RISK"
    elif shortfall_pct <= 15.0:
        risk = "HIGH RISK"
    else:
        risk = "CRITICAL RISK"
        
    # Contributing factors
    factors = []
    if dt >= 4.0:
        factors.append(f"Major Equipment Downtime ({dt} hrs) -> Critical bottleneck halting hoisting/crushing.")
    elif dt > 0:
        factors.append(f"Minor Equipment Stoppage ({dt} hrs).")
        
    if trans < 85:
        factors.append(f"Severe Transport Shortage ({trans}%) -> Haul truck shortage starving surface transfer.")
    elif trans < 95:
        factors.append(f"Moderate Logistics Friction ({trans}% transport availability).")
        
    if rain >= 30:
        factors.append(f"Monsoon Precipitation ({rain} mm) -> Sump flooding and slippery haul road ramps.")
    elif rain >= 10:
        factors.append(f"Light/Moderate Rainfall ({rain} mm) impacting dumper speeds.")
        
    if blast >= 2.5:
        factors.append(f"Blasting Postponement ({blast} hrs) -> Lack of fragmented muckpile for loading shovels.")
        
    if lab < 90:
        factors.append(f"Labour Absenteeism ({lab}% attendance) -> Underground drilling & timbering crews shorthanded.")
        
    if not factors:
        factors.append("Optimal shift parameters. Zero machine stops and dry surface conditions.")
        
    # Actionable Directives
    recs = []
    if dt >= 3.0:
        recs.append("DECISION-SUPPORT: Dispatch emergency fitters to primary hoist/crusher. Route tramming to backup ore pass.")
    if trans < 88:
        recs.append("DECISION-SUPPORT: Request auxiliary external contractor dumpers; dump to emergency pithead surge stockpile.")
    if rain >= 25:
        recs.append("DECISION-SUPPORT: Start secondary pit dewatering pumps; spread crushed dolomite gravel on primary ramp switchbacks.")
    if blast >= 2.0:
        recs.append("DECISION-SUPPORT: Divert shovel teams to secondary fragmented stock; adjust ventilation shift clearance.")
    if lab < 88:
        recs.append("DECISION-SUPPORT: Reassign surface civil maintenance staff to support underground development faces.")
    if not recs:
        recs.append("DECISION-SUPPORT: Shift proceeding on plan. Maintain standard weighbridge monitoring.")
        
    return {
        'mine': mine_info['name'],
        'method': mine_info['method'],
        'planned': planned,
        'pred_actual': pred_actual,
        'pred_shortfall': pred_gbr,
        'shortfall_pct': shortfall_pct,
        'risk': risk,
        'factors': factors,
        'recs': recs
    }

def print_banner():
    print("\nSIH26009: MANGANESE PRODUCTION SHORTFALL FORECASTER")
    print("Smart India Hackathon 2026 Prototype | MOIL Central Indian Belt\n")

def print_results(res):
    print("\n------------------------------------------------------------")
    print(f" OPERATIONAL FORECAST: {res['mine'].upper()} ({res['method'].upper()})")
    print("------------------------------------------------------------")
    print(f" Target Budget Quota:        {res['planned']:8.1f} Tonnes")
    print(f" Forecasted Actual Output:   {res['pred_actual']:8.1f} Tonnes")
    print(f" Expected Daily Shortfall:   {res['pred_shortfall']:8.1f} Tonnes ({res['shortfall_pct']:+.1f}%)")
    print(f" Operational Threat Level:   [{res['risk']}]")
    print("------------------------------------------------------------")
    
    print("\n[ROOT-CAUSE ATTRIBUTION]:")
    for f in res['factors']:
        print(f"  * {f}")
        
    print("\n[ADVISORY DECISION-SUPPORT DIRECTIVES]:")
    for r in res['recs']:
        print(f"  [!] {r}")
    print()

def run_cli():
    print_banner()
    
    print("Select a Mine to Evaluate:")
    for num, m in MINES.items():
        print(f"  [{num}] {m['name']:14s} ({m['state']:14s}) - {m['method']:11s} [Planned: {m['planned']} t/day]")
        
    try:
        choice = input("\nEnter Mine Number (1-7) [Default: 1 - Balaghat]: ").strip()
        mine_idx = int(choice) if choice in [str(i) for i in range(1, 8)] else 1
    except:
        mine_idx = 1
        
    selected_mine = MINES[mine_idx]
    print(f"\n>> Selected: {selected_mine['name']} ({selected_mine['method']})")
    
    print("\nSelect an Operational Mode:")
    print("  [1] Normal Dry Day (Optimal Shift)")
    print("  [2] Major Equipment Breakdown (e.g. 7.5 hrs hoist failure)")
    print("  [3] Heavy Monsoon Rain (e.g. 50 mm cloudburst + road slush)")
    print("  [4] Blasting Postponement & Workforce Shortage")
    print("  [5] Custom Input (Enter your own shift numbers)")
    
    mode = input("\nChoose Mode (1-5) [Default: 5 - Custom Input]: ").strip()
    
    if mode == '1':
        planned = selected_mine['planned']
        dt = 0.0
        trans = 98.0
        rain = 0.0
        blast = 0.0
        lab = 98.0
    elif mode == '2':
        planned = selected_mine['planned']
        dt = 7.5
        trans = 88.0
        rain = 5.0
        blast = 0.0
        lab = 94.0
    elif mode == '3':
        planned = selected_mine['planned']
        dt = 0.0
        trans = 75.0
        rain = 55.0
        blast = 1.5
        lab = 92.0
    elif mode == '4':
        planned = selected_mine['planned']
        dt = 0.0
        trans = 86.0
        rain = 4.0
        blast = 4.0
        lab = 84.0
    else:
        print("\nEnter Shift Conditions (Press Enter to accept default):")
        
        inp = input(f"  - Planned Target Quota (Tonnes) [Default: {selected_mine['planned']}]: ").strip()
        planned = float(inp) if inp else selected_mine['planned']
        
        inp = input("  - Equipment Downtime (Hours, 0-24) [Default: 0.0]: ").strip()
        dt = float(inp) if inp else 0.0
        
        inp = input("  - Internal Transport Availability (% 50-100) [Default: 95.0]: ").strip()
        trans = float(inp) if inp else 95.0
        
        inp = input("  - On-Site Rainfall (mm, 0-150) [Default: 0.0]: ").strip()
        rain = float(inp) if inp else 0.0
        
        inp = input("  - Blasting Delay (Hours, 0-6) [Default: 0.0]: ").strip()
        blast = float(inp) if inp else 0.0
        
        inp = input("  - Labour Attendance (% 50-100) [Default: 95.0]: ").strip()
        lab = float(inp) if inp else 95.0

    res = evaluate_shift(selected_mine, planned, dt, trans, rain, blast, lab)
    print_results(res)

if __name__ == '__main__':
    run_cli()
