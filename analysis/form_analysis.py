"""
User-study analysis for the Depth-Aware Prosthetic Vision project.

Reads the anonymized survey responses (data/survey_responses.csv) and
regenerates the four result figures in results/.

Usage:
    python analysis/form_analysis.py
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "survey_responses.csv"
RESULTS_DIR = ROOT / "results"

def analyze_vision_data_final(file_path):
    print(f"Processing file: {file_path}")
    
    # 1. Load data
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: File not found.")
        return

    # 2. Rename columns (standardization)
    if len(df.columns) >= 14:
        df.columns = [
            'Timestamp', 'Age',
            'A1_Distinguish', 'A1_Identify',
            'A2_Distinguish', 'A2_Identify',
            'B_Pref_1', 'B_Pref_2', 'B_Pref_3',
            'Safety_Score',
            'C_Navigation',
            'D_Movement_Pref',
            'Fatigue', 'Discomfort'
        ]
    
    # --- 3. CONFIGURATION: LABELS ---
    # In the (Italian) survey, "Modalità 2" / "Modalità B" is the proposed Hybrid mode
    # and "Modalità 1" / "Modalità A" is the RGB-only baseline.
    label_hybrid = "Hybrid Mode"
    label_rgb = "RGB-only Mode"
    label_neutral = "Neutral"

    def map_pref_static(text):
        if isinstance(text, str):
            if "Modalità 2" in text: return label_hybrid
            elif "Modalità 1" in text: return label_rgb
        return label_neutral

    def map_pref_dynamic(text):
        if isinstance(text, str):
            if "Modalità B" in text or "Modalita B" in text or "Modalità 2" in text: 
                return label_hybrid
            if "Modalità A" in text or "Modalita A" in text or "Modalità 1" in text: 
                return label_rgb
        return label_neutral

    def is_detected(text):
        return isinstance(text, str) and "Sì" in text

    # 4. Apply mapping
    df['B_Pref_1_Mapped'] = df['B_Pref_1'].apply(map_pref_static)
    df['B_Pref_2_Mapped'] = df['B_Pref_2'].apply(map_pref_static)
    df['B_Pref_3_Mapped'] = df['B_Pref_3'].apply(map_pref_static)
    df['D_Movement_Pref_Mapped'] = df['D_Movement_Pref'].apply(map_pref_dynamic)

    # 5. Compute statistics
    b1_dist = df['B_Pref_1_Mapped'].value_counts(normalize=True) * 100
    b2_dist = df['B_Pref_2_Mapped'].value_counts(normalize=True) * 100
    b3_dist = df['B_Pref_3_Mapped'].value_counts(normalize=True) * 100
    d_dist = df['D_Movement_Pref_Mapped'].value_counts(normalize=True) * 100

    detect_obj1 = df['A1_Distinguish'].apply(is_detected).mean() * 100
    detect_obj2 = df['A2_Distinguish'].apply(is_detected).mean() * 100
    acc_obj1 = df['A1_Identify'].str.contains("Laptop", case=False, na=False).mean() * 100
    acc_obj2 = df['A2_Identify'].str.contains("Bottiglia", case=False, na=False).mean() * 100

    safety_mean = df['Safety_Score'].mean()
    safety_std = df['Safety_Score'].std()
    fatigue_mean = df['Fatigue'].mean()
    fatigue_std = df['Fatigue'].std()
    discomfort_mean = df['Discomfort'].mean()
    discomfort_std = df['Discomfort'].std()

    # --- 6. PLOTTING (ENGLISH ACADEMIC) ---
    sns.set_style("whitegrid")
    
    # Colors: Green for Hybrid (Project Success), Red for RGB-only (Baseline)
    colors_map = {label_hybrid: '#2ecc71', label_rgb: '#e74c3c', label_neutral: '#95a5a6'}

    # --> GRAPH 1: DYNAMIC MOVEMENT (Test D)
    plt.figure(figsize=(10, 7))
    order_d = [label_hybrid, label_rgb, label_neutral]
    d_plot_data = d_dist.reindex(order_d, fill_value=0)
    labels_d_visual = [x.replace(" Mode", "\nMode") for x in d_plot_data.index]

    sns.barplot(x=labels_d_visual, y=d_plot_data.values, hue=labels_d_visual, palette=[colors_map[x] for x in order_d], legend=False)
    plt.title('Dynamic Scenario Performance (Movement)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('User Preference (%)', fontsize=14)
    plt.ylim(0, 100)
    plt.xticks(fontsize=12)
    for i, v in enumerate(d_plot_data.values):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=12, fontweight='bold')
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(RESULTS_DIR / 'Graph_1_Movement_Final.png', dpi=300)

    # --> GRAPH 2: STATIC SCENARIOS (Test B)
    scenarios_map_eng = {
        'Video 1 (Geometry)': 'Chair\n(Geometry)',
        'Video 2 (Cleanliness)': 'Bed\n(Obstacles)',
        'Video 3 (Details)': 'Door\n(Contrast)'
    }
    
    b_data = pd.DataFrame({
        'Scenario': [scenarios_map_eng['Video 1 (Geometry)'], scenarios_map_eng['Video 2 (Cleanliness)'], scenarios_map_eng['Video 3 (Details)']],
        label_hybrid: [b1_dist.get(label_hybrid, 0), b2_dist.get(label_hybrid, 0), b3_dist.get(label_hybrid, 0)],
        label_rgb: [b1_dist.get(label_rgb, 0), b2_dist.get(label_rgb, 0), b3_dist.get(label_rgb, 0)],
        label_neutral: [b1_dist.get(label_neutral, 0), b2_dist.get(label_neutral, 0), b3_dist.get(label_neutral, 0)]
    })
    b_data_melted = b_data.melt(id_vars="Scenario", var_name="Algorithm", value_name="Preference")

    plt.figure(figsize=(12, 8))
    sns.barplot(data=b_data_melted, x="Scenario", y="Preference", hue="Algorithm", palette=colors_map)
    plt.title('Geometric Perception Comparison (Static Scenarios)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('User Preference (%)', fontsize=14)
    plt.ylim(0, 100)
    plt.xticks(fontsize=12)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12, title="Algorithm")
    plt.subplots_adjust(bottom=0.2, right=0.8)
    plt.savefig(RESULTS_DIR / 'Graph_2_Static_Final.png', dpi=300)

    # --> GRAPH 3: SEMANTIC RECOGNITION (Test A)
    labels_a = ['Laptop\n(Target 1)', 'Bottle\n(Target 2)']
    detect_vals = [detect_obj1, detect_obj2]
    identify_vals = [acc_obj1, acc_obj2]
    x = range(len(labels_a))
    width = 0.35
    plt.figure(figsize=(10, 7))
    plt.bar([p - width/2 for p in x], detect_vals, width, label='Detection (Shape)', color='#3498db')
    plt.bar([p + width/2 for p in x], identify_vals, width, label='Identification (Name)', color='#2c3e50')
    plt.ylabel('Success Rate (%)', fontsize=14)
    plt.title('Semantic Efficacy of Bionic Vision', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(x, labels_a, fontsize=12)
    plt.ylim(0, 100)
    plt.legend(fontsize=12)
    for i in x:
        plt.text(i - width/2, detect_vals[i] + 2, f"{detect_vals[i]:.1f}%", ha='center', fontweight='bold')
        plt.text(i + width/2, identify_vals[i] + 2, f"{identify_vals[i]:.1f}%", ha='center', fontweight='bold')
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(RESULTS_DIR / 'Graph_3_Recognition_Final.png', dpi=300)

    # --> GRAPH 4: UX ASSESSMENT
    ux_labels = ['Mental\nFatigue', 'Visual\nDiscomfort', 'Perceived\nSafety']
    ux_scores = [fatigue_mean, discomfort_mean, safety_mean]
    ux_stds = [fatigue_std, discomfort_std, safety_std]
    plt.figure(figsize=(10, 7))
    colors_ux = ['#e67e22', '#e74c3c', '#3498db']
    bars = plt.bar(ux_labels, ux_scores, yerr=ux_stds, capsize=10, color=colors_ux, alpha=0.9)
    plt.title('User Experience Assessment (1-5 Scale)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Mean Score', fontsize=14)
    plt.ylim(0, 5.5)
    plt.axhline(y=3, color='gray', linestyle='--', alpha=0.5)
    plt.xticks(fontsize=12)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f"{yval:.2f}", ha='center', fontsize=12, fontweight='bold')
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(RESULTS_DIR / 'Graph_4_UX_Final.png', dpi=300)
    
    print(f"Graphs successfully generated in {RESULTS_DIR}")


if __name__ == "__main__":
    analyze_vision_data_final(DATA_FILE)
