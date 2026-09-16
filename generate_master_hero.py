"""
Modul: E-Com Master Hero Generator
Beschreibung: Generiert ein gestochen scharfes, 100% echtes 4-in-1 Executive Data-Science Dashboard
              direkt aus den 12.330 realen Online-Shopping-Sessions.
Autoren: Alex & Alan
"""

from pathlib import Path
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# 1. Stil- und Farbkonfiguration (Dark Modern Slate Theme)
BG_COLOR = '#0b111e'
SURFACE_COLOR = '#131f32'
TEXT_MAIN = '#f8fafc'
TEXT_MUTED = '#94a3b8'
GRID_COLOR = '#1e2f4a'
ACCENT_LEMON = '#ccff00'
ACCENT_CYAN = '#38bdf8'
ACCENT_PURPLE = '#c084fc'

mpl.rcParams.update({
    'figure.facecolor': BG_COLOR,
    'axes.facecolor': SURFACE_COLOR,
    'axes.edgecolor': GRID_COLOR,
    'axes.labelcolor': TEXT_MUTED,
    'xtick.color': TEXT_MUTED,
    'ytick.color': TEXT_MUTED,
    'text.color': TEXT_MAIN,
    'grid.color': GRID_COLOR,
    'grid.linestyle': '--',
    'grid.alpha': 0.6,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans']
})

# 2. Daten laden & vorbereiten
base_dir = Path(__file__).resolve().parent
csv_path = base_dir / 'online_shoppers_intention.csv'
df = pd.read_csv(csv_path)

top_features = ['PageValues', 'ExitRates', 'ProductRelated_Duration']
X = df[top_features]
y = df['Revenue'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(16, 8), random_state=42, max_iter=300)
mlp.fit(X_train_scaled, y_train)
probs = mlp.predict_proba(X_test_scaled)[:, 1]

# 3. 4-Panel Master Dashboard erstellen (16:9 Aspect Ratio)
fig = plt.figure(figsize=(16, 9), dpi=300)
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25, left=0.08, right=0.95, top=0.88, bottom=0.1)

# Gesamt-Titel & Header
fig.text(0.08, 0.94, 'E-COMMERCE CONVERSION INTELLIGENCE // EXECUTIVE ML DASHBOARD', 
         fontsize=18, fontweight='bold', color=TEXT_MAIN)
fig.text(0.08, 0.91, 'Modell: Multi-Layer Perceptron (Scikit-Learn) | Datensatz: 12.330 Sessions | Inferenz: 3 Kernprädiktoren', 
         fontsize=11, color=ACCENT_CYAN)

# Panel 1: Wahrscheinlichkeitsdichte & Smart Voucher Trigger Zone
ax1 = fig.add_subplot(gs[0, 0])
counts, bins, _ = ax1.hist(probs, bins=45, color='#253856', edgecolor=BG_COLOR, alpha=0.9)
# Highlight 40%-60%
ax1.axvspan(0.40, 0.60, color=ACCENT_LEMON, alpha=0.35, label='Smart Voucher Zielkorridor (40–60%)')
ax1.axvline(0.40, color=ACCENT_LEMON, linestyle=':', linewidth=1.5)
ax1.axvline(0.60, color=ACCENT_LEMON, linestyle=':', linewidth=1.5)
ax1.set_title('1. Vorhersage-Verteilung & Margenschutz-Zone', fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('Kaufwahrscheinlichkeit P(Kauf)', fontsize=10)
ax1.set_ylabel('Anzahl Shopper-Sessions', fontsize=10)
ax1.grid(True)
ax1.legend(loc='upper right', frameon=True, facecolor=SURFACE_COLOR, edgecolor=GRID_COLOR, fontsize=8)

# Panel 2: ROC-Kurve & Inferenz-Güte
ax2 = fig.add_subplot(gs[0, 1])
fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)
ax2.plot(fpr, tpr, color=ACCENT_LEMON, linewidth=2.5, label=f'MLP Modell (AUC = {roc_auc:.3f})')
ax2.plot([0, 1], [0, 1], color=TEXT_MUTED, linestyle='--', linewidth=1, label='Zufalls-Klassifikator (AUC = 0.500)')
ax2.fill_between(fpr, tpr, color=ACCENT_LEMON, alpha=0.15)
ax2.set_title(f'2. ROC-Kurve & Modellgüte (AUC: {roc_auc:.3f})', fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel('False Positive Rate (1 - Spezifität)', fontsize=10)
ax2.set_ylabel('True Positive Rate (Sensitivität)', fontsize=10)
ax2.grid(True)
ax2.legend(loc='lower right', frameon=True, facecolor=SURFACE_COLOR, edgecolor=GRID_COLOR, fontsize=8)

# Panel 3: Saisonaler Drift & Conversion-Rate
ax3 = fig.add_subplot(gs[1, 0])
df_q4 = df[df['Month'].isin(['Nov', 'Dec'])]
df_rest = df[~df['Month'].isin(['Nov', 'Dec'])]
cr_q4 = df_q4['Revenue'].mean() * 100
cr_rest = df_rest['Revenue'].mean() * 100

bars = ax3.bar(['Q1-Q3 (Regulär)', 'Q4 (Black Friday / Peak)'], [cr_rest, cr_q4], 
               color=['#253856', ACCENT_CYAN], width=0.5, edgecolor=GRID_COLOR)
ax3.set_ylim(0, 35)
ax3.set_title('3. Saisonaler Data Drift: Conversion-Rate Uplift', fontsize=12, fontweight='bold', pad=10)
ax3.set_ylabel('Conversion-Rate (%)', fontsize=10)
ax3.grid(axis='y')
for bar in bars:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, yval + 1.2, f"{yval:.1f} %", 
             ha='center', va='bottom', color=TEXT_MAIN, fontweight='bold', fontsize=11)

# Panel 4: Bot Detection via Isolation Forest
ax4 = fig.add_subplot(gs[1, 1])
numerics = df.select_dtypes(include=[np.number])
iso = IsolationForest(contamination=0.015, random_state=42)
anoms = iso.fit_predict(numerics)

normal = df[anoms == 1]
bots = df[anoms == -1]
ax4.scatter(normal['ProductRelated'], normal['ProductRelated_Duration'], color='#253856', alpha=0.45, s=20, label='Reale Shopper (98.5%)')
ax4.scatter(bots['ProductRelated'], bots['ProductRelated_Duration'], color=ACCENT_LEMON, marker='x', s=55, label='Erkannte Scraper-Bots (1.5%)')
ax4.set_title('4. Unsupervised Bot Cleaning (Isolation Forest)', fontsize=12, fontweight='bold', pad=10)
ax4.set_xlabel('Anzahl besuchter Produkte', fontsize=10)
ax4.set_ylabel('Verweildauer (Sekunden)', fontsize=10)
ax4.grid(True)
ax4.legend(loc='upper right', frameon=True, facecolor=SURFACE_COLOR, edgecolor=GRID_COLOR, fontsize=8)

# Speichern
out_path = base_dir / 'images' / 'ecom_real_data_master_dashboard.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Master Dashboard erfolgreich generiert: {out_path}")
