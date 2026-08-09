import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

os.makedirs('images', exist_ok=True)

C_DARKBLUE = '#20344f'
C_BLUEGREY = '#52627b'
C_LIGHTGREY = '#edeef0'
C_LEMON = '#e0ff01'
mpl.rcParams['figure.facecolor'] = C_DARKBLUE
mpl.rcParams['axes.facecolor'] = C_DARKBLUE
mpl.rcParams['axes.edgecolor'] = C_BLUEGREY
mpl.rcParams['axes.labelcolor'] = C_LIGHTGREY
mpl.rcParams['xtick.color'] = C_LIGHTGREY
mpl.rcParams['ytick.color'] = C_LIGHTGREY
mpl.rcParams['text.color'] = C_LIGHTGREY
mpl.rcParams['grid.color'] = C_BLUEGREY
mpl.rcParams['savefig.facecolor'] = C_DARKBLUE

df = pd.read_csv("online_shoppers_intention.csv")
df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop(columns=['Revenue'])
y = df_encoded['Revenue'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(16, 8), random_state=42, max_iter=300)
mlp.fit(X_train_scaled, y_train)

# 1. Proba
probs = mlp.predict_proba(X_test_scaled)[:, 1]
plt.figure(figsize=(9, 5))
plt.hist(probs, bins=40, color=C_BLUEGREY, edgecolor=C_DARKBLUE)
plt.axvspan(0.4, 0.6, color=C_LEMON, alpha=0.3)
plt.title("Verteilung der Kaufwahrscheinlichkeiten", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Wahrscheinlichkeit zu Kaufen (0 bis 1)", fontsize=12)
plt.ylabel("Anzahl Besucher", fontsize=12)
for spine in plt.gca().spines.values(): spine.set_visible(False)
plt.tight_layout()
plt.savefig('images/smart_vouchers.png', dpi=300)
plt.close()

# 2. Features
top_features = ['PageValues', 'ExitRates', 'ProductRelated_Duration']
X_top = df[top_features]
X_train_top, X_test_top, _, _ = train_test_split(X_top, y, test_size=0.2, random_state=42)
scaler_top = StandardScaler()
X_train_top_scaled = scaler_top.fit_transform(X_train_top)
X_test_top_scaled = scaler_top.transform(X_test_top)
mlp_top = MLPClassifier(hidden_layer_sizes=(16, 8), random_state=42, max_iter=300)
mlp_top.fit(X_train_top_scaled, y_train)
acc_full = accuracy_score(y_test, mlp.predict(X_test_scaled))
acc_top = accuracy_score(y_test, mlp_top.predict(X_test_top_scaled))

plt.figure(figsize=(7, 5))
bars = plt.bar(['Alle Features (>60)', 'Nur Top 3'], [acc_full, acc_top], color=[C_BLUEGREY, C_LEMON])
plt.ylim(0.75, 0.95)
plt.title("Genauigkeits-Vergleich (Accuracy)", fontsize=14, fontweight='bold', pad=15)
plt.ylabel("Accuracy", fontsize=12)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval*100:.1f}%", ha='center', color=C_LIGHTGREY, fontweight='bold')
for spine in plt.gca().spines.values(): spine.set_visible(False)
plt.tight_layout()
plt.savefig('images/feature_selection.png', dpi=300)
plt.close()

# 3. Seasonal
df_q4 = df[df['Month'].isin(['Nov', 'Dec'])]
df_rest = df[~df['Month'].isin(['Nov', 'Dec'])]
cr_q4 = df_q4['Revenue'].mean() * 100
cr_rest = df_rest['Revenue'].mean() * 100

plt.figure(figsize=(7, 5))
bars = plt.bar(['Rest des Jahres', 'Q4 (Nov/Dez)'], [cr_rest, cr_q4], color=[C_BLUEGREY, C_LEMON])
plt.title("Käufer-Anteil (Conversion Rate) im Vergleich", fontsize=14, fontweight='bold', pad=15)
plt.ylabel("Kaufwahrscheinlichkeit in %", fontsize=12)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.1f}%", ha='center', color=C_DARKBLUE if yval>15 else C_LIGHTGREY, fontweight='bold')
for spine in plt.gca().spines.values(): spine.set_visible(False)
plt.tight_layout()
plt.savefig('images/seasonal.png', dpi=300)
plt.close()

# 4. Bot Detection
numerics = df.select_dtypes(include=[np.number])
iso = IsolationForest(contamination=0.01, random_state=42)
anoms = iso.fit_predict(numerics)

plt.figure(figsize=(10, 6))
normal = df[anoms == 1]
plt.scatter(normal['ProductRelated'], normal['ProductRelated_Duration'], color=C_BLUEGREY, alpha=0.5, edgecolors='none')
bots = df[anoms == -1]
plt.scatter(bots['ProductRelated'], bots['ProductRelated_Duration'], color=C_LEMON, marker='x', s=60)
plt.title("Bot-Erkennung: Seitenaufrufe vs. Verweildauer", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Anzahl aufgerufener Produkte", fontsize=12)
plt.ylabel("Verweildauer bei Produkten (Sekunden)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
for spine in plt.gca().spines.values(): spine.set_visible(False)
plt.tight_layout()
plt.savefig('images/bot_detection.png', dpi=300)
plt.close()
