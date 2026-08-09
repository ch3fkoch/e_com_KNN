# E-Com KNN: Conversion Prediction

Dieses Projekt demonstriert die Nutzung von Maschinellem Lernen (Künstliche Neuronale Netze / Multi-Layer Perceptron), um im E-Commerce die Kaufwahrscheinlichkeit von Webseiten-Besuchern vorherzusagen.

## 📖 Inhalt

Das Projekt besteht aus drei aufeinander aufbauenden Jupyter Notebooks, die ideal zur Vorbereitung auf die **Scikit-Learn Zertifizierung** geeignet sind:

1. **`ecommerce_conversion_knn.ipynb`**
   - Einstieg: KNN mit synthetisch generierten Daten.
   - Grundlagen: Train-Test-Split, StandardScaler, MLPClassifier.
2. **`ecommerce_real_data_knn.ipynb`**
   - Reale Daten: Nutzung des "Online Shoppers Purchasing Intention" Datensatzes.
   - Auswahl numerischer Features und Evaluation.
3. **`ecommerce_advanced_knn.ipynb`**
   - Fortgeschritten: Kategoriale Daten verarbeiten mit One-Hot-Encoding (`pd.get_dummies`).
   - Hyperparameter-Tuning: Automatische Modell-Optimierung via `GridSearchCV`.
4. **`ecommerce_visualization_knn.ipynb`**
   - Visualisierung: Lernkurve, Heatmap, ROC-Kurve und Feature Importance.
5. **`ecommerce_business_usecases.ipynb`**
   - Business Cases: Smart Vouchers (`predict_proba`), Feature Selection, Saisonale Splitts & Bot-Detection (`IsolationForest`).

## 🚀 Setup & Ausführung

Stelle sicher, dass die benötigten Bibliotheken installiert sind:

```bash
pip install -r requirements.txt
```

## 🔮 Ausblick: Der Schritt zu First-Party Data (User-Accounts)

Aktuell basiert das Modell auf anonymen Session-Daten (Cookies). Für ein echtes Produktionssystem im modernen E-Commerce ist der Wechsel auf eingeloggte Benutzer (User-Accounts) der nächste große Hebel.

**Warum First-Party Data das Machine Learning revolutioniert:**
1. **Cross-Device Tracking:** Eine zusammenhängende Customer Journey über Handy und Laptop hinweg wird erkennbar, statt als zwei abbrechende Sessions gewertet zu werden.
2. **Historische Features:** Das KNN kann mit mächtigen neuen Spalten trainiert werden:
   - `Tage_seit_letztem_Kauf`
   - `Historischer_Durchschnitts_Warenkorb`
   - `Retourenquote`
   - `Wunschzettel_Aktivität`
3. **DSGVO & Tracking-Resilienz:** Eingeloggte User bieten saubere, datenschutzkonforme Datenpunkte, die unabhängig von Cookie-Blockern (z.B. Apple ITP) funktionieren.

Starte danach einfach Jupyter Notebook oder öffne die Dateien direkt in deiner IDE.
