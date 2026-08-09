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

Starte danach einfach Jupyter Notebook oder öffne die Dateien direkt in deiner IDE.
