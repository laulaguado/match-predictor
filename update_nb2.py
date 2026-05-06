import nbformat

# Notebook 2: 02_modelamiento.ipynb
nb2 = nbformat.v4.new_notebook()

cells2 = []

# Celda 1: Título
cells2.append(nbformat.v4.new_markdown_cell('# Modelamiento y Selección de Modelos - Speed Dating Columbia University\n\nEste notebook realiza el entrenamiento, validación y selección de modelos para predecir matches en citas rápidas.'))

# Celda 2: Imports
cells2.append(nbformat.v4.new_code_cell('''import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import os
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)

PALETA_ROSA = ["#FF1493", "#FF69B4", "#FFB6C1", "#FFC0CB", "#DB7093"]
sns.set_palette(PALETA_ROSA)

pd.set_option("display.max_columns", None)

print("Librerías importadas correctamente")
print("Paleta rosa principal: #FF1493")'''))

# Celda 3: Load data
cells2.append(nbformat.v4.new_markdown_cell('## 1. Carga de Datos Preparados'))

cells2.append(nbformat.v4.new_code_cell('''# Cargar datasets preprocesados
X_train = pd.read_csv("../data/X_train.csv")
X_test = pd.read_csv("../data/X_test.csv")
y_train = pd.read_csv("../data/y_train.csv").values.ravel()
y_test = pd.read_csv("../data/y_test.csv").values.ravel()

print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")

# Cargar feature names
with open("../data/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)
print(f"\nNúmero de features: {len(feature_names)}")

# Cargar scaler
with open("../data/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
print("Scaler cargado correctamente")

# Distribución del target
print(f"\nDistribución Train - Match: {sum(y_train)} ({sum(y_train)/len(y_train)*100:.1f}%)")
print(f"Distribución Test - Match: {sum(y_test)} ({sum(y_test)/len(y_test)*100:.1f}%)")

# Verificación
print(f"\nVerificación:")
print(f"  Features coincidentes: {set(X_train.columns) == set(X_test.columns)}")
print(f"  Nulos en X_train: {X_train.isnull().sum().sum()}")
print(f"  Nulos en X_test: {X_test.isnull().sum().sum()}")'''))

# Celda 4: Model training
cells2.append(nbformat.v4.new_markdown_cell('## 2. Entrenamiento de Modelos Base'))

cells2.append(nbformat.v4.new_code_cell('''from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time

# Definir 7 modelos
modelos = {
    "Árbol de Decisión": DecisionTreeClassifier(random_state=42),
    "MLP": MLPClassifier(random_state=42, max_iter=1000),
    "SVM": SVC(random_state=42, probability=True),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss", use_label_encoder=False),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# Validación cruzada estratificada 10-fold
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Métricas
scoring = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score),
    "recall": make_scorer(recall_score),
    "f1": make_scorer(f1_score),
    "roc_auc": make_scorer(roc_auc_score)
}

resultados = []
cv_scores_dict = {}

print("Entrenando 7 modelos con Validación Cruzada Estratificada 10-fold...")
print("=" * 80)

for nombre, modelo in modelos.items():
    print(f"\\n{nombre}...")
    inicio = time.time()
    cv_results = cross_validate(modelo, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)
    tiempo = time.time() - inicio
    
    resultado = {
        "Modelo": nombre,
        "Accuracy_mean": cv_results["test_accuracy"].mean(),
        "Accuracy_std": cv_results["test_accuracy"].std(),
        "Precision_mean": cv_results["test_precision"].mean(),
        "Precision_std": cv_results["test_precision"].std(),
        "Recall_mean": cv_results["test_recall"].mean(),
        "Recall_std": cv_results["test_recall"].std(),
        "F1_mean": cv_results["test_f1"].mean(),
        "F1_std": cv_results["test_f1"].std(),
        "ROC_AUC_mean": cv_results["test_roc_auc"].mean(),
        "ROC_AUC_std": cv_results["test_roc_auc"].std(),
        "Tiempo": tiempo
    }
    resultados.append(resultado)
    cv_scores_dict[nombre] = cv_results["test_roc_auc"]
    
    print(f"  Accuracy: {resultado['Accuracy_mean']:.4f} (+/- {resultado['Accuracy_std']:.4f})")
    print(f"  Precision: {resultado['Precision_mean']:.4f} (+/- {resultado['Precision_std']:.4f})")
    print(f"  Recall: {resultado['Recall_mean']:.4f} (+/- {resultado['Recall_std']:.4f})")
    print(f"  F1: {resultado['F1_mean']:.4f} (+/- {resultado['F1_std']:.4f})")
    print(f"  ROC-AUC: {resultado['ROC_AUC_mean']:.4f} (+/- {resultado['ROC_AUC_std']:.4f})")
    print(f"  Tiempo: {tiempo:.2f}s")

# Tabla comparativa
df_resultados = pd.DataFrame(resultados)
df_resultados = df_resultados.sort_values("ROC_AUC_mean", ascending=False)

print("\\n" + "=" * 80)
print("TABLA COMPARATIVA DE MODELOS")
print("=" * 80)
display_cols = ["Modelo", "Accuracy_mean", "Precision_mean", "Recall_mean", "F1_mean", "ROC_AUC_mean", "Tiempo"]
df_resultados[display_cols].style.background_gradient(cmap="RdBu_r", subset=["ROC_AUC_mean"])
'''))

# Celda 5: ROC curves
cells2.append(nbformat.v4.new_markdown_cell('## 3. Curvas ROC Comparativas'))

cells2.append(nbformat.v4.new_code_cell('''from sklearn.metrics import RocCurveDisplay

fig, ax = plt.subplots(figsize=(12, 10))

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    RocCurveDisplay.from_estimator(modelo, X_test, y_test, ax=ax, name=nombre)

ax.plot([0, 1], [0, 1], "k--", label="Aleatorio")
ax.set_xlabel("Tasa de Falsos Positivos", fontsize=12)
ax.set_ylabel("Tasa de Verdaderos Positivos", fontsize=12)
ax.set_title("Curvas ROC - Comparación de Modelos", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)

os.makedirs("../reports", exist_ok=True)
plt.savefig("../reports/curvas_roc_comparativas.png", dpi=300, bbox_inches="tight")
print("✓ Curvas ROC guardadas en ../reports/curvas_roc_comparativas.png")
plt.show()
'''))

# Celda 6: ANOVA
cells2.append(nbformat.v4.new_markdown_cell('## 4. ANOVA y Tukey HSD sobre Scores de CV'))

cells2.append(nbformat.v4.new_code_cell('''from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Preparar datos para ANOVA
cv_scores_list = []
for nombre, scores in cv_scores_dict.items():
    for score in scores:
        cv_scores_list.append({"Modelo": nombre, "ROC_AUC": score})

df_cv = pd.DataFrame(cv_scores_list)

# ANOVA
grupos = [df_cv[df_cv["Modelo"] == m]["ROC_AUC"].values for m in df_resultados["Modelo"]]
stat, p_valor = stats.f_oneway(*grupos)

print("=" * 80)
print("ANOVA - Diferencias entre modelos")
print("=" * 80)
print(f"Estadístico F: {stat:.4f}")
print(f"Valor p: {p_valor:.6f}")

if p_valor < 0.05:
    print("\\nResultado: Diferencias significativas entre modelos (p < 0.05)")
else:
    print("\\nResultado: No hay diferencias significativas (p >= 0.05)")

# Tukey HSD
tukey = pairwise_tukeyhsd(df_cv["ROC_AUC"], df_cv["Modelo"], alpha=0.05)
print("\\n" + "=" * 80)
print("PRUEBA TUKEY HSD - Comparaciones por pares")
print("=" * 80)
print(tukey)

# Guardar resultados
with open("../reports/anova_results.txt", "w") as f:
    f.write("ANOVA Results\\n")
    f.write("=" * 80 + "\\n")
    f.write(f"F-statistic: {stat:.4f}\\n")
    f.write(f"p-value: {p_valor:.6f}\\n\\n")
    f.write(str(tukey))
print("\\n✓ Resultados guardados en ../reports/anova_results.txt")
'''))

# Celda 7: Top 3
cells2.append(nbformat.v4.new_markdown_cell('## 5. Selección de Top 3 Modelos por ROC-AUC'))

cells2.append(nbformat.v4.new_code_cell('''# Seleccionar top 3
top3 = df_resultados.nlargest(3, "ROC_AUC_mean")
print("=" * 80)
print("TOP 3 MODELOS POR ROC-AUC")
print("=" * 80)
display(top3[display_cols])

top3_nombres = top3["Modelo"].tolist()
print(f"\\nModelos seleccionados para tuning: {top3_nombres}")

top3_modelos = {nombre: modelos[nombre] for nombre in top3_nombres}
print("\\nInicializando hiperparámetros para tuning...")
'''))

# Celda 8: RandomizedSearchCV
cells2.append(nbformat.v4.new_markdown_cell('## 6. Hiperparametrización - RandomizedSearchCV'))

cells2.append(nbformat.v4.new_code_cell('''# RandomizedSearchCV para top 3
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV

parametros_rf = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None]
}

parametros_xgb = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.3],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0]
}

parametros_gb = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.3],
    "subsample": [0.8, 0.9, 1.0]
}

# Asignar parámetros según modelo
param_grids = {}
for nombre in top3_nombres:
    if "Random Forest" in nombre:
        param_grids[nombre] = parametros_rf
    elif "XGBoost" in nombre:
        param_grids[nombre] = parametros_xgb
    elif "Gradient Boosting" in nombre:
        param_grids[nombre] = parametros_gb
    else:
        param_grids[nombre] = {}

print("Realizando RandomizedSearchCV (n_iter=10, cv=5)...")
print("=" * 80)

mejores_modelos_rs = {}
resultados_rs = []

for nombre in top3_nombres:
    print(f"\\n{nombre}...")
    rs = RandomizedSearchCV(
        top3_modelos[nombre],
        param_grids[nombre],
        n_iter=10,
        cv=5,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1
    )
    rs.fit(X_train, y_train)
    mejores_modelos_rs[nombre] = rs.best_estimator_
    resultados_rs.append({
        "Modelo": nombre,
        "Mejor ROC-AUC": rs.best_score_,
        "Mejores Parametros": str(rs.best_params_)
    })
    print(f"  Mejor ROC-AUC: {rs.best_score_:.4f}")
    print(f"  Mejores parámetros: {rs.best_params_}")

df_rs = pd.DataFrame(resultados_rs)

# Seleccionar el mejor
mejor_nombre = df_rs.loc[df_rs["Mejor ROC-AUC"].idxmax(), "Modelo"]
mejor_modelo_rs = mejores_modelos_rs[mejor_nombre]

print(f"\\n" + "=" * 80)
print(f"MEJOR MODELO (RandomizedSearchCV): {mejor_nombre}")
print("=" * 80)
'''))

# Celda 9: BayesSearchCV
cells2.append(nbformat.v4.new_markdown_cell('## 7. BayesSearchCV para el Mejor Modelo'))

cells2.append(nbformat.v4.new_code_cell('''# BayesSearchCV
print("Realizando BayesSearchCV (n_iter=20)...")
print("=" * 80)

from skopt.space import Real, Integer, Categorical
from skopt import BayesSearchCV

# Definir espacio de búsqueda bayesiano según el mejor modelo
if "Random Forest" in mejor_nombre:
    busqueda_bayesiana = {
        "n_estimators": Integer(50, 300),
        "max_depth": Integer(5, 50),
        "min_samples_split": Integer(2, 20),
        "min_samples_leaf": Integer(1, 10),
        "max_features": Categorical(["sqrt", "log2", None])
    }
elif "XGBoost" in mejor_nombre:
    busqueda_bayesiana = {
        "n_estimators": Integer(50, 300),
        "max_depth": Integer(3, 10),
        "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
        "subsample": Real(0.7, 1.0),
        "colsample_bytree": Real(0.7, 1.0)
    }
elif "Gradient Boosting" in mejor_nombre:
    busqueda_bayesiana = {
        "n_estimators": Integer(50, 300),
        "max_depth": Integer(3, 10),
        "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
        "subsample": Real(0.7, 1.0)
    }
else:
    busqueda_bayesiana = {}

if busqueda_bayesiana:
    bayes_search = BayesSearchCV(
        mejor_modelo_rs,
        busqueda_bayesiana,
        n_iter=20,
        cv=5,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1
    )
    bayes_search.fit(X_train, y_train)
    mejor_modelo_final = bayes_search.best_estimator_
    print(f"\\nMejor ROC-AUC (BayesSearchCV): {bayes_search.best_score_:.4f}")
    print(f"Mejores parámetros: {bayes_search.best_params_}")
else:
    mejor_modelo_final = mejor_modelo_rs
    print("No se aplicó BayesSearchCV (modelo no compatible)")
    print(f"Modelo final: {mejor_nombre}")
'''))

# Celda 10: Evaluación
cells2.append(nbformat.v4.new_markdown_cell('## 8. Evaluación del Modelo Final en Test'))

cells2.append(nbformat.v4.new_code_cell('''# Evaluar en test
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib

y_pred = mejor_modelo_final.predict(X_test)
y_pred_proba = mejor_modelo_final.predict_proba(X_test)[:, 1]

# Métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("=" * 80)
print("EVALUACIÓN DEL MODELO FINAL EN TEST")
print("=" * 80)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Match", "Match"]))

# Gráficos
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="RdBu_r", ax=axes[0],
            xticklabels=["No Match", "Match"],
            yticklabels=["No Match", "Match"])
axes[0].set_title("Matriz de Confusión", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Real", fontsize=12)
axes[0].set_xlabel("Predicción", fontsize=12)

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[1].plot(fpr, tpr, color="#FF1493", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
axes[1].plot([0, 1], [0, 1], "k--", lw=2, label="Aleatorio")
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel("Tasa de Falsos Positivos", fontsize=12)
axes[1].set_ylabel("Tasa de Verdaderos Positivos", fontsize=12)
axes[1].set_title("Curva ROC - Modelo Final", fontsize=14, fontweight="bold")
axes[1].legend(loc="lower right", fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../reports/evaluacion_final.png", dpi=300, bbox_inches="tight")
print("\\n✓ Gráficos guardados en ../reports/evaluacion_final.png")
plt.show()
'''))

# Celda 11: Feature importance
cells2.append(nbformat.v4.new_markdown_cell('## 9. Importancia de Features'))

cells2.append(nbformat.v4.new_code_cell('''# Feature Importance
if hasattr(mejor_modelo_final, "feature_importances_"):
    importancia = pd.DataFrame({
        "Feature": feature_names,
        "Importancia": mejor_modelo_final.feature_importances_
    }).sort_values("Importancia", ascending=False)
    
    print("Top 20 features más importantes:")
    display(importancia.head(20).style.background_gradient(cmap="Reds_r", subset=["Importancia"]))
    
    # Gráfico
    fig, ax = plt.subplots(figsize=(12, 8))
    top_features = importancia.head(20)
    ax.barh(range(len(top_features)), top_features["Importancia"].values, color="#FF1493")
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["Feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Importancia", fontsize=12)
    ax.set_title("Top 20 Features - Importancia", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig("../reports/feature_importance.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    print("\\n✓ Gráfico guardado en reports/feature_importance.png")
else:
    print("El modelo no tiene atributo feature_importances_")
'''))

# Celda 12: Guardar pipeline
cells2.append(nbformat.v4.new_markdown_cell('## 10. Guardado del Pipeline Final'))

cells2.append(nbformat.v4.new_code_cell('''# Guardar pipeline y métricas
os.makedirs("../models", exist_ok=True)

# Pipeline
pipeline = mejor_modelo_final
joblib.dump(pipeline, "../models/pipeline_match_predictor.pkl")
print("✓ Pipeline guardado en ../models/pipeline_match_predictor.pkl")

# Métricas finales
metricas_finales = {
    "modelo": mejor_nombre,
    "accuracy_test": float(accuracy),
    "precision_test": float(precision),
    "recall_test": float(recall),
    "f1_test": float(f1),
    "roc_auc_test": float(roc_auc),
    "cv_roc_auc_mean": float(df_rs.loc[df_rs["Modelo"] == mejor_nombre, "Mejor ROC-AUC"].values[0]),
    "feature_names": feature_names,
    "n_features": len(feature_names),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "train_balance": float(y_train.mean()),
    "test_balance": float(y_test.mean()),
    "mejores_parametros": bayes_search.best_params_ if "bayes_search" in locals() else rs.best_params_
}

with open("../models/metricas_finales.pkl", "wb") as f:
    pickle.dump(metricas_finales, f)
print("✓ Métricas finales guardadas en ../models/metricas_finales.pkl")

# Guardar también como JSON
metricas_json = metricas_finales.copy()
metricas_json["feature_names"] = metricas_json["feature_names"][:10]
with open("../models/metricas_finales.json", "w") as f:
    import json
    json.dump({k: v for k, v in metricas_json.items() if k not in ["feature_names", "mejores_parametros"]}, f, indent=2, default=str)
print("✓ Métricas finales guardadas en ../models/metricas_finales.json")

print("\\n" + "=" * 80)
print("PROCESO COMPLETADO SATISFACTORIAMENTE")
print("=" * 80)
print(f"\\nMejor modelo: {mejor_nombre}")
print(f"ROC-AUC Test: {roc_auc:.4f}")
print(f"Accuracy Test: {accuracy:.4f}")
print(f"\\nArchivos generados:")
print("  - models/pipeline_match_predictor.pkl")
print("  - models/metricas_finales.pkl")
print("  - models/metricas_finales.json")
print("  - reports/curvas_roc_comparativas.png")
print("  - reports/evaluacion_final.png")
print("  - reports/feature_importance.png")
'''))

nb2.cells = cells2

with open("notebooks/02_modelamiento.ipynb", "w") as f:
    nbformat.write(nb2, f)

print("✓ Notebook 02 actualizado correctamente")
"