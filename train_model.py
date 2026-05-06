import pandas as pd
import numpy as np
import pickle
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_validate, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer

print("="*80)
print("STEP 1: Cargar datos preparados")
print("="*80)

X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').values.ravel()
y_test = pd.read_csv('data/y_test.csv').values.ravel()

with open('data/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

with open('data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"Features: {len(feature_names)}")
print(f"y_train - Match rate: {y_train.mean():.3f}")
print(f"y_test - Match rate: {y_test.mean():.3f}")
print("Scaler cargado correctamente")

# ============================================================================
# STEP 2: Verificación anti-leakage
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Verificación anti-leakage")
print("="*80)

LEAKAGE_COLS = ['dec','dec_o','like','prob','like_o','prob_o',
                'attr_o','sinc_o','intel_o','fun_o','amb_o','shar_o',
                'int_corr','match_es']

encontradas = [c for c in LEAKAGE_COLS if c in X_train.columns]
if encontradas:
    print("🚨 LEAKAGE DETECTADO: %s" % encontradas)
    print("   Detener y corregir antes de continuar")
    raise ValueError("LEAKAGE en X_train — ver columnas arriba")
else:
    print("✅ Sin leakage detectado")
    print("   Columnas en X_train: %d" % X_train.shape[1])
    print("   Train size: %d | Test size: %d" % (len(X_train), len(X_test)))
    print("   Train match rate: %.3f (SMOTE)" % y_train.mean())
    print("   Test match rate:  %.3f (real)"  % y_test.mean())

# ============================================================================
# STEP 3: Entrenar múltiples modelos con cross-validation
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Entrenar múltiples modelos con 10-fold CV")
print("="*80)

models = {
    "Arbol": DecisionTreeClassifier(random_state=42),
    "MLP": MLPClassifier(random_state=42, max_iter=1000),
    "SVM": SVC(random_state=42, probability=True),
    "KNN": KNeighborsClassifier(),
    "RF": RandomForestClassifier(random_state=42),
    "XGB": XGBClassifier(random_state=42, eval_metric="logloss", use_label_encoder=False),
    "GB": GradientBoostingClassifier(random_state=42)
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score),
    "recall": make_scorer(recall_score),
    "f1": make_scorer(f1_score),
    "roc_auc": make_scorer(roc_auc_score)
}

from sklearn.metrics import make_scorer

results = []
cv_scores = {}

print("Entrenando 7 modelos 10-fold CV...")
for name, model in models.items():
    print("  %s..." % name)
    cv_result = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)
    
    results.append({
        "Modelo": name,
        "Accuracy_mean": cv_result["test_accuracy"].mean(),
        "Accuracy_std": cv_result["test_accuracy"].std(),
        "Precision_mean": cv_result["test_precision"].mean(),
        "Precision_std": cv_result["test_precision"].std(),
        "Recall_mean": cv_result["test_recall"].mean(),
        "Recall_std": cv_result["test_recall"].std(),
        "F1_mean": cv_result["test_f1"].mean(),
        "F1_std": cv_result["test_f1"].std(),
        "ROC_AUC_mean": cv_result["test_roc_auc"].mean(),
        "ROC_AUC_std": cv_result["test_roc_auc"].std()
    })
    cv_scores[name] = cv_result["test_roc_auc"]
    print("    ROC-AUC: %.4f" % results[-1]["ROC_AUC_mean"])

df_results = pd.DataFrame(results).sort_values("ROC_AUC_mean", ascending=False)

print("\nTabla comparativa:")
print(df_results[["Modelo","Accuracy_mean","Precision_mean","Recall_mean","F1_mean","ROC_AUC_mean"]].to_string(index=False))

# ============================================================================
# STEP 4: Top 3 - RandomizedSearchCV y BayesSearchCV
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Hyperparameter tuning para top 3 modelos")
print("="*80)

top3 = df_results.nlargest(3, "ROC_AUC_mean")
top3_names = top3["Modelo"].tolist()
top3_models = {n: models[n] for n in top3_names}

print(f"\nModelos top 3: {top3_names}")

# RandomizedSearchCV
print("\n" + "-"*40)
print("RandomizedSearchCV (n_iter=10, cv=5)")
print("-"*40)

param_grids = {}
for name in top3_names:
    if "RF" in name:
        param_grids[name] = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    elif "XGB" in name:
        param_grids[name] = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3]
        }
    elif "GB" in name:
        param_grids[name] = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3]
        }
    elif "SVM" in name:
        param_grids[name] = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'poly']
        }
    else:
        param_grids[name] = {}

random_search_models = {}
random_search_results = []

for name in top3_names:
    print("  %s..." % name)
    if param_grids[name]:
        rs = RandomizedSearchCV(
            top3_models[name], param_grids[name],
            n_iter=10, cv=5, scoring="roc_auc",
            random_state=42, n_jobs=-1
        )
        rs.fit(X_train, y_train)
        random_search_models[name] = rs.best_estimator_
        random_search_results.append({
            "Modelo": name,
            "ROC-AUC": rs.best_score_,
            "Params": str(rs.best_params_)
        })
        print("    ROC-AUC: %.4f" % rs.best_score_)
    else:
        random_search_models[name] = top3_models[name]
        random_search_results.append({
            "Modelo": name,
            "ROC-AUC": 0,
            "Params": "N/A"
        })

df_rs = pd.DataFrame(random_search_results)
best_name = df_rs.loc[df_rs["ROC-AUC"].idxmax(), "Modelo"]
best_model = random_search_models[best_name]

print("\nMejor modelo de RandomizedSearchCV: %s" % best_name)

# BayesSearchCV
print("\n" + "-"*40)
print("BayesSearchCV (n_iter=20, cv=5)")
print("-"*40)

if "RF" in best_name:
    bayes_space = {
        'n_estimators': Integer(50, 300),
        'max_depth': Integer(5, 50),
        'min_samples_split': Integer(2, 20)
    }
elif "XGB" in best_name:
    bayes_space = {
        'n_estimators': Integer(50, 300),
        'max_depth': Integer(3, 10),
        'learning_rate': Real(0.01, 0.3, prior='log-uniform')
    }
elif "GB" in best_name:
    bayes_space = {
        'n_estimators': Integer(50, 300),
        'max_depth': Integer(3, 10),
        'learning_rate': Real(0.01, 0.3, prior='log-uniform')
    }
elif "SVM" in best_name:
    bayes_space = {
        'C': Real(0.1, 10, prior='log-uniform'),
        'gamma': Real(0.001, 1, prior='log-uniform')
    }
else:
    bayes_space = {}

if bayes_space:
    bayes_search = BayesSearchCV(
        best_model, bayes_space,
        n_iter=20, cv=5, scoring="roc_auc",
        random_state=42, n_jobs=-1
    )
    bayes_search.fit(X_train, y_train)
    final_model = bayes_search.best_estimator_
    print("  ROC-AUC (Bayes): %.4f" % bayes_search.best_score_)
    print("  Mejores params: %s" % str(bayes_search.best_params_))
else:
    final_model = best_model
    print("  Sin tuning para este modelo")

# ============================================================================
# STEP 5: Evaluación en test set
# ============================================================================
print("\n" + "="*80)
print("STEP 5: Evaluación en test set")
print("="*80)

y_pred = final_model.predict(X_test)
y_prob = final_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\nMétricas en test set:")
print("  Accuracy:  %.4f" % acc)
print("  Precision: %.4f" % pre)
print("  Recall:    %.4f" % rec)
print("  F1:        %.4f" % f1)
print("  ROC-AUC:   %.4f" % auc)

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=["No Match", "Match"]))

# Verificar que ROC-AUC no sea cercano a 1.0 (overfitting/leakage)
if auc > 0.95:
    print("\n⚠️  ADVERTENCIA: ROC-AUC muy alto (%.4f). Revisar posible leakage." % auc)
else:
    print("\n✓ ROC-AUC en rango esperado (%.4f)" % auc)

# ============================================================================
# STEP 6: Guardar modelo y métricas
# ============================================================================
print("\n" + "="*80)
print("STEP 6: Guardar modelo y métricas")
print("="*80)

os.makedirs('models', exist_ok=True)

# Guardar pipeline
model_filename = 'models/pipeline_match_predictor.pkl'
joblib.dump(final_model, model_filename)
print(f"Modelo guardado: {model_filename}")

# Guardar métricas
metricas_finales = {
    "modelo": best_name,
    "modelo_completo": str(type(final_model).__name__),
    "accuracy_test": float(acc),
    "precision_test": float(pre),
    "recall_test": float(rec),
    "f1_test": float(f1),
    "roc_auc_test": float(auc),
    "n_features": len(feature_names),
    "cv_results": results,
    "random_search_results": random_search_results,
    "feature_names": feature_names
}

with open('models/metricas_finales.pkl', 'wb') as f:
    pickle.dump(metricas_finales, f)
print("Métricas guardadas: models/metricas_finales.pkl")

# Guardar resultados como CSV
df_results.to_csv('models/cv_results.csv', index=False)
df_rs.to_csv('models/random_search_results.csv', index=False)
print("Resultados CV: models/cv_results.csv")
print("Resultados RandomSearch: models/random_search_results.csv")

print("\n" + "="*80)
print("PROCESO COMPLETADO")
print("="*80)
print(f"Modelo final: {best_name}")
print(f"ROC-AUC test: {auc:.4f}")
print(f"Accuracy test: {acc:.4f}")
print("="*80)