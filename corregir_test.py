import nbformat

with open('notebooks/02_modelamiento.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)

print("=== VERIFICACION IMPORTS (celda 2) ===")
cell_imports = nb.cells[1].source
imports_requeridos = ['classification_report', 'confusion_matrix', 'roc_curve', 'joblib']
for imp in imports_requeridos:
    if imp in cell_imports:
        print(f"✓ {imp} - OK")
    else:
        print(f"✗ {imp} - FALTA")

print("\n=== BUSCANDO SECCION 8 (EVALUACION EN TEST) ===")
for i, cell in enumerate(nb.cells):
    if hasattr(cell, 'source'):
        src = cell.source
        if 'confusion_matrix' in src and 'savefig' in src and 'roc_auc_score' in src:
            print(f"Encontrada celda {i+1}: EVALUACION EN TEST")
            print("Contenido actual:")
            print(src[:500])
            break

print("\n=== BUSCANDO SECCION 10 (GUARDAR PIPELINE) ===")
for i, cell in enumerate(nb.cells):
    if hasattr(cell, 'source'):
        src = cell.source
        if 'joblib.dump' in src and 'pipeline_match_predictor' in src:
            print(f"Encontrada celda {i+1}: GUARDAR PIPELINE")
            print("Contenido actual:")
            print(src)
            break

print("\n=== APLICANDO CORRECCIONES ===")

# 1. Verificar/agregar imports
if not all(imp in cell_imports for imp in imports_requeridos):
    print("Agregando imports faltantes a celda 2...")
    # Reemplazar la celda de imports con la versión corregida
    nuevos_imports = '''import matplotlib
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
import os
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)
PALETA_ROSA = ["#FF1493", "#FF69B4", "#FFB6C1", "#FFC0CB", "#DB7093"]
sns.set_palette(PALETA_ROSA)
pd.set_option("display.max_columns", None)
print("Librerias importadas")'''
    nb.cells[1].source = nuevos_imports
    print("✓ Imports corregidos")
else:
    print("✓ Todos los imports ya están presentes")

# 2. Y 3. Corregir celda de guardar pipeline (seccion 10)
for i, cell in enumerate(nb.cells):
    if hasattr(cell, 'source'):
        src = cell.source
        if 'joblib.dump' in src and 'pipeline_match_predictor' in src:
            print(f"\nCorrigiendo celda {i+1} (GUARDAR PIPELINE)...")
            nuevo_contenido = '''# Guardar pipeline y metricas
os.makedirs("../models", exist_ok=True)

joblib.dump(mfinal, "../models/pipeline_match_predictor.pkl")
print(f"Modelo guardado: {mn}")
print(f"ROC-AUC test: {auc:.4f}")
print(f"Accuracy test: {acc:.4f}")
print(f"Features usadas: {len(fn)}")
print("\\nNOTA: Si ROC-AUC es cercano a 1.0, revisar leakage en variables dec/dec_o/like/prob")

metricas_finales = {
    "modelo": mn,
    "accuracy": float(acc),
    "precision": float(pre),
    "recall": float(rec),
    "f1": float(f1),
    "roc_auc": float(auc),
    "n_features": len(fn)
}

with open("../models/metricas_finales.pkl", "wb") as f:
    pickle.dump(metricas_finales, f)
print("Metricas guardadas: models/metricas_finales.pkl")

print("\\n" + "=" * 80)
print("PROCESO COMPLETADO")
print("=" * 80)
print("Modelo: %s" % mn)
print("ROC-AUC test: %.4f" % auc)
print("Accuracy test: %.4f" % acc)'''
            nb.cells[i].source = nuevo_contenido
            print("✓ Celda corregida")
            break

# Guardar
with open('notebooks/02_modelamiento.ipynb', 'w') as f:
    nbformat.write(nb, f)

print("\n=== NOTEBOOK ACTUALIZADO ===")
