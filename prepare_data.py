import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: Cargar datos
# ============================================================================
print("="*80)
print("STEP 1: Cargar datos")
print("="*80)

df = pd.read_csv('data/Speed Dating Data.csv')
print(f"Dimensiones originales: {df.shape}")

# ============================================================================
# STEP 2: Eliminar IDs y leakage
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Eliminar IDs y leakage")
print("="*80)

# Lista exacta de variables de leakage
LEAKAGE = ['dec','dec_o','like','prob','attr','sinc','intel','fun','amb','shar',
           'attr_o','sinc_o','intel_o','fun_o','amb_o','shar_o','int_corr','met',
           'like_o','prob_o','match_es']

# Variables de identificación a eliminar
ID_VARS = ['iid','id','idg','partner','pid','wave','round','position','positin1','order']

# Eliminar variables
cols_to_drop = ID_VARS + LEAKAGE
cols_to_drop = [c for c in cols_to_drop if c in df.columns]
df = df.drop(columns=cols_to_drop)
print(f"Variables eliminadas: {cols_to_drop}")
print(f"Dimensiones después de eliminar leakage: {df.shape}")

# ============================================================================
# STEP 3: Preparar target y features - Usar solo Time 1 features
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Preparar target y features (Time 1)")
print("="*80)

# Separar target (match) - está en el dataset original
y = df['match'].copy()

# Time 1 features - las variables de evaluación iniciales
# (attr1_1, sinc1_1, etc.) y características demográficas
# Excluir variables de Time 2 (attr, sinc, etc. que son promedios)
TIME2_VARS = ['attr','sinc','intel','fun','amb','shar']  # Estas ya fueron eliminadas como leakage

# Seleccionar features: Time 1 + demográficas + hobbies
# Las features principales de Time 1 son:
FEATURE_COLS = [col for col in df.columns if col not in ['match']]

# Limpiar: eliminar filas con target NaN
valid_mask = y.notna()
df = df[valid_mask]
y = y[valid_mask]

print(f"Target distribution - Match: {y.mean():.3f}, No Match: {1-y.mean():.3f}")
print(f"Total samples: {len(y)}")

# ============================================================================
# STEP 4: Imputar nulos (median para numéricas)
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Imputar nulos (median para numéricas)")
print("="*80)

# Guardar medianas para imputación
medians = {}
for col in FEATURE_COLS:
    if col in df.columns and df[col].isna().any():
        median_val = df[col].median()
        medians[col] = median_val
        df[col] = df[col].fillna(median_val)
        print(f"  {col}: imputado con mediana={median_val:.2f}")

print(f"Total features: {len(FEATURE_COLS)}")
print(f"Nulos restantes: {df[FEATURE_COLS].isna().sum().sum()}")

# ============================================================================
# STEP 5: Winsorizing (5-95 percentiles)
# ============================================================================
print("\n" + "="*80)
print("STEP 5: Winsorizing (5-95 percentiles)")
print("="*80)

for col in FEATURE_COLS:
    if col in df.columns:
        p5 = df[col].quantile(0.05)
        p95 = df[col].quantile(0.95)
        df[col] = df[col].clip(lower=p5, upper=p95)

print("Winsorizing completado para todas las features")

# ============================================================================
# STEP 6: train_test_split 70/30 estratificado
# ============================================================================
print("\n" + "="*80)
print("STEP 6: train_test_split 70/30 estratificado")
print("="*80)

X = df[FEATURE_COLS].copy()

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print(f"X_train: {X_train_raw.shape}")
print(f"X_test: {X_test_raw.shape}")
print(f"y_train - Match rate: {y_train.mean():.3f}")
print(f"y_test - Match rate: {y_test.mean():.3f}")

# ============================================================================
# STEP 7: Scaler - fit SOLO en train, transform en test
# ============================================================================
print("\n" + "="*80)
print("STEP 7: Scaler - fit en train, transform en test")
print("="*80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=FEATURE_COLS, index=X_train_raw.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=FEATURE_COLS, index=X_test_raw.index)

print("Scaler aplicado correctamente")
print(f"X_train escalado - mean: {X_train_scaled.mean().mean():.6f}, std: {X_train_scaled.std().mean():.4f}")
print(f"X_test escalado - mean: {X_test_scaled.mean().mean():.6f}, std: {X_test_scaled.std().mean():.4f}")

# ============================================================================
# STEP 8: SMOTE SOLO en X_train - AL FINAL balancear
# ============================================================================
print("\n" + "="*80)
print("STEP 8: SMOTE SOLO en X_train")
print("="*80)

smote = SMOTE(random_state=42)
X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train)

X_train_final = pd.DataFrame(X_train_final, columns=FEATURE_COLS)
y_train_final = pd.Series(y_train_final)

print(f"X_train antes de SMOTE: {X_train_scaled.shape}")
print(f"X_train después de SMOTE: {X_train_final.shape}")
print(f"y_train antes de SMOTE - Match rate: {y_train.mean():.3f}")
print(f"y_train después de SMOTE - Match rate: {y_train_final.mean():.3f}")

# ============================================================================
# STEP 9: Verificaciones
# ============================================================================
print("\n" + "="*80)
print("STEP 9: Verificaciones")
print("="*80)

assert abs(y_train_final.mean() - 0.5) < 0.01, f"SMOTE no aplicado - mean: {y_train_final.mean()}"
print(f"✓ assert y_train.mean() == 0.5 - SMOTE aplicado correctamente (mean={y_train_final.mean():.3f})")

assert 0.14 < y_test.mean() < 0.20, f"y_test corrupto - mean: {y_test.mean()}"
print(f"✓ assert 0.14 < y_test.mean() < 0.20 (mean={y_test.mean():.3f})")

assert X_train_final.mean().abs().mean() < 0.1, f"Scaler mal aplicado - mean: {X_train_final.mean().abs().mean()}"
print(f"✓ assert X_train.mean().mean() < 0.1 (mean={X_train_final.mean().abs().mean():.6f})")

# ============================================================================
# STEP 10: Guardar archivos
# ============================================================================
print("\n" + "="*80)
print("STEP 10: Guardar archivos")
print("="*80)

X_train_final.to_csv('data/X_train.csv', index=False)
X_test_scaled.to_csv('data/X_test.csv', index=False)
y_train_final.to_csv('data/y_train.csv', index=False, header=True)
y_test.to_csv('data/y_test.csv', index=False, header=True)

with open('data/feature_names.pkl', 'wb') as f:
    pickle.dump(FEATURE_COLS, f)

with open('data/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Guardar medianas también
with open('data/medianas_imputadas.pkl', 'wb') as f:
    pickle.dump(medians, f)

print("Archivos guardados:")
print("  - data/X_train.csv")
print("  - data/X_test.csv")
print("  - data/y_train.csv")
print("  - data/y_test.csv")
print("  - data/feature_names.pkl")
print("  - data/scaler.pkl")
print("  - data/medianas_imputadas.pkl")

print("\n" + "="*80)
print("PROCESO COMPLETADO EXITOSAMENTE")
print("="*80)