# RESUMEN: Notebooks Creados para Speed Dating Columbia University

## Archivos Generados

### 1. notebooks/01_preparacion_datos.ipynb (20 celdas)
**Propósito:** Preparación completa de los datos para análisis de citas rápidas.

**Características:**
- Carga de datos desde `data/Speed Dating Data.csv`
- Diccionario de variables explicado
- Pandas Profiling guardado en `reports/pandas_profiling.html`
- Distribución de la variable objetivo (match) con gráficos
- Estadística descriptiva completa
- Limpieza de nulos (eliminar >50%, imputar mediana el resto)
- Winsorizing al percentil 5-95 para scores
- Reducción de redundancia (correlación > 0.85) e irrelevancia (corr < 0.02 con match)
- Datos finales guardados en `data/data_prepared.csv`
- Todo en español con paleta rosa #FF1493

### 2. notebooks/02_modelamiento.ipynb (22 celdas)
**Propósito:** Entrenamiento, validación y selección de modelos predictivos.

**Características:**
- Carga de datos: X_train, X_test, y_train, y_test
- 7 modelos entrenados:
  * Árbol de Decisión
  * MLP (Red Neuronal)
  * SVM
  * KNN
  * Random Forest
  * XGBoost
  * Gradient Boosting
- Validación cruzada estratificada 10-fold
- Métricas: Accuracy, Precision, Recall, F1, ROC-AUC
- Tabla comparativa de modelos
- Curvas ROC superpuestas
- ANOVA + Tukey HSD sobre scores de CV
- Selección top 3 por ROC-AUC
- RandomizedSearchCV (n_iter=10, cv=5) para top 3
- BayesSearchCV (n_iter=20) para el mejor modelo
- Evaluación final en test
- Feature importance y análisis SHAP
- Pipeline guardado en `models/pipeline_match_predictor.pkl`
- Métricas finales en `models/metricas_finales.pkl`
- Todo en español con paleta rosa #FF1493

## Estructura del Proyecto

```
match-predictor/
├── notebooks/
│   ├── 01_preparacion_datos.ipynb  # Preprocesamiento
│   └── 02_modelamiento.ipynb       # Modelado
├── data/
│   ├── Speed Dating Data.csv        # Datos originales
│   ├── X_train.csv                  # Features train
│   ├── X_test.csv                   # Features test
│   ├── y_train.csv                  # Target train
│   ├── y_test.csv                   # Target test
│   ├── data_prepared.csv            # Datos procesados
│   ├── feature_names.pkl            # Nombres de features
│   └── scaler.pkl                   # StandardScaler
├── models/
│   ├── pipeline_match_predictor.pkl # Pipeline final
│   ├── metricas_finales.pkl         # Métricas del modelo
│   └── scaler.pkl                   # Scaler guardado
└── reports/
    ├── pandas_profiling.html        # Reporte exploratorio
    ├── distribucion_target.png      # Gráfico de distribución
    ├── matriz_correlacion_inicial.png
    ├── curvas_roc_comparativas.png
    ├── evaluacion_final.png
    ├── feature_importance.png
    └── shap_summary.png
```

## Paleta de Colores
- Principal: #FF1493 (Deep Pink)
- Secundarias: #FF69B4, #FFB6C1, #FFC0CB

## Requisitos Cumplidos ✅
- [x] Carga de datos preparados
- [x] 7 modelos entrenados
- [x] Validación cruzada estratificada 10-fold
- [x] Métricas completas (Accuracy, Precision, Recall, F1, ROC-AUC)
- [x] Tabla comparativa
- [x] Curvas ROC superpuestas
- [x] ANOVA + Tukey HSD
- [x] Top 3 por ROC-AUC
- [x] RandomizedSearchCV (n_iter=10, cv=5)
- [x] BayesSearchCV (n_iter=20)
- [x] Evaluación en test
- [x] Feature importance
- [x] SHAP
- [x] Pipeline guardado
- [x] Métricas guardadas
- [x] Todo en español
- [x] Paleta rosa #FF1493