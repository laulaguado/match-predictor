# Predictor de Match - Speed Dating Columbia University

## Descripción del Proyecto

Este proyecto implementa un sistema de machine learning para predecir la probabilidad de **match (conexión exitosa)** entre participantes en citas rápidas (speed dating), utilizando datos del estudio de Columbia University.

El sistema analiza características personales, preferencias y comportamientos para determinar qué factores influyen en la atracción y compatibilidad entre personas, aplicando el marco metodológico **CRISP-DM** (Cross-Industry Standard Process for Data Mining).

## 🎯 Objetivo

Desarrollar un modelo predictivo que determine la probabilidad de que dos personas tengan un match exitoso en una cita rápida, basándose en:
- Atractivo mutuo
- Sinceridad
- Inteligencia
- Nivel de diversión
- Ambición
- Intereses compartidos
- Preferencias y características demográficas

## 📊 Dataset

**Fuente:** Speed Dating Dataset - Columbia University  
**Tamaño:** 8,379 registros  
**Variables:** 195 variables iniciales → 16 variables finales  
**Variable Objetivo:** `match` (1 = match exitoso, 0 = no match)

### Variables Principales
- `attr` - Atractivo calificado por el compañero
- `sinc` - Sinceridad calificada por el compañero
- `intel` - Inteligencia calificada por el compañero
- `fun` - Nivel de diversión calificado por el compañero
- `amb` - Ambición calificada por el compañero
- `shar` - Intereses compartidos
- `like` - Gusto por el compañero
- `prob` - Probabilidad de una segunda cita
- `age`, `age_o` - Edad (participante y compañero)
- `gender`, `race` - Género y raza
- `goal`, `date`, `go_out` - Objetivos y frecuencia de citas

## 🏗️ Metodología CRISP-DM

### 1. Comprensión del Negocio
Predecir matches en citas rápidas para identificar factores clave de compatibilidad.

### 2. Comprensión de los Datos
Análisis exploratorio exhaustivo (82 variables originales, 8,379 registros)

### 3. Preparación de los Datos
- Eliminación de 16 variables irrelevantes
- Manejo de nulos: eliminación (>50%) e imputación con mediana (<50%)
- Winsorizing al percentil 5-95 (manejo de outliers)
- Eliminación de redundancias (correlación > 0.85)
- Eliminación de irrelevancias (correlación < 0.02 con target)

### 4. Modelado
Entrenamiento de 7 algoritmos:
- Árbol de Decisión
- MLP (Red Neuronal)
- SVM
- KNN
- Random Forest
- XGBoost
- Gradient Boosting

### 5. Evaluación
- Validación cruzada estratificada 10-fold
- Métricas: Accuracy, Precision, Recall, F1, ROC-AUC
- ANOVA y Tukey HSD
- Selección del mejor modelo

### 6. Despliegue
Pipeline de producción integrado en aplicación Streamlit

## 🖥️ Instalación

### Requisitos Previos
- Python 3.8+
- pip

### Instalación Rápida

```bash
# Clonar el repositorio
git clone <URL-del-repositorio>
cd match-predictor

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt

# O instalar manualmente
pip install pandas numpy scikit-learn xgboost imbalanced-learn \
            ydata-profiling matplotlib seaborn plotly streamlit \
            joblib scipy statsmodels scikit-optimize shap missingno
```

## 🚀 Uso

### Ejecutar la Aplicación Web

```bash
streamlit run app.py
```

La aplicación estará disponible en: http://localhost:8501

### Funcionalidades de la App

#### 1. 🏠 Inicio
- Métricas principales del modelo
- Gráfico radar interactivo
- Estadísticas del dataset

#### 2. 🎯 Predice tu Match
Ajusta los sliders para predecir tu probabilidad de match:
- Atractivo, sinceridad, inteligencia, diversión, ambición
- Intereses compartidos, gusto por el compañero
- Edad, raza, objetivos
- Visualización con gauge chart y radar chart

#### 3. 📊 Explorar Dataset
- Filtros interactivos
- Histogramas, boxplots, heatmaps
- Análisis de correlaciones
- Comparativas por resultado

#### 4. 📋 Sobre el Modelo
- Proceso CRISP-DM detallado
- Configuración del pipeline
- Métricas de evaluación
- Mejores hiperparámetros

### Ejecutar los Notebooks

```bash
# 1. Preparación de datos
jupyter notebook notebooks/01_preparacion_datos.ipynb

# 2. Modelamiento
jupyter notebook notebooks/02_modelamiento.ipynb
```

## 📈 Resultados del Modelo

### Métricas Principales

| Métrica | Valor |
|---------|-------|
| **ROC-AUC** | **0.91** |
| Accuracy | 0.85 |
| Precision | 0.82 |
| Recall | 0.88 |
| F1-Score | 0.85 |

### Mejor Modelo
- **Algoritmo:** Random Forest
- **Hiperparámetros:** 100 árboles, profundidad máxima automática
- **Validación Cruzada:** 10-fold estratificada
- **ROC-AUC CV:** 0.90

### Importancia de Features
1. Like (gusto por el compañero) - Muy alta
2. Prob (probabilidad esperada) - Alta
3. Atractivo - Alta
4. Sinceridad - Media-Alta
5. Inteligencia - Media

## 📁 Estructura del Proyecto

```
match-predictor/
├── app.py                          # Aplicación Streamlit principal
├── requirements.txt                # Dependencias del proyecto
├── README.md                       # Este archivo
├── README_APP.md                   # Documentación de la app
├── README_NOTEBOOKS.md             # Documentación de notebooks
├── data/
│   ├── Speed Dating Data.csv        # Datos originales (1.9 MB)
│   ├── data_prepared.csv            # Datos preprocesados
│   ├── X_train.csv, X_test.csv      # Features train/test
│   ├── y_train.csv, y_test.csv      # Target train/test
│   ├── feature_names.pkl            # Nombres de features
│   └── scaler.pkl                   # StandardScaler
├── models/
│   ├── pipeline_match_predictor.pkl # Pipeline final
│   ├── metricas_finales.pkl         # Métricas del modelo
│   └── scaler.pkl                   # Scaler guardado
├── notebooks/
│   ├── 01_preparacion_datos.ipynb   # Preprocesamiento (20 celdas)
│   └── 02_modelamiento.ipynb        # Modelado (22 celdas)
└── reports/
    ├── pandas_profiling.html        # Reporte exploratorio
    ├── distribucion_target.png      # Distribución del target
    ├── matriz_correlacion_inicial.png
    ├── curvas_roc_comparativas.png
    ├── evaluacion_final.png
    ├── feature_importance.png
    └── shap_summary.png
```

## 🔍 Hallazgos Clave

1. **El gusto mutuo (like) es el predictor más importante** - Tiene la correlación más alta con el match exitoso

2. **La probabilidad esperada es muy predictiva** - Las personas son buenas evaluando su propia compatibilidad

3. **El atractivo físico importa, pero no es todo** - Importancia alta pero menor que factores de personalidad

4. **Los matches exitosos tienen perfiles equilibrados** - Alta sinceridad, inteligencia y diversión combinadas

5. **La edad no es tan determinante** - Correlación baja con el match, sugiriendo compatibilidad más allá de la edad

## 🤝 Contribución

1. Fork del proyecto
2. Crear branch para la feature (`git checkout -b feature/AmazingFeature`)
3. Commit de los cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para más detalles

## 🙏 Reconocimientos

- Datos: Columbia University Speed Dating Study
- Framework: Streamlit
- Bibliotecas: Scikit-learn, XGBoost, Plotly, y otros contribuidores de código abierto

## 📞 Contacto

Proyecto: [Speed Dating Predictor](https://github.com/tu-usuario/match-predictor)  
Email: tu.email@ejemplo.com

---

**Hecho con 💕 para el análisis de compatibilidad en citas rápidas**