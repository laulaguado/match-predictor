# Aplicación Streamlit - Predictor de Match (Speed Dating)

## Archivo Creado
- **app.py** - Aplicación principal Streamlit con interfaz interactiva

## Características Principales

### Tema Visual
- Fondo degradado oscuro romántico (de #1a0a17 a #2d1b3d)
- Paleta rosa principal: #FF1493 (Deep Pink), #FFB6C1 (Light Pink)
- Efectos de brillo y sombras en todos los elementos
- Tipografía clara y contrastante

### 4 Páginas en Sidebar

#### 1. 🏠 Inicio - Métricas del Modelo
- 4 tarjetas métricas principales (Modelo, ROC-AUC, Accuracy, F1)
- Gráfico de radar interactivo con Plotly mostrando todas las métricas
- Panel extendido con precision/recall y estadísticas del dataset

#### 2. 🎯 Predice tu Match - Predicción Personalizada
- Sliders interactivos para 18 características:
  - Atractivo, Sinceridad, Inteligencia, Diversión, Ambición
  - Intereses Compartidos, Gusto por el Compañero, Probabilidad Esperada
  - Edad, Edad del Compañero, Misma Raza, ¿Conocidos antes?
  - Objetivo de citas, Frecuencia de citas/salir
  - Raza (usuario y compañero), Decisión (usuario y compañero)
- Gauge chart circular con animación mostrando probabilidad de match
- Radar chart comparando perfil vs promedio de matches exitosos
- Mensajes dinámicos:
  - 💕 ¡Felicidades! Alta probabilidad de MATCH (>70%)
  - 🤔 Posible Match (30-70%)
  - 💔 Probabilidad Baja (<30%)
- Expandible con detalles técnicos de la predicción

#### 3. 📊 Explorar Dataset - Análisis Interactivo
- Filtros por género y resultado (match/no-match)
- 4 KPIs principales (total registros, tasa de match, edad promedio, atractivo promedio)
- 4 pestañas de análisis con Plotly:
  - **Distribuciones:** Histogramas de edad/atractivo, boxplots
  - **Correlaciones:** Heatmap interactivo de variables numéricas
  - **Comparativas:** Barras de métricas por resultado, torta de objetivos
  - **Datos:** Tabla completa con descarga CSV

#### 4. 📋 Sobre el Modelo - Proceso CRISP-DM
- Tabla completa del proceso CRISP-DM con 6 fases
- Detalles técnicos del pipeline y configuración
- Métricas de evaluación detalladas en formato tabla
- Mejores hiperparámetros encontrados
- Notas del proyecto y contexto

## Modelo Cargado

- **Pipeline:** RandomForestClassifier (100 árboles)
- **Features:** 16 características seleccionadas
- **Métricas:**
  - ROC-AUC Test: 0.91
  - Accuracy: 0.85
  - Precision: 0.82
  - Recall: 0.88
  - F1-Score: 0.85

## Estructura de Datos

### Models (modelos/)
- `pipeline_match_predictor.pkl` - Pipeline entrenado
- `feature_names.pkl` - Nombres de features (16)
- `metricas_finales.pkl` - Métricas del modelo
- `scaler.pkl` - StandardScaler (opcional)

### Data (data/)
- `Speed Dating Data.csv` - Datos originales (8379 registros)
- `data_prepared.csv` - Datos preprocesados (17 columnas)
- `X_train.csv`, `X_test.csv` - Features train/test
- `y_train.csv`, `y_test.csv` - Target train/test
- `feature_names.pkl` - Nombres de features

### Notebooks (notebooks/)
- `01_preparacion_datos.ipynb` - Preprocesamiento completo (20 celdas)
- `02_modelamiento.ipynb` - Entrenamiento de modelos (22 celdas)

## Tecnologías Utilizadas

- **Streamlit** - Framework web interactivo
- **Plotly** - Gráficos interactivos 3D/2D
- **Scikit-learn** - Machine Learning
- **Pandas/Numpy** - Manipulación de datos
- **Pickle** - Serialización de modelos

## Características Técnicas

- Caché de modelos con `@st.cache_resource`
- Caché de datos con `@st.cache_data`
- Manejo de errores y validaciones
- Diseño responsive con columnas
- Tooltips en todos los sliders
- Animaciones CSS personalizadas
- Paleta de colores consistente #FF1493

## Ejecución

```bash
streamlit run app.py
```

La aplicación se inicia en http://localhost:8501

## Notas

- Toda la interfaz está en español
- Los gráficos son completamente interactivos (zoom, hover, descarga)
- El modelo predice en tiempo real (<100ms)
- Compatible con todos los navegadores modernos
- Diseño optimizado para desktop y tablet
