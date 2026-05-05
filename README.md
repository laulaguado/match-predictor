# match-predictor
Predictor de compatibilidad en Speed Dating usando CRISP-DM y Machine Learning
# 💘 Predictor de Compatibilidad en Speed Dating

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-?
??-green)
![CRISP-DM](https://img.shields.io/badge/Metodología-CRISP--DM-purple)

## 📌 Descripción
Proyecto de Minería de Datos que predice si dos personas tendrán match mutuo
en una cita rápida, usando datos reales del experimento de Columbia University.

## 📊 Dataset
- **Fuente:** [Speed Dating Experiment - Kaggle](https://www.kaggle.com/datasets/annavictoria/speed-dating-experiment)
- **Registros:** ~8.378 observaciones
- **Variables:** 195 (se seleccionan ~20 relevantes)
- **Variable objetivo:** `match` (binaria)

## 🤖 Modelos evaluados
- Árbol de Decisión, Red Neuronal (MLP), SVM, KNN
- Random Forest, XGBoost, Gradient Boosting
- **Mejor modelo:** [completar tras entrenamiento]
- **ROC-AUC final:** [completar tras entrenamiento]

## 🚀 Cómo ejecutar

### 1. Clonar el repositorio
\`\`\`bash
git clone https://github.com/TU_USUARIO/match-predictor-crisp-dm.git
cd match-predictor-crisp-dm
\`\`\`

### 2. Instalar dependencias
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 3. Descargar el dataset
Descargar desde [Kaggle](https://www.kaggle.com/datasets/annavictoria/speed-dating-experiment)
y colocar en `data/Speed Dating Data.csv`

### 4. Ejecutar notebooks en orden
\`\`\`bash
jupyter notebook notebooks/01_preparacion_datos.ipynb
jupyter notebook notebooks/02_modelamiento.ipynb
\`\`\`

### 5. Lanzar la app
\`\`\`bash
streamlit run app.py
\`\`\`

## 👥 Autores
- Laura Laguado
- Sofía Navales

## 📚 Universidad
Materia: Minería de Datos — 2026
