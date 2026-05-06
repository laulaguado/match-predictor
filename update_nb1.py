import nbformat

# Notebook 1: 01_preparacion_datos.ipynb
nb1 = nbformat.v4.new_notebook()

cells1 = []

# Celda 1: Título
cells1.append(nbformat.v4.new_markdown_cell('# Preparación de Datos - Speed Dating Columbia University\n\nEste notebook realiza la preparación completa de los datos para el análisis de citas rápidas utilizando la metodología CRISP-DM.'))

# Celda 2: Imports
cells1.append(nbformat.v4.new_code_cell('''import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

PALETA_ROSA = ["#FF1493", "#FF69B4", "#FFB6C1", "#FFC0CB"]
sns.set_palette(PALETA_ROSA)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

print("Librerías importadas correctamente")
print("Paleta de colores: #FF1493 (Deep Pink)")'''))

# Celda 3: Data loading
cells1.append(nbformat.v4.new_markdown_cell('## 1. Carga de Datos'))

cells1.append(nbformat.v4.new_code_cell('''data_path = "../data/Speed Dating Data.csv"
df = pd.read_csv(data_path, encoding="latin-1")

print(f"Dimensiones del dataset: {df.shape}")
print(f"  - Filas: {df.shape[0]:,}")
print(f"  - Columnas: {df.shape[1]}")
print(f"\nColumnas totales: {list(df.columns)}")
print(f"\nPrimeras 5 filas:")
df.head()'''))

# Celda 4: Diccionario
cells1.append(nbformat.v4.new_markdown_cell('## 2. Diccionario de Variables'))

cells1.append(nbformat.v4.new_code_cell('''diccionario_variables = {
    "iid": "Identificador único del participante",
    "id": "Identificador único de la cita específica",
    "idg": "Identificador del grupo de citas",
    "gender": "Género (0=mujer, 1=hombre)",
    "age": "Edad del participante",
    "race": "Raza/Etnia del participante",
    "goal": "Objetivo principal de las citas",
    "date": "Frecuencia de citas",
    "go_out": "Frecuencia de salir",
    "field": "Campo de estudio",
    "income": "Ingresos anuales",
    "attr1_1": "Atractivo ideal (autoevaluación)",
    "sinc1_1": "Sinceridad ideal (autoevaluación)",
    "intel1_1": "Inteligencia ideal (autoevaluación)",
    "fun1_1": "Diversión ideal (autoevaluación)",
    "amb1_1": "Ambición ideal (autoevaluación)",
    "shar1_1": "Intereses compartidos ideal",
    "attr3_1": "Atractivo autoevaluado (ronda 3)",
    "sinc3_1": "Sinceridad autoevaluada (ronda 3)",
    "intel3_1": "Inteligencia autoevaluada (ronda 3)",
    "fun3_1": "Diversión autoevaluada (ronda 3)",
    "amb3_1": "Ambición autoevaluada (ronda 3)",
    "attr": "Atractivo calificado por el compañero",
    "sinc": "Sinceridad calificada por el compañero",
    "intel": "Inteligencia calificada por el compañero",
    "fun": "Diversión calificada por el compañero",
    "amb": "Ambición calificada por el compañero",
    "shar": "Intereses compartidos calificados",
    "like": "Gusto por el compañero (0-10)",
    "prob": "Probabilidad de una segunda cita (0-10)",
    "met": "Se han conocido previamente (1=sí, 0=no)",
    "age_o": "Edad del compañero",
    "race_o": "Raza del compañero",
    "samerace": "Misma raza (1=sí, 0=no)",
    "imprace": "Importancia de la raza (0-10)",
    "attr_o": "Atractivo calificado por el participante",
    "sinc_o": "Sinceridad calificada por el participante",
    "intel_o": "Inteligencia calificada por el participante",
    "fun_o": "Diversión calificada por el participante",
    "amb_o": "Ambición calificada por el participante",
    "shar_o": "Intereses compartidos calificados",
    "like_o": "Gusto calificado por el participante",
    "prob_o": "Probabilidad calificada por el participante",
    "dec": "Decisión del participante (1=sí, 0=no)",
    "dec_o": "Decisión del compañero (1=sí, 0=no)",
    "int_corr": "Correlación de intereses",
    "match": "Match conseguido (1=sí, 0=no) - Variable objetivo"
}

df_dict = pd.DataFrame(list(diccionario_variables.items()), columns=["Variable", "Descripción"])
print(f"Total de variables en el diccionario: {len(df_dict)}")
print("\nVariables principales:")
df_dict.head(15)'''))

# Celda 5: Profiling
cells1.append(nbformat.v4.new_markdown_cell('## 3. Análisis Exploratorio con Pandas Profiling'))

cells1.append(nbformat.v4.new_code_cell('''import os
os.makedirs("../reports", exist_ok=True)

try:
    from ydata_profiling import ProfileReport
    print("Usando ydata-profiling...")
except ImportError:
    try:
        from pandas_profiling import ProfileReport
        print("Usando pandas-profiling...")
    except ImportError:
        print("Instalando ydata-profiling...")
        import subprocess
        subprocess.check_call(["pip", "install", "ydata-profiling", "-q"])
        from ydata_profiling import ProfileReport

print("Generando perfil del dataset...")
profile = ProfileReport(df, title="Speed Dating - Análisis Exploratorio", explorative=True, minimal=True)
profile.to_file("../reports/pandas_profiling.html")
print("✓ Pandas Profiling guardado en reports/pandas_profiling.html")'''))

# Celda 6: Target distribution
cells1.append(nbformat.v4.new_markdown_cell('## 4. Distribución de la Variable Objetivo (match)'))

cells1.append(nbformat.v4.new_code_cell('''match_counts = df["match"].value_counts()
print("Distribución de la variable 'match':")
print(f"  No Match (0): {match_counts[0]:,} ({match_counts[0]/len(df)*100:.1f}%)")
print(f"  Match (1):    {match_counts[1]:,} ({match_counts[1]/len(df)*100:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico de barras
bars = axes[0].bar(["No Match", "Match"], 
                   [match_counts[0], match_counts[1]],
                   color=["#FFB6C1", "#FF1493"], 
                   edgecolor="black", linewidth=0.8)
axes[0].set_title("Distribución de Matches", fontsize=14, fontweight="bold", color="#FF1493")
axes[0].set_ylabel("Frecuencia", fontsize=12)
axes[0].set_xlabel("Resultado", fontsize=12)
axes[0].tick_params(colors="#FF1493")

for bar in bars:
    h = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., h, 
                 f"{int(h):,}",
                 ha="center", va="bottom", fontweight="bold", fontsize=12)

# Gráfico de torta
wedges, texts, autotexts = axes[1].pie([match_counts[0], match_counts[1]],
                                        labels=["No Match", "Match"],
                                        autopct="%1.1f%%",
                                        colors=["#FFB6C1", "#FF1493"],
                                        startangle=90,
                                        wedgeprops={"edgecolor": "black", "linewidth": 0.5})
axes[1].set_title("Proporción de Matches", fontsize=14, fontweight="bold", color="#FF1493")

for autotext in autotexts:
    autotext.set_color("white")
    autotext.set_fontweight("bold")
    autotext.set_fontsize(12)

plt.tight_layout()
plt.savefig("../reports/distribucion_target.png", dpi=300, bbox_inches="tight", facecolor="white")
print("\n✓ Gráfico guardado en reports/distribucion_target.png")
plt.show()

print(f"\nTotal de registros: {len(df):,}")'''))

# Celda 7: Descriptive stats
cells1.append(nbformat.v4.new_markdown_cell('## 5. Estadística Descriptiva'))

cells1.append(nbformat.v4.new_code_cell('''numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Variables numéricas: {len(numeric_cols)}")

descriptivas = df[numeric_cols].describe().T
descriptivas["missing_count"] = df[numeric_cols].isnull().sum()
descriptivas["missing_pct"] = (df[numeric_cols].isnull().sum() / len(df) * 100).round(2)

print("\nEstadísticas descriptivas (primeras 10 variables):")
descriptivas_display = descriptivas[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "missing_count", "missing_pct"]]
print(descriptivas_display.head(10).to_string())

descriptivas.to_csv("../reports/estadisticas_descriptivas.csv")
print("\n✓ Estadísticas guardadas en reports/estadisticas_descriptivas.csv")

# Matriz de correlación
plt.figure(figsize=(16, 14))
corr_matrix = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap="RdBu_r", center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8},
            fmt=".2f", annot=False, vmin=-1, vmax=1,
            annot_kws={"size": 8})
plt.title("Matriz de Correlación Inicial", fontsize=16, fontweight="bold", color="#FF1493", pad=20)
plt.tight_layout()
plt.savefig("../reports/matriz_correlacion_inicial.png", dpi=300, bbox_inches="tight", facecolor="white")
print("✓ Matriz de correlación guardada en reports/matriz_correlacion_inicial.png")
plt.show()
print("\n✓ Análisis exploratorio completado")'''))

# Celda 8: Missing values
cells1.append(nbformat.v4.new_markdown_cell('## 6. Limpieza de Nulos'))

cells1.append(nbformat.v4.new_code_cell('''df_clean = df.copy()

cols_to_drop = ["iid", "id", "idg", "partner", "pid", "wave", "round", 
                "position", "positin1", "order", "condtn", "undergrd", 
                "zipcode", "career", "from", "field"]

print(f"Variables a eliminar: {len(cols_to_drop)}")
print(f"Nombres: {cols_to_drop}")

df_clean.drop(columns=cols_to_drop, inplace=True, errors="ignore")
print(f"\n✓ Variables eliminadas")
print(f"  Columnas restantes: {df_clean.shape[1]}")
print(f"  Filas: {df_clean.shape[0]}")

nulos_por_columna = df_clean.isnull().sum()
pct_nulos = (nulos_por_columna / len(df_clean) * 100).round(2)

total_nulos = df_clean.isnull().sum().sum()
print(f"\n✓ Análisis de nulos:")
print(f"  Total de valores nulos: {total_nulos:,}")
print(f"  Porcentaje: {(total_nulos / (df_clean.shape[0] * df_clean.shape[1]) * 100):.2f}%")

columnas_mas_50 = nulos_por_columna[pct_nulos > 50].index.tolist()
print(f"\n✓ Columnas con >50% nulos: {len(columnas_mas_50)}")
if columnas_mas_50:
    print(f"  Nombres: {columnas_mas_50}")

if columnas_mas_50:
    df_clean.drop(columns=columnas_mas_50, inplace=True)
    print(f"\n✓ Columnas eliminadas: {columnas_mas_50}")
    print(f"  Columnas restantes: {df_clean.shape[1]}")

df_nulos = pd.DataFrame({
    "Columna": nulos_por_columna.index,
    "Nulos": nulos_por_columna.values,
    "Porcentaje": pct_nulos.values
})
df_nulos.to_csv("../reports/nulos_analisis.csv", index=False)
print(f"\n✓ Análisis de nulos guardado en reports/nulos_analisis.csv")'''))

# Celda 9: Imputation
cells1.append(nbformat.v4.new_markdown_cell('## 7. Imputación de Nulos con Mediana'))

cells1.append(nbformat.v4.new_code_cell('''numeric_cols_clean = df_clean.select_dtypes(include=[np.number]).columns.tolist()
if "match" in numeric_cols_clean:
    numeric_cols_clean.remove("match")

medianas = {}
imputados = 0
for col in numeric_cols_clean:
    if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
        med = df_clean[col].median()
        nulos = df_clean[col].isnull().sum()
        medianas[col] = med
        df_clean[col].fillna(med, inplace=True)
        imputados += nulos
        print(f"  ✓ {col}: mediana={med:.2f}, nulos imputados={nulos}")

total_nulos = df_clean.isnull().sum().sum()
print(f"\n✓ Imputación completada:")
print(f"  Valores imputados: {imputados}")
print(f"  Nulos restantes: {total_nulos}")

pd.DataFrame(list(medianas.items()), columns=["Columna", "Mediana"]).to_csv("../reports/medianas_imputadas.csv", index=False)
print(f"\n✓ Medianas guardadas en reports/medianas_imputadas.csv")'''))

# Celda 10: Winsorizing
cells1.append(nbformat.v4.new_markdown_cell('## 8. Winsorizing (Percentil 5-95)'))

cells1.append(nbformat.v4.new_code_cell('''# Identificar columnas de scores (excluyendo variables categóricas y target)
score_cols = [c for c in df_clean.select_dtypes(include=[np.number]).columns 
              if c not in ["match", "gender", "race", "race_o", "samerace", 
                          "goal", "date", "go_out", "field_cd", "career_c", 
                          "met", "dec", "dec_o", "int_corr"]]

print(f"✓ Columnas a winsorizar: {len(score_cols)}")
print(f"  Variables: {score_cols[:10]}...")

outliers_total = 0
for col in score_cols:
    if col in df_clean.columns:
        p5 = df_clean[col].quantile(0.05)
        p95 = df_clean[col].quantile(0.95)
        outliers = ((df_clean[col] < p5) | (df_clean[col] > p95)).sum()
        outliers_total += outliers
        df_clean[col] = df_clean[col].clip(lower=p5, upper=p95)

print(f"\n✓ Winsorizing completado:")
print(f"  Outliers recortados: {outliers_total}")
print(f"  Shape final: {df_clean.shape}")

df_clean.to_csv("../data/data_prepared.csv", index=False)
print(f"\n✓ Datos preparados guardados en data/data_prepared.csv")
print(f"  Filas: {df_clean.shape[0]:,}")
print(f"  Columnas: {df_clean.shape[1]}")'''))

# Celda 11: Redundancia
cells1.append(nbformat.v4.new_markdown_cell('## 9. Reducción de Redundancia e Irrelevancia'))

cells1.append(nbformat.v4.new_code_cell('''print("✓ Análisis de correlación con la variable objetivo:")
corr_m = df_clean.corr()
target_c = corr_m["match"].abs().sort_values(ascending=False)
print(target_c.head(15).to_string())

# Identificar variables irrelevantes
irrelevantes = target_c[target_c < 0.02].index.tolist()
irrelevantes = [c for c in irrelevantes if c != "match"]
print(f"\n✓ Variables irrelevantes (corr < 0.02 con match): {len(irrelevantes)}")
if irrelevantes:
    print(f"  Nombres: {irrelevantes}")

# Identificar variables redundantes
redundantes = []
corr_abs = corr_m.abs()
np.fill_diagonal(corr_abs.values, 0)

for i in range(len(corr_abs.columns)):
    for j in range(i+1, len(corr_abs.columns)):
        if corr_abs.iloc[i, j] > 0.85:
            redundantes.append((corr_abs.columns[i], corr_abs.columns[j], corr_abs.iloc[i, j]))

print(f"\n✓ Pares redundantes (corr > 0.85): {len(redundantes)}")
for r in redundantes[:10]:
    print(f"  {r[0]} - {r[1]}: {r[2]:.3f}")

print(f"\n✓ Preprocesamiento completado exitosamente")
print(f"  Dataset final: {df_clean.shape[0]:,} filas × {df_clean.shape[1]} columnas")'''))

nb1.cells = cells1

with open("notebooks/01_preparacion_datos.ipynb", "w") as f:
    nbformat.write(nb1, f)

print("✓ Notebook 01 actualizado correctamente")
