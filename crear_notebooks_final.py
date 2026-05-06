import nbformat
import os

print("Creando notebooks...")

# Notebook 1
nb1 = nbformat.v4.new_notebook()
cells1 = []

cells1.append(nbformat.v4.new_markdown_cell('# Preparacion de Datos\nPreparacion completa de datos.'))

cells1.append(nbformat.v4.new_code_cell('''import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
PALETA_ROSA = ["#FF1493", "#FF69B4", "#FFB6C1", "#FFC0CB"]
sns.set_palette(PALETA_ROSA)
pd.set_option("display.max_columns", None)
print("Librerias importadas")'''))

cells1.append(nbformat.v4.new_markdown_cell('## 1. Carga de Datos'))

cells1.append(nbformat.v4.new_code_cell('''df = pd.read_csv("../data/Speed Dating Data.csv", encoding="latin-1")
print("Dimensiones: %s" % str(df.shape))
df.head()'''))

cells1.append(nbformat.v4.new_markdown_cell('## 2. Diccionario'))

cells1.append(nbformat.v4.new_code_cell('''dicc = {"iid":"ID participante","gender":"Genero","age":"Edad","attr":"Atractivo","sinc":"Sinceridad","intel":"Inteligencia","fun":"Diversion","amb":"Ambicion","shar":"Intereses","like":"Gusto","prob":"Probabilidad","age_o":"Edad companero","race_o":"Raza companero","samerace":"Misma raza","met":"Conocidos","match":"Match TARGET"}
df_dict = pd.DataFrame(list(dicc.items()), columns=["Variable","Descripcion"])
print("Variables: %d" % len(dicc))
df_dict'''))

cells1.append(nbformat.v4.new_markdown_cell('## 3. Pandas Profiling'))

cells1.append(nbformat.v4.new_code_cell('''import os
os.makedirs("../reports", exist_ok=True)
try:
    from ydata_profiling import ProfileReport
except:
    from pandas_profiling import ProfileReport
profile = ProfileReport(df, title="Speed Dating", explorative=True, minimal=True)
profile.to_file("../reports/pandas_profiling.html")
print("Guardado: reports/pandas_profiling.html")'''))

cells1.append(nbformat.v4.new_markdown_cell('## 4. Distribucion Target'))

cells1.append(nbformat.v4.new_code_cell('''mc = df["match"].value_counts()
print("No Match: %d (%.1f%%)" %% (mc[0], mc[0]*100.0/len(df)))
print("Match: %d (%.1f%%)" %% (mc[1], mc[1]*100.0/len(df)))
fig, ax = plt.subplots(1,2,figsize=(15,6))
bars = ax[0].bar(["No Match","Match"],[mc[0],mc[1]],color=["#FFB6C1","#FF1493"],edgecolor="black")
ax[0].set_title("Distribucion",fontweight="bold")
ax[0].set_ylabel("Frecuencia")
for b in bars:
    h=b.get_height()
    ax[0].text(b.get_x()+b.get_width()/2.,h,"%.0f" %% h,ha="center",va="bottom",fontweight="bold")
ax[1].pie([mc[0],mc[1]],labels=["No Match","Match"],autopct="%%1.1f%%",colors=["#FFB6C1","#FF1493"])
ax[1].set_title("Proporcion",fontweight="bold")
plt.tight_layout()
plt.savefig("../reports/distribucion_target.png",dpi=300,bbox_inches="tight")
plt.show()'''))

cells1.append(nbformat.v4.new_markdown_cell('## 5. Estadistica'))

cells1.append(nbformat.v4.new_code_cell('''nc = df.select_dtypes(include=[np.number]).columns.tolist()
print("Variables numericas: %d" %% len(nc))
desc = df[nc].describe().T
desc["missing"] = df[nc].isnull().sum()
desc["pct_miss"] = (df[nc].isnull().sum()/len(df)*100).round(2)
print(desc.head(10).to_string())
desc.to_csv("../reports/estadisticas_descriptivas.csv")
plt.figure(figsize=(16,14))
cm = df[nc].corr()
mask = np.triu(np.ones_like(cm,dtype=bool))
sns.heatmap(cm,mask=mask,cmap="RdBu_r",center=0,square=True,linewidths=0.5,cbar_kws={"shrink":.8},fmt=".2f",annot=False,vmin=-1,vmax=1)
plt.title("Matriz Correlacion",fontweight="bold",pad=20)
plt.tight_layout()
plt.savefig("../reports/matriz_correlacion_inicial.png",dpi=300,bbox_inches="tight")
plt.show()'''))

cells1.append(nbformat.v4.new_markdown_cell('## 6. Nulos'))

cells1.append(nbformat.v4.new_code_cell('''dfc = df.copy()
dfc.drop(columns=["iid","id","idg","partner","pid","wave","round","position","positin1","order","condtn","undergrd","zipcode","career","from","field"],inplace=True,errors="ignore")
print("Columnas: %d" %% dfc.shape[1])
total_nulos = dfc.isnull().sum().sum()
print("Nulos totales: %d" %% total_nulos)
col50 = dfc.isnull().sum()[dfc.isnull().sum()/len(dfc)>0.5].index.tolist()
if col50:
    dfc.drop(columns=col50,inplace=True)
    print("Eliminadas >50%% nulos: %s" %% col50)
dfc.to_csv("../data/data_prepared.csv", index=False)
print("Guardado: data/data_prepared.csv")'''))

cells1.append(nbformat.v4.new_markdown_cell('## 7. Imputacion'))

cells1.append(nbformat.v4.new_code_cell('''numc = dfc.select_dtypes(include=[np.number]).columns.tolist()
if "match" in numc:
    numc.remove("match")
for col in numc:
    if dfc[col].isnull().sum()>0:
        med=dfc[col].median()
        dfc[col].fillna(med,inplace=True)
        print("%s: mediana=%.2f" %% (col,med))
print("Nulos restantes: %d" %% dfc.isnull().sum().sum())'''))

cells1.append(nbformat.v4.new_markdown_cell('## 8. Winsorizing'))

cells1.append(nbformat.v4.new_code_cell('''scols = [c for c in dfc.select_dtypes(include=[np.number]).columns if c not in ["match","gender","race","race_o","samerace","goal","date","go_out","field_cd","career_c","met","dec","dec_o","int_corr"]]
for col in scols:
    if col in dfc.columns:
        p5=dfc[col].quantile(0.05)
        p95=dfc[col].quantile(0.95)
        dfc[col]=dfc[col].clip(lower=p5,upper=p95)
print("Winsorizing completado")
dfc.to_csv("../data/data_prepared.csv", index=False)'''))

cells1.append(nbformat.v4.new_markdown_cell('## 9. Redundancia'))

cells1.append(nbformat.v4.new_code_cell('''corr_m = dfc.corr()
tc = corr_m["match"].abs().sort_values(ascending=False)
print(tc.head(15))
irr = tc[tc<0.02].index.tolist()
irr = [c for c in irr if c!="match"]
print("Irrelevantes: %d" %% len(irr))
red=[]
ca=corr_m.abs()
np.fill_diagonal(ca.values,0)
for i in range(len(ca.columns)):
    for j in range(i+1,len(ca.columns)):
        if ca.iloc[i,j]>0.85:
            red.append((ca.columns[i],ca.columns[j],ca.iloc[i,j]))
print("Redundantes: %d" %% len(red))
for r in red[:10]:
    print("  %s - %s: %.3f" %% (r[0],r[1],r[2]))'''))

nb1.cells = cells1
with open("notebooks/01_preparacion_datos.ipynb","w") as f:
    nbformat.write(nb1,f)
print("Notebook 1 OK")

# Notebook 2
nb2 = nbformat.v4.new_notebook()
cells2 = []

cells2.append(nbformat.v4.new_markdown_cell('# Modelamiento\nEntrenamiento de modelos.'))

cells2.append(nbformat.v4.new_code_cell('''import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
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
print("Librerias importadas")'''))

cells2.append(nbformat.v4.new_markdown_cell('## 1. Datos'))

cells2.append(nbformat.v4.new_code_cell('''X_train=pd.read_csv("../data/X_train.csv")
X_test=pd.read_csv("../data/X_test.csv")
y_train=pd.read_csv("../data/y_train.csv").values.ravel()
y_test=pd.read_csv("../data/y_test.csv").values.ravel()
print("X_train: %s" %% str(X_train.shape))
print("X_test: %s" %% str(X_test.shape))
with open("../data/feature_names.pkl","rb") as f:
    fn=pickle.load(f)
print("Features: %d" %% len(fn))
with open("../data/scaler.pkl","rb") as f:
    sc=pickle.load(f)
print("Scaler OK")'''))

cells2.append(nbformat.v4.new_markdown_cell('## 2. Modelos'))

cells2.append(nbformat.v4.new_code_cell('''from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time
mo={"Arbol":DecisionTreeClassifier(random_state=42),
    "MLP":MLPClassifier(random_state=42,max_iter=1000),
    "SVM":SVC(random_state=42,probability=True),
    "KNN":KNeighborsClassifier(),
    "RF":RandomForestClassifier(random_state=42),
    "XGB":XGBClassifier(random_state=42,eval_metric="logloss",use_label_encoder=False),
    "GB":GradientBoostingClassifier(random_state=42)}
cv=StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
scoring={"accuracy":make_scorer(accuracy_score),"precision":make_scorer(precision_score),"recall":make_scorer(recall_score),"f1":make_scorer(f1_score),"roc_auc":make_scorer(roc_auc_score)}
res=[]
cvsc={}
print("Entrenando 7 modelos 10-fold CV...")
for n,m in mo.items():
    print("  %s..." %% n)
    t0=time.time()
    r=cross_validate(m,X_train,y_train,cv=cv,scoring=scoring,return_train_score=True)
    t=time.time()-t0
    res.append({"Modelo":n,"Accuracy_mean":r["test_accuracy"].mean(),"Accuracy_std":r["test_accuracy"].std(),"Precision_mean":r["test_precision"].mean(),"Precision_std":r["test_precision"].std(),"Recall_mean":r["test_recall"].mean(),"Recall_std":r["test_recall"].std(),"F1_mean":r["test_f1"].mean(),"F1_std":r["test_f1"].std(),"ROC_AUC_mean":r["test_roc_auc"].mean(),"ROC_AUC_std":r["test_roc_auc"].std(),"Tiempo":t})
    cvsc[n]=r["test_roc_auc"]
    print("    ROC-AUC: %.4f" %% res[-1]["ROC_AUC_mean"])
dfr=pd.DataFrame(res).sort_values("ROC_AUC_mean",ascending=False)
print("\\nTabla comparativa:")
dfr[["Modelo","Accuracy_mean","Precision_mean","Recall_mean","F1_mean","ROC_AUC_mean"]]'''))

cells2.append(nbformat.v4.new_markdown_cell('## 3. ROC'))

cells2.append(nbformat.v4.new_code_cell('''from sklearn.metrics import RocCurveDisplay
fig,ax=plt.subplots(figsize=(12,10))
for n,m in mo.items():
    m.fit(X_train,y_train)
    RocCurveDisplay.from_estimator(m,X_test,y_test,ax=ax,name=n)
ax.plot([0,1],[0,1],"k--",label="Aleatorio")
ax.set_xlabel("Falsos Positivos")
ax.set_ylabel("Verdaderos Positivos")
ax.set_title("Curvas ROC",fontweight="bold")
ax.legend(loc="lower right")
os.makedirs("../reports",exist_ok=True)
plt.savefig("../reports/curvas_roc_comparativas.png",dpi=300,bbox_inches="tight")
plt.show()'''))

cells2.append(nbformat.v4.new_markdown_cell('## 4. ANOVA'))

cells2.append(nbformat.v4.new_code_cell('''from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
cvl=[]
for n,s in cvsc.items():
    for v in s:
        cvl.append({"Modelo":n,"ROC_AUC":v})
dfcv=pd.DataFrame(cvl)
grp=[dfcv[dfcv["Modelo"]==m]["ROC_AUC"].values for m in dfr["Modelo"]]
stat,p=stats.f_oneway(*grp)
print("ANOVA: F=%.4f, p=%.6f" %% (stat,p))
if p<0.05:
    print("Diferencias significativas")
tk=pairwise_tukeyhsd(dfcv["ROC_AUC"],dfcv["Modelo"],alpha=0.05)
print(tk)
with open("../reports/anova_results.txt","w") as f:
    f.write("ANOVA F=%.4f p=%.6f\\n" %% (stat,p))
    f.write(str(tk))
print("Guardado: reports/anova_results.txt")'''))

cells2.append(nbformat.v4.new_markdown_cell('## 5. Top 3'))

cells2.append(nbformat.v4.new_code_cell('''top3=dfr.nlargest(3,"ROC_AUC_mean")
print("TOP 3:")
display(top3[["Modelo","ROC_AUC_mean"]])
top3n=top3["Modelo"].tolist()
top3m={n:mo[n] for n in top3n}
print("Modelos: %s" %% top3n)'''))

cells2.append(nbformat.v4.new_markdown_cell('## 6. RandomizedSearchCV'))

cells2.append(nbformat.v4.new_code_cell('''from sklearn.model_selection import RandomizedSearchCV
prf={"n_estimators":[50,100,200,300],"max_depth":[None,10,20,30],"min_samples_split":[2,5,10]}
pxg={"n_estimators":[50,100,200],"max_depth":[3,5,7],"learning_rate":[0.01,0.1,0.3]}
pgb={"n_estimators":[50,100,200],"max_depth":[3,5,7],"learning_rate":[0.01,0.1,0.3]}
pg={}
for n in top3n:
    if "RF" in n: pg[n]=prf
    elif "XGB" in n: pg[n]=pxg
    elif "GB" in n: pg[n]=pgb
    else: pg[n]={}
print("RandomizedSearchCV...")
mrs={}
rsr=[]
for n in top3n:
    print("  %s..." %% n)
    rs=RandomizedSearchCV(top3m[n],pg[n],n_iter=10,cv=5,scoring="roc_auc",random_state=42,n_jobs=-1)
    rs.fit(X_train,y_train)
    mrs[n]=rs.best_estimator_
    rsr.append({"Modelo":n,"ROC-AUC":rs.best_score_,"Params":str(rs.best_params_)})
    print("    ROC-AUC: %.4f" %% rs.best_score_)
dfrs=pd.DataFrame(rsr)
mn=dfrs.loc[dfrs["ROC-AUC"].idxmax(),"Modelo"]
mm=mrs[mn]
print("\\nMejor: %s" %% mn)'''))

cells2.append(nbformat.v4.new_markdown_cell('## 7. BayesSearchCV'))

cells2.append(nbformat.v4.new_code_cell('''from skopt import BayesSearchCV
from skopt.space import Real,Integer,Categorical
print("BayesSearchCV...")
if "RF" in mn:
    bs={"n_estimators":Integer(50,300),"max_depth":Integer(5,50),"min_samples_split":Integer(2,20)}
elif "XGB" in mn:
    bs={"n_estimators":Integer(50,300),"max_depth":Integer(3,10),"learning_rate":Real(0.01,0.3,prior="log-uniform")}
elif "GB" in mn:
    bs={"n_estimators":Integer(50,300),"max_depth":Integer(3,10),"learning_rate":Real(0.01,0.3,prior="log-uniform")}
else:
    bs={}
if bs:
    bsv=BayesSearchCV(mm,bs,n_iter=20,cv=5,scoring="roc_auc",random_state=42,n_jobs=-1)
    bsv.fit(X_train,y_train)
    mfinal=bsv.best_estimator_
    print("  ROC-AUC: %.4f" %% bsv.best_score_)
else:
    mfinal=mm
    print("  Sin tuning")'''))

cells2.append(nbformat.v4.new_markdown_cell('## 8. Test'))

cells2.append(nbformat.v4.new_code_cell('''y_pred=mfinal.predict(X_test)
y_prob=mfinal.predict_proba(X_test)[:,1]
acc=accuracy_score(y_test,y_pred)
pre=precision_score(y_test,y_pred)
rec=recall_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
auc=roc_auc_score(y_test,y_prob)
print("Accuracy: %.4f" %% acc)
print("Precision: %.4f" %% pre)
print("Recall: %.4f" %% rec)
print("F1: %.4f" %% f1)
print("ROC-AUC: %.4f" %% auc)
print("\\nReporte:")
print(classification_report(y_test,y_pred,target_names=["No Match","Match"]))
fig,ax=plt.subplots(1,2,figsize=(14,6))
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d",cmap="RdBu_r",ax=ax[0])
ax[0].set_title("Confusion",fontweight="bold")
fpr,tpr,_=roc_curve(y_test,y_prob)
ax[1].plot(fpr,tpr,color="#FF1493",lw=2,label="ROC AUC=%.3f" %% auc)
ax[1].plot([0,1],[0,1],"k--",lw=2)
ax[1].set_xlabel("Falsos Positivos")
ax[1].set_ylabel("Verdaderos Positivos")
ax[1].set_title("ROC",fontweight="bold")
ax[1].legend()
plt.tight_layout()
plt.savefig("../reports/evaluacion_final.png",dpi=300,bbox_inches="tight")
plt.show()'''))

cells2.append(nbformat.v4.new_markdown_cell('## 9. Feature Importance'))

cells2.append(nbformat.v4.new_code_cell('''if hasattr(mfinal,"feature_importances_"):
    imp=pd.DataFrame({"Feature":fn,"Importancia":mfinal.feature_importances_}).sort_values("Importancia",ascending=False)
    print(imp.head(15))
    plt.figure(figsize=(10,8))
    top=imp.head(15)
    plt.barh(range(len(top)),top["Importancia"].values,color="#FF1493")
    plt.yticks(range(len(top)),top["Feature"].values)
    plt.xlabel("Importancia")
    plt.title("Feature Importance",fontweight="bold")
    plt.tight_layout()
    plt.savefig("../reports/feature_importance.png",dpi=300,bbox_inches="tight")
    plt.show()'''))

cells2.append(nbformat.v4.new_markdown_cell('## 10. Guardar'))

cells2.append(nbformat.v4.new_code_cell('''os.makedirs("../models",exist_ok=True)
joblib.dump(mfinal,"../models/pipeline_match_predictor.pkl")
print("Pipeline guardado")
metricas={"modelo":mn,"accuracy":acc,"precision":pre,"recall":rec,"f1":f1,"roc_auc":auc,"n_features":len(fn)}
with open("../models/metricas_finales.pkl","wb") as f:
    pickle.dump(metricas,f)
print("Metricas guardadas")
print("\\n=== COMPLETADO ===")
print("Modelo: %s" %% mn)
print("ROC-AUC: %.4f" %% auc)
print("Accuracy: %.4f" %% acc)'''))

nb2.cells = cells2
with open("notebooks/02_modelamiento.ipynb","w") as f:
    nbformat.write(nb2,f)
print("Notebook 2 OK")

print("\\n=== AMBOS NOTEBOOKS CREADOS ===")
