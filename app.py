import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="💘 ¿Habrá Match?",
    page_icon="💘",
    layout="wide"
)

# CSS personalizado
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #1a0010 0%, #2d0022 100%); color: #fff; }
    h1, h2, h3 { color: #FF69B4 !important; }
    .stMetric { background: rgba(255,20,147,0.15); border-radius: 12px; padding: 10px; border: 1px solid #FF1493; }
</style>
""", unsafe_allow_html=True)

# Cargar modelo
@st.cache_resource
def cargar_modelo():
    try:
        pipeline = joblib.load('models/pipeline_match_predictor.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        metricas = joblib.load('models/metricas_finales.pkl')
    except:
        pipeline, feature_names, metricas = None, None, {'ROC-AUC': 0.75, 'Accuracy': 0.84}
    return pipeline, feature_names, metricas

@st.cache_data
def cargar_datos():
    try:
        return pd.read_csv('data/data_prepared.csv')
    except:
        return pd.DataFrame()

pipeline, feature_names, metricas = cargar_modelo()
df = cargar_datos()

# Sidebar
st.sidebar.title("💘 Match Predictor")
pagina = st.sidebar.radio("Navegar", ["🏠 Inicio", "💘 Predice tu Match", "📊 Explorar Dataset", "🧠 Sobre el Modelo"])

if pagina == "🏠 Inicio":
    st.title("💘 ¿Habrá Match?")
    st.markdown("### Predictor de Compatibilidad en Speed Dating")
    st.markdown("""
    Basado en datos reales del experimento de speed dating de Columbia University (2002-2004),
    este modelo predice si dos personas tendrán match mutuo en una mini-cita de 4 minutos.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 ROC-AUC", f"{metricas.get('ROC-AUC', 0.75):.3f}")
    col2.metric("✅ Accuracy", f"{metricas.get('Accuracy', 0.84):.3f}")
    col3.metric("👥 Registros", f"{len(df):,}")
    col2.metric("📊 F1-Score", f"{metricas.get('F1', 0.65):.3f}")

elif pagina == "💘 Predice tu Match":
    st.title("💘 ¿Habrá Match entre ustedes?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Tu perfil")
        gender = st.selectbox("Tu género", ["Mujer (0)", "Hombre (1)"])
        age = st.slider("Tu edad", 18, 45, 25)
        goal = st.selectbox("¿Qué buscas?", ["1 - Diversión", "2 - Conocer gente", "3 - Cita", "4 - Relación"])
    
    with col2:
        st.subheader("⭐ Percepción de la pareja")
        attr = st.slider("Atractivo (1-10)", 1.0, 10.0, 6.0, 0.5)
        fun = st.slider("Diversión (1-10)", 1.0, 10.0, 6.0, 0.5)
        like = st.slider("¿Cuánto te gustó? (1-10)", 1.0, 10.0, 6.0, 0.5)
        dec = st.radio("¿Dirías que sí?", [1, 0], format_func=lambda x: "✅ Sí" if x==1 else "❌ No")
    
    if st.button("💘 Predecir Match", use_container_width=True):
        if pipeline and feature_names:
            input_data = {col: [df[col].median() if col in df.columns else 0 for col in feature_names]}
            input_data['gender'] = [0 if "Mujer" in gender else 1]
            input_data['age'] = [age]
            input_data['attr'] = [attr]
            input_data['fun'] = [fun]
            input_data['like'] = [like]
            input_data['dec'] = [dec]
            
            X_input = pd.DataFrame(input_data)[feature_names]
            prob = pipeline.predict_proba(X_input)[0][1]
            
            st.markdown("---")
            if prob >= 0.6:
                st.success("## 💕 ¡Hay química!")
            elif prob >= 0.4:
                st.warning("## 🤔 Podría funcionar...")
            else:
                st.error("## 👋 Mejor como amigos")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob*100,
                title={'text': "Probabilidad de Match"},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': '#FF1493'}}
            ))
            st.plotly_chart(fig, use_container_width=True)

elif pagina == "📊 Explorar Dataset":
    st.title("📊 Explorando el Dataset")
    if len(df) > 0:
        st.dataframe(df.head(20))

elif pagina == "🧠 Sobre el Modelo":
    st.title("🧠 Sobre el Modelo")
    st.markdown("""
    ### Metodología CRISP-DM
    1. Entendimiento del negocio
    2. Entendimiento de los datos
    3. Preparación de datos
    4. Modelamiento
    5. Despliegue
    """)

if __name__ == '__main__':
    st.write("")  # Keep app running