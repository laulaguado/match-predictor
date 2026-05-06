import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuración de página
st.set_page_config(
    page_title="Predictor de Match - Speed Dating",
    page_icon="💕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar modelos y datos
@st.cache_resource
def load_models():
    try:
        with open('models/pipeline_match_predictor.pkl', 'rb') as f:
            pipeline = pickle.load(f)
        with open('models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        with open('models/metricas_finales.pkl', 'rb') as f:
            metricas = pickle.load(f)
        return pipeline, feature_names, metricas
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None, None, None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/Speed Dating Data.csv', encoding='latin-1')
        df_clean = pd.read_csv('data/data_prepared.csv')
        return df, df_clean
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None, None

pipeline, feature_names, metricas = load_models()
df_original, df_clean = load_data()

# CSS personalizado para tema oscuro romántico
st.markdown("""
<style>
    /* Tema oscuro romántico */
    .stApp {
        background: linear-gradient(135deg, #1a0a17 0%, #2d1b3d 50%, #1a0a17 100%);
        color: #ffe4ec;
    }
    .css-1d391kg {
        background: linear-gradient(135deg, #1a0a17 0%, #2d1b3d 100%);
    }
    .st-bb {
        background-color: transparent;
    }
    .st-at {
        background-color: #FF1493;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FF1493 !important;
        text-shadow: 0 0 10px rgba(255, 20, 147, 0.5);
    }
    .stMetric {
        background: linear-gradient(135deg, rgba(255, 20, 147, 0.2), rgba(255, 182, 193, 0.1));
        border: 1px solid #FF1493;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 0 20px rgba(255, 20, 147, 0.3);
    }
    .stMetric .label {
        color: #FFB6C1 !important;
    }
    .stMetric .value {
        color: #FF1493 !important;
        font-weight: bold;
    }
    .stButton>button {
        background: linear-gradient(135deg, #FF1493, #FF69B4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(255, 20, 147, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 20, 147, 0.6);
    }
    .stSlider>div>div>div {
        background: linear-gradient(90deg, #FF1493, #FFB6C1);
    }
    .stSelectbox>div>div>div, .stNumberInput>div>div>div {
        background: rgba(255, 20, 147, 0.1);
        border: 1px solid #FF1493;
        color: #ffe4ec;
    }
    .css-1n76uvr {
        color: #ffe4ec;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a0a17, #2d1b3d);
        color: #ffe4ec;
    }
    .sidebar .sidebar-content h1, .sidebar .sidebar-content h2, .sidebar .sidebar-content h3 {
        color: #FF1493 !important;
    }
    /* Gauge chart container */
    .gauge-container {
        text-align: center;
        padding: 20px;
    }
    /* Custom CSS cards */
    .card-rosa {
        background: linear-gradient(135deg, rgba(255, 20, 147, 0.15), rgba(255, 182, 193, 0.1));
        border: 1px solid #FF1493;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(255, 20, 147, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/ios-filled/100/FF1493/heart.png", width=100)
    st.title("💕 Predictor de Match")
    st.markdown("---")
    pagina = st.radio(
        "Navegación",
        ["🏠 Inicio", "🎯 Predice tu Match", "📊 Explorar Dataset", "📋 Sobre el Modelo"],
        index=0
    )
    st.markdown("---")
    st.caption("Speed Dating Columbia University")

# Página 1: Inicio
if pagina == "🏠 Inicio":
    st.title("🏠 Métricas del Modelo")
    st.markdown("---\n")
    
    if metricas:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Modelo Seleccionado", metricas.get('modelo', 'N/A'))
        with col2:
            st.metric("ROC-AUC Test", f"{metricas.get('roc_auc_test', 0):.4f}")
        with col3:
            st.metric("Accuracy Test", f"{metricas.get('accuracy', 0):.4f}")
        with col4:
            st.metric("F1-Score Test", f"{metricas.get('f1', 0):.4f}")
        
        st.markdown("---\n")
        
        col5, col6 = st.columns(2)
        with col5:
            st.markdown("### Precision y Recall")
            st.metric("Precision", f"{metricas.get('precision', 0):.4f}")
            st.metric("Recall", f"{metricas.get('recall', 0):.4f}")
        with col6:
            st.markdown("### Información del Dataset")
            st.write(f"- Features: {metricas.get('n_features', 0)}")
            st.write(f"- Muestras Entrenamiento: {metricas.get('n_train', 0)}")
            st.write(f"- Muestras Test: {metricas.get('n_test', 0)}")
            st.write(f"- Balance Train: {metricas.get('train_balance', 0):.2%}")
            st.write(f"- Balance Test: {metricas.get('test_balance', 0):.2%}")
    else:
        st.warning("No se pudieron cargar las métricas")
    
    st.markdown("---\n")
    st.subheader("Desempeño del Pipeline")
    
    # Gráfico de radar de métricas
    if metricas:
        categories = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
        values = [
            metricas.get('accuracy', 0),
            metricas.get('precision', 0),
            metricas.get('recall', 0),
            metricas.get('f1', 0),
            metricas.get('roc_auc_test', 0)
        ]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(255, 20, 147, 0.3)',
            line=dict(color='#FF1493', width=2),
            name='Modelo'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    gridcolor='#FFB6C1',
                    gridwidth=1
                ),
                angularaxis=dict(gridcolor='#FFB6C1')
            ),
            showlegend=False,
            title="Radar de Métricas - Modelo Final",
            title_font_color='#FF1493',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

# Página 2: Predice tu Match
elif pagina == "🎯 Predice tu Match":
    st.title("🎯 Predice tu Match")
    st.markdown("Ajusta los valores para predecir la probabilidad de match\n")
    
    if pipeline and feature_names:
        # Calcular medianas del dataset para features no mostrados
        if df_clean is not None:
            medianas_features = {}
            for feat in feature_names:
                if feat in df_clean.columns:
                    medianas_features[feat] = df_clean[feat].median()
        
        # Inputs del usuario
        st.markdown("### Características Principales")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            attr = st.slider("Atractivo", 0, 10, 5, help="¿Qué tan atractivo te consideras?")
            sinc = st.slider("Sinceridad", 0, 10, 5, help="¿Qué tan sincero eres?")
            intel = st.slider("Inteligencia", 0, 10, 5, help="¿Qué tan inteligente te consideras?")
            fun = st.slider("Diversión", 0, 10, 5, help="¿Qué tan divertido eres?")
        
        with col2:
            amb = st.slider("Ambición", 0, 10, 5, help="¿Qué tan ambicioso eres?")
            shar = st.slider("Intereses Compartidos", 0, 10, 5, help="Intereses compartidos con la pareja")
            like = st.slider("Gusto por el Compañero", 0, 10, 5, help="¿Cuánto te gusta la otra persona?")
            prob = st.slider("Probabilidad Esperada", 0, 10, 5, help="Probabilidad de una segunda cita")
        
        with col3:
            age = st.slider("Edad", 18, 60, 25, help="Tu edad")
            age_o = st.slider("Edad del Compañero", 18, 60, 25, help="Edad de la otra persona")
            samerace = st.selectbox("Misma Raza", [0, 1], format_func=lambda x: "Sí" if x else "No")
            met = st.selectbox("Se Han Conocido Antes", [0, 1], format_func=lambda x: "Sí" if x else "No")
        
        st.markdown("### Otras Características")
        col4, col5 = st.columns(2)
        
        with col4:
            goal = st.selectbox("Objetivo de las Citas", [0, 1, 2, 3, 4, 5, 6], 
                              format_func=lambda x: ["Diversión", "Conocer gente", "Buscar pareja", 
                                                    "Matrimonio", "Amistad", "Pasar el rato", "Otro"][x])
            date = st.slider("Frecuencia de Citas", 0, 10, 5)
            go_out = st.slider("Frecuencia de Salir", 0, 10, 5)
        
        with col5:
            race = st.selectbox("Tu Raza", [0, 1, 2, 3, 4, 5, 6, 7, 8], format_func=lambda x: f"Raza {x}")
            race_o = st.selectbox("Raza del Compañero", [0, 1, 2, 3, 4, 5, 6, 7, 8], format_func=lambda x: f"Raza {x}")
            dec = st.selectbox("Tu Decisión", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
            dec_o = st.selectbox("Decisión del Compañero", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
        
        # Construir el input para el modelo
        input_data = {}
        for feat in feature_names:
            if feat in ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar', 'like', 'prob', 'age', 'age_o', 
                       'samerace', 'goal', 'date', 'go_out', 'race', 'race_o', 'dec', 'dec_o', 'met']:
                input_data[feat] = [locals()[feat]]
            else:
                # Usar medianas para otras features
                input_data[feat] = [medianas_features.get(feat, 0)]
        
        input_df = pd.DataFrame(input_data)
        
        # Botón de predicción
        st.markdown("---\n")
        if st.button("Predecir Match 💕", type="primary"):
            try:
                # Predecir
                prediccion = pipeline.predict(input_df)
                probabilidad = pipeline.predict_proba(input_df)[0]
                prob_match = probabilidad[1]
                
                # Mostrar resultados
                st.markdown("### Resultado de la Predicción")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Gauge chart
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob_match * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Probabilidad de Match", 'font': {'color': '#FF1493', 'size': 20}},
                        gauge={
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': '#FFB6C1'},
                            'bar': {'color': '#FF1493'},
                            'bgcolor': 'rgba(0,0,0,0)',
                            'borderwidth': 2,
                            'bordercolor': '#FF1493',
                            'steps': [
                                {'range': [0, 30], 'color': 'rgba(255, 20, 147, 0.1)'},
                                {'range': [30, 70], 'color': 'rgba(255, 182, 193, 0.3)'},
                                {'range': [70, 100], 'color': 'rgba(255, 20, 147, 0.5)'}
                            ],
                            'threshold': {
                                'line': {'color': '#FF1493', 'width': 4},
                                'thickness': 0.75,
                                'value': prob_match * 100
                            }
                        }
                    ))
                    fig_gauge.update_layout(
                        height=300,
                        paper_bgcolor='rgba(0,0,0,0)',
                        font={'color': '#FFB6C1'}
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                with col2:
                    # Radar chart vs promedio de matches exitosos
                    if df_original is not None:
                        # Filtrar matches exitosos
                        matches_exitosos = df_original[df_original['match'] == 1]
                        if len(matches_exitosos) > 0:
                            promedio_exitosos = {
                                'attr': matches_exitosos['attr'].mean(),
                                'sinc': matches_exitosos['sinc'].mean(),
                                'intel': matches_exitosos['intel'].mean(),
                                'fun': matches_exitosos['fun'].mean(),
                                'amb': matches_exitosos['amb'].mean(),
                                'shar': matches_exitosos['shar'].mean()
                            }
                            
                            # Normalizar a 0-10
                            tu_valor = [attr, sinc, intel, fun, amb, shar]
                            promedio_val = [promedio_exitosos.get(f, 5) for f in ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar']]
                            
                            fig_radar = go.Figure()
                            fig_radar.add_trace(go.Scatterpolar(
                                r=tu_valor + [tu_valor[0]],
                                theta=['Atractivo', 'Sinceridad', 'Inteligencia', 'Diversión', 'Ambición', 'Intereses', 'Atractivo'],
                                fill='toself',
                                fillcolor='rgba(255, 20, 147, 0.3)',
                                line=dict(color='#FF1493', width=2),
                                name='Tú'
                            ))
                            fig_radar.add_trace(go.Scatterpolar(
                                r=promedio_val + [promedio_val[0]],
                                theta=['Atractivo', 'Sinceridad', 'Inteligencia', 'Diversión', 'Ambición', 'Intereses', 'Atractivo'],
                                fill='toself',
                                fillcolor='rgba(255, 182, 193, 0.2)',
                                line=dict(color='#FFB6C1', width=2, dash='dash'),
                                name='Promedio Matches Exitosos'
                            ))
                            fig_radar.update_layout(
                                polar=dict(
                                    radialaxis=dict(visible=True, range=[0, 10], gridcolor='#FFB6C1'),
                                    angularaxis=dict(gridcolor='#FFB6C1')
                                ),
                                showlegend=True,
                                title="Comparación vs Promedio de Matches Exitosos",
                                title_font_color='#FF1493',
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                legend=dict(font=dict(color='#FFB6C1'))
                            )
                            st.plotly_chart(fig_radar, use_container_width=True)
                
                # Mensaje del resultado
                st.markdown("---\n")
                st.markdown("### Análisis del Resultado")
                
                if prediccion[0] == 1:
                    st.success("💕 ¡Felicidades! Alta probabilidad de MATCH")
                    st.info(f"Probabilidad: {prob_match*100:.1f}% - Parece que hay buena química")
                elif prob_match > 0.5:
                    st.warning("🤔 Posible Match - Hay oportunidad")
                    st.info(f"Probabilidad: {prob_match*100:.1f}% - Podría funcionar con buena conversación")
                else:
                    st.error("💔 Probabilidad Baja de Match")
                    st.info(f"Probabilidad: {prob_match*100:.1f}% - Quizás no sea la mejor combinación")
                
                # Detalles adicionales
                with st.expander("Ver detalles de la predicción"):
                    st.write("Probabilidad de Match:", f"{prob_match:.4f}")
                    st.write("Probabilidad de No-Match:", f"{probabilidad[0]:.4f}")
                    st.write("\nValores ingresados:")
                    st.json(input_df.to_dict())
                    
            except Exception as e:
                st.error(f"Error en la predicción: {e}")
                st.code(str(e))
    else:
        st.warning("No se pudieron cargar los modelos necesarios")

# Página 3: Explorar Dataset
elif pagina == "📊 Explorar Dataset":
    st.title("📊 Explorar Dataset")
    st.markdown("Análisis interactivo del dataset de Speed Dating\n")
    
    if df_original is not None:
        # Filtros
        st.sidebar.header("Filtros")
        gender_filter = st.sidebar.multiselect(
            "Género",
            options=df_original['gender'].unique(),
            default=df_original['gender'].unique()
        )
        match_filter = st.sidebar.multiselect(
            "Resultado",
            options=df_original['match'].unique(),
            default=df_original['match'].unique()
        )
        
        df_filtered = df_original[
            (df_original['gender'].isin(gender_filter)) & 
            (df_original['match'].isin(match_filter))
        ]
        
        st.write(f"Mostrando {len(df_filtered)} de {len(df_original)} registros")
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Registros", len(df_filtered))
        with col2:
            tasa_match = df_filtered['match'].mean() * 100
            st.metric("Tasa de Match", f"{tasa_match:.1f}%")
        with col3:
            edad_prom = df_filtered['age'].mean()
            st.metric("Edad Promedio", f"{edad_prom:.1f}")
        with col4:
            atract_prom = df_filtered['attr'].mean()
            st.metric("Atractivo Promedio", f"{atract_prom:.2f}")
        
        st.markdown("---\n")
        
        # Gráficos
        tab1, tab2, tab3, tab4 = st.tabs(["Distribuciones", "Correlaciones", "Comparativas", "Datos"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Distribución de Edades")
                fig = px.histogram(df_filtered, x='age', color='match', 
                                 color_discrete_map={0: '#FFB6C1', 1: '#FF1493'},
                                 nbins=20, title="Distribución por Edades")
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#FFB6C1'),
                    title_font_color='#FF1493'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Distribución de Atractivo")
                fig = px.histogram(df_filtered, x='attr', color='match',
                                 color_discrete_map={0: '#FFB6C1', 1: '#FF1493'},
                                 title="Atractivo vs Match")
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#FFB6C1'),
                    title_font_color='#FF1493'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Boxplot Atractivo")
                fig = px.box(df_filtered, x='match', y='attr', color='match',
                           color_discrete_map={0: '#FFB6C1', 1: '#FF1493'},
                           labels={'match': 'Match', 'attr': 'Atractivo'},
                           title="Atractivo por Resultado")
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#FFB6C1'),
                    title_font_color='#FF1493',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col4:
                st.subheader("Scatter Atractivo-Edad")
                fig = px.scatter(df_filtered.sample(min(500, len(df_filtered))), 
                               x='age', y='attr', color='match',
                               color_discrete_map={0: '#FFB6C1', 1: '#FF1493'},
                               hover_data=['gender', 'like', 'prob'],
                               title="Atractivo vs Edad")
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#FFB6C1'),
                    title_font_color='#FF1493'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Matriz de Correlación")
            numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
            corr = df_filtered[numeric_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale=[[0, '#FFB6C1'], [0.5, '#1a0a17'], [1, '#FF1493']],
                text=corr.values.round(2),
                texttemplate='%{text}',
                textfont=dict(color='white')
            ))
            fig.update_layout(
                title="Correlación entre Variables Numéricas",
                title_font_color='#FF1493',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#FFB6C1'),
                xaxis=dict(gridcolor='#FFB6C1'),
                yaxis=dict(gridcolor='#FFB6C1')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Promedio por Match")
                metricas_match = df_filtered.groupby('match')[['attr', 'sinc', 'intel', 'fun', 'amb', 'like', 'prob']].mean()
                fig = go.Figure()
                for col in metricas_match.columns:
                    fig.add_trace(go.Bar(
                        x=metricas_match.index,
                        y=metricas_match[col],
                        name=col,
                        marker_color='#FF1493' if col in ['attr', 'like'] else '#FFB6C1'
                    ))
                fig.update_layout(
                    barmode='group',
                    title="Promedio de Métricas por Resultado",
                    title_font_color='#FF1493',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#FFB6C1'),
                    xaxis=dict(title="Match", gridcolor='#FFB6C1'),
                    yaxis=dict(gridcolor='#FFB6C1')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Distribución de Intenciones")
                goal_counts = df_filtered['goal'].value_counts()
                fig = px.pie(
                    values=goal_counts.values,
                    names=['Diversión', 'Conocer gente', 'Pareja', 'Matrimonio', 'Amistad', 'Pasar rato', 'Otro'],
                    title="Objetivos de las Citas",
                    color_discrete_sequence=['#FF1493', '#FF69B4', '#FFB6C1', '#DB7093', '#FFC0CB', '#FF1493', '#FF69B4']
                )
                fig.update_layout(
                    title_font_color='#FF1493',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#FFB6C1')
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Datos Filtrados")
            st.write(df_filtered)
            
            st.markdown("---\n")
            st.download_button(
                label="Descargar datos filtrados (CSV)",
                data=df_filtered.to_csv(index=False),
                file_name="speed_dating_filtered.csv",
                mime="text/csv",
                type="secondary"
            )
    else:
        st.warning("No se pudieron cargar los datos")

# Página 4: Sobre el Modelo
elif pagina == "📋 Sobre el Modelo":
    st.title("📋 Sobre el Modelo")
    st.markdown("Información detallada sobre el pipeline y el proceso de modelado\n")
    
    # Tabla CRISP-DM
    st.subheader("Proceso CRISP-DM")
    
    crisp_dm_data = {
        "Fase": [
            "Comprensión del Negocio",
            "Comprensión de los Datos",
            "Preparación de los Datos",
            "Modelado",
            "Evaluación",
            "Despliegue"
        ],
        "Descripción": [
            "Predecir la probabilidad de match en citas rápidas basándose en características personales y preferencias",
            "Análisis exploratorio del dataset Speed Dating de Columbia University (8379 registros, 195 variables iniciales)",
            "Eliminación de 16 variables irrelevantes, manejo de nulos (>50% eliminados, <50% imputados con mediana), winsorizing 5-95, eliminación de redundancias (corr>0.85) e irrelevancias (corr<0.02 con target)",
            "Entrenamiento de 7 modelos (Decision Tree, MLP, SVM, KNN, Random Forest, XGBoost, Gradient Boosting) con validación cruzada 10-fold estratificada",
            "Selección del mejor modelo por ROC-AUC, tuning con RandomizedSearchCV y BayesSearchCV, evaluación final en test",
            "Pipeline serializado listo para predicción en tiempo real"
        ],
        "Técnicas/Herramientas": [
            "Machine Learning Supervisado",
            "Pandas, NumPy, Matplotlib, Seaborn",
            "Pandas Profiling, Scikit-learn",
            "Scikit-learn, XGBoost",
            "Scikit-learn, SciPy (ANOVA, Tukey HSD)",
            "Pickle, Streamlit"
        ]
    }
    
    crisp_df = pd.DataFrame(crisp_dm_data)
    
    # Estilizar tabla
    st.dataframe(
        crisp_df.style.set_properties(
            **{
                'background-color': 'rgba(255, 20, 147, 0.1)',
                'color': '#FFB6C1',
                'border': '1px solid #FF1493',
                'padding': '10px'
            },
            subset=pd.IndexSlice[:, :]
        ).set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#FF1493'), ('color', 'white'), ('border', '1px solid #FF1493')]},
            {'selector': 'td', 'props': [('border', '1px solid #FF1493')]},
            {'selector': 'tr', 'props': [('background-color', 'rgba(255, 20, 147, 0.05)')]}
        ]),  
        height=400,
        use_container_width=True
    )
    
    st.markdown("---\n")
    
    # Métricas detalladas
    st.subheader("Métricas de Evaluación")
    
    if metricas:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Modelo Final Seleccionado")
            st.metric("", metricas.get('modelo', 'N/A'))
            
            st.markdown("#### Métricas en Test")
            metricas_df = pd.DataFrame({
                'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                'Valor': [
                    metricas.get('accuracy', 0),
                    metricas.get('precision', 0),
                    metricas.get('recall', 0),
                    metricas.get('f1', 0),
                    metricas.get('roc_auc_test', 0)
                ]
            })
            metricas_df['Valor'] = metricas_df['Valor'].apply(lambda x: f"{x:.4f}")
            st.table(metricas_df.style.set_properties(
                **{'background-color': 'rgba(255, 20, 147, 0.1)', 'color': '#FFB6C1'}
            ).set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#FF1493'), ('color', 'white')]}
            ]))
        
        with col2:
            st.markdown("#### Configuración de Validación")
            st.write("- **Validación Cruzada:** 10-fold estratificada")
            st.write("- **Métrica de optimización:** ROC-AUC")
            st.write("- **RandomizedSearchCV:** n_iter=10, cv=5")
            st.write("- **BayesSearchCV:** n_iter=20")
            st.write(f"- **Mejor ROC-AUC CV:** {metricas.get('cv_roc_auc_mean', 0):.4f}")
            
            st.markdown("#### Mejores Hiperparámetros")
            if 'mejores_parametros' in metricas:
                for param, value in metricas['mejores_parametros'].items():
                    st.write(f"- {param}: {value}")
    
    st.markdown("---\n")
    
    # Gráfico de comparación de modelos
    st.subheader("Comparación de Modelos (Validación Cruzada)")
    
    # Datos simulados de comparación (basados en el entrenamiento real)
    if pipeline and hasattr(pipeline, 'estimators_'):
        # Si es un VotingClassifier o similar
        pass
    
    # Mostrar información del pipeline
    st.markdown("#### Pipeline Utilizado")
    st.code(f"""Pipeline({type(pipeline).__name__})
    Steps: {len(pipeline.steps) if hasattr(pipeline, 'steps') else 'N/A'}
    Features de entrada: {len(feature_names) if feature_names else 0}
    Features seleccionadas: {metricas.get('n_features', 0) if metricas else 0}""")
    
    st.markdown("---\n")
    
    st.info("""
    **Notas del Proyecto:**
    - El modelo fue entrenado con datos históricos de citas rápidas de la Universidad de Columbia
    - Se priorizó el ROC-AUC como métrica principal debido al desbalance en la variable objetivo
    - La interpretabilidad se mantuvo mediante feature importance y análisis SHAP
    - El pipeline incluye preprocesamiento completo para predicción en tiempo real
    """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #FFB6C1;'>💕 Predictor de Match - Speed Dating Columbia University 2026</p>", unsafe_allow_html=True)