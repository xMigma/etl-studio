"""Streamlit page for supervised ML workflows."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from etl_studio.app import setup_page
from etl_studio.app.data import fetch, fetch_table_csv
from etl_studio.ai import model
from pathlib import Path

setup_page("Modelo Predictivo · ETL Studio")


def show() -> None:
    """Render the model training and inference workspace."""

    st.header("ML Model Training")
    
    # Seleccionar dataset de Gold
    st.subheader("1. Seleccionar Dataset")
    
    gold_tables, is_mock = fetch("gold", "tables")
    
    if not gold_tables:
        st.warning("No hay tablas disponibles en Gold. Crea primero datasets en Gold Layer.")
        if st.button("Ir a Gold"):
            st.switch_page("pages/3_Gold.py")
        return
    
    table_names = [t["name"] for t in gold_tables]
    selected_table = st.selectbox("Dataset:", table_names)
    
    if not selected_table:
        return
    
    # Cargar datos
    df, _ = fetch_table_csv("gold", selected_table)
    
    if df is None:
        st.error("No se pudo cargar el dataset")
        return
    
    st.success(f"Dataset cargado: {len(df)} filas, {len(df.columns)} columnas")
    
    with st.expander("Vista previa del dataset"):
        st.dataframe(df.head(10), use_container_width=True)
    
    st.divider()
    
    # Configuración del modelo
    st.subheader("2. Configuración del Modelo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_column = st.selectbox(
            "Variable objetivo (target):",
            df.columns.tolist()
        )
    
    with col2:
        task_type = st.selectbox(
            "Tipo de tarea:",
            ["Clasificación", "Regresión"]
        )
    
    # Hiperparámetros
    with st.expander("Hiperparámetros"):
        col_hp1, col_hp2 = st.columns(2)
        with col_hp1:
            n_estimators = st.slider("N° de árboles", 10, 500, 100)
        with col_hp2:
            max_depth = st.slider("Profundidad máxima", 3, 50, 10)
    
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth
    }
    
    st.divider()
    
    # Entrenar modelo
    st.subheader("3. Entrenar Modelo")
    
    if st.button("🚀 Entrenar Modelo", type="primary", use_container_width=True):
        with st.spinner("Entrenando modelo..."):
            try:
                metrics = model.train_model(df, target_column, params)
                
                st.success("✅ Modelo entrenado correctamente")
                
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                with col_m2:
                    st.metric("Train Samples", metrics['train_samples'])
                with col_m3:
                    st.metric("Test Samples", metrics['test_samples'])
                
                st.info(f"Modelo guardado en: `{metrics['model_path']}`")
                
            except Exception as e:
                st.error(f"Error al entrenar: {e}")
    
    st.divider()
    
    # Predicción
    st.subheader("4. Realizar Predicciones")
    
    model_files = list(Path("models").glob("*.pkl")) if Path("models").exists() else []
    
    if not model_files:
        st.info("No hay modelos entrenados disponibles")
    else:
        selected_model = st.selectbox(
            "Modelo a utilizar:",
            [str(m) for m in model_files]
        )
        
        if st.button("🔮 Predecir", use_container_width=True):
            with st.spinner("Generando predicciones..."):
                try:
                    # Eliminar target si existe
                    df_pred = df.drop(columns=[target_column], errors='ignore')
                    
                    predictions = model.predict(df_pred, selected_model)
                    
                    st.success("✅ Predicciones generadas")
                    st.dataframe(predictions.head(20), use_container_width=True)
                    
                    # Opción de descarga
                    csv = predictions.to_csv(index=False)
                    st.download_button(
                        "📥 Descargar predicciones",
                        csv,
                        "predictions.csv",
                        "text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error al predecir: {e}")


if __name__ == "__main__":
    show()