"""Streamlit page for supervised ML workflows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from etl_studio.app import setup_page
from etl_studio.app.data import fetch, fetch_table_csv
from etl_studio.ai import model

setup_page("Modelo Predictivo · ETL Studio")


def render_hyperparameters(model_name: str, task_type: str) -> dict:
    """Render hyperparameter controls dynamically based on model."""
    models_dict = model.get_available_models(task_type)
    model_config = models_dict[model_name]
    params = {}
    
    if not model_config["params"]:
        st.info("Este modelo no tiene hiperparámetros configurables")
        return params
    
    st.markdown("##### Hiperparámetros")
    
    for param_name, param_config in model_config["params"].items():
        if param_config["type"] == "slider":
            params[param_name] = st.slider(
                param_name,
                min_value=param_config["min"],
                max_value=param_config["max"],
                value=param_config["default"],
                step=param_config.get("step", 1),
            )
        elif param_config["type"] == "select":
            params[param_name] = st.selectbox(
                param_name, param_config["options"], index=param_config["options"].index(param_config["default"])
            )
    
    return params


def plot_confusion_matrix(cm: list[list[int]]) -> go.Figure:
    """Create confusion matrix heatmap."""
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[f"Pred {i}" for i in range(len(cm))],
            y=[f"True {i}" for i in range(len(cm))],
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
        )
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400,
    )
    return fig


def plot_feature_importance(importance_data: list[dict]) -> go.Figure:
    """Plot feature importance."""
    df = pd.DataFrame(importance_data).head(15)
    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        title="Top 15 Feature Importance",
        labels={"importance": "Importance", "feature": "Feature"},
    )
    fig.update_layout(height=500)
    return fig


def plot_regression_results(y_test: list, y_pred: list) -> go.Figure:
    """Plot actual vs predicted for regression."""
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=y_pred,
            mode="markers",
            name="Predictions",
            marker=dict(size=8, opacity=0.6),
        )
    )
    
    # Perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Perfect Prediction",
            line=dict(color="red", dash="dash"),
        )
    )
    
    fig.update_layout(
        title="Actual vs Predicted Values",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        height=500,
    )
    
    return fig


def show() -> None:
    """Render the model training and inference workspace."""
    
    st.header("Machine Learning Model Training")
    
    # Seleccionar dataset de Gold
    st.subheader("Dataset Selection")
    
    gold_tables, is_mock = fetch("gold", "tables")
    
    if not gold_tables:
        st.warning("No hay tablas disponibles en Gold. Crea primero datasets en Gold Layer.")
        if st.button("Ir a Gold", icon=":material/arrow_forward:"):
            st.switch_page("pages/3_Gold.py")
        return
    
    table_names = [t["name"] for t in gold_tables]
    selected_table = st.selectbox("Selecciona el dataset:", table_names, key="dataset_select")
    
    if not selected_table:
        return
    
    # Cargar datos
    with st.spinner("Cargando dataset..."):
        df, _ = fetch_table_csv("gold", selected_table)
    
    if df is None:
        st.error("No se pudo cargar el dataset")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Filas", f"{len(df):,}")
    with col2:
        st.metric("Columnas", len(df.columns))
    
    with st.expander("Vista previa del dataset", expanded=False):
        st.dataframe(df.head(20), use_container_width=True, height=300)
    
    st.divider()
    
    # Configuración del modelo
    st.subheader("Model Configuration")
    
    col_task, col_model = st.columns(2)
    
    with col_task:
        task_type = st.selectbox(
            "Tipo de tarea:",
            ["classification", "regression"],
            format_func=lambda x: "Clasificación" if x == "classification" else "Regresión",
            key="task_select",
        )
    
    with col_model:
        models_dict = model.get_available_models(task_type)
        model_name = st.selectbox("Modelo:", list(models_dict.keys()), key="model_select")
    
    st.divider()
    
    # Feature Selection y Target
    st.subheader("Feature Selection & Target")
    
    col_target, col_features = st.columns([1, 2])
    
    with col_target:
        target_column = st.selectbox("Variable objetivo (Target):", df.columns.tolist(), key="target_select")
    
    with col_features:
        available_features = [col for col in df.columns if col != target_column]
        
        use_all_features = st.checkbox("Usar todas las features", value=True, key="use_all_features")
        
        if use_all_features:
            selected_features = None
            st.info(f"✓ Usando todas las features ({len(available_features)})")
        else:
            selected_features = st.multiselect(
                "Selecciona las features a usar:",
                available_features,
                default=available_features[:5] if len(available_features) > 5 else available_features,
                key="features_multiselect",
            )
            if not selected_features:
                st.warning("⚠️ Debes seleccionar al menos una feature")
                return
    
    st.divider()
    
    # Hyperparameters y opciones avanzadas
    col_hyper, col_advanced = st.columns([1, 1])
    
    with col_hyper:
        params = render_hyperparameters(model_name, task_type)
    
    with col_advanced:
        st.markdown("##### Opciones Avanzadas")
        test_size = st.slider("Proporción de test:", 0.1, 0.5, 0.2, 0.05, key="test_size_slider")
        use_cv = st.checkbox("Usar Cross-Validation (5-fold)", value=True, key="use_cv_checkbox")
    
    st.divider()
    
    # Entrenar modelo
    st.subheader("Model Training")
    
    if st.button("Train Model", type="primary", use_container_width=True, key="train_button"):
        
        # Validations
        if not selected_features and not use_all_features:
            st.error("Debes seleccionar features o marcar 'Usar todas las features'")
            return
        
        # Training
        with st.spinner("Entrenando modelo... Esto puede tardar unos minutos."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Preparando datos...")
                progress_bar.progress(20)
                
                status_text.text(f"Entrenando {model_name}...")
                progress_bar.progress(40)
                
                metrics = model.train_model(
                    df=df,
                    target_column=target_column,
                    model_name=model_name,
                    task_type=task_type,
                    params=params,
                    selected_features=selected_features,
                    test_size=test_size,
                    use_cross_validation=use_cv,
                )
                
                progress_bar.progress(80)
                status_text.text("Calculando métricas...")
                
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                
                st.success("✅ Modelo entrenado correctamente")
                
                # Store metrics in session state
                st.session_state["last_metrics"] = metrics
                st.session_state["last_task_type"] = task_type
                
            except Exception as e:
                st.error(f"❌ Error al entrenar: {str(e)}")
                return
    
    # Mostrar resultados
    if "last_metrics" in st.session_state:
        st.divider()
        st.subheader("Training Results")
        
        metrics = st.session_state["last_metrics"]
        task = st.session_state["last_task_type"]
        
        # Métricas principales
        st.markdown("##### Metrics")
        
        if task == "classification":
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy (Train)", f"{metrics['accuracy_train']:.3f}")
            with col2:
                st.metric("Accuracy (Test)", f"{metrics['accuracy_test']:.3f}")
            with col3:
                st.metric("Precision", f"{metrics['precision']:.3f}")
            with col4:
                st.metric("Recall", f"{metrics['recall']:.3f}")
            
            col5, col6 = st.columns(2)
            with col5:
                st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
            with col6:
                if use_cv:
                    st.metric("CV Score", f"{metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
        
        else:  # regression
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MAE (Test)", f"{metrics['mae_test']:.4f}")
            with col2:
                st.metric("RMSE (Test)", f"{metrics['rmse_test']:.4f}")
            with col3:
                st.metric("R² (Test)", f"{metrics['r2_test']:.3f}")
            
            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric("MAE (Train)", f"{metrics['mae_train']:.4f}")
            with col5:
                st.metric("RMSE (Train)", f"{metrics['rmse_train']:.4f}")
            with col6:
                st.metric("R² (Train)", f"{metrics['r2_train']:.3f}")
        
        # Visualizaciones
        st.markdown("##### Visualizations")
        
        if task == "classification":
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                if "confusion_matrix" in metrics:
                    fig_cm = plot_confusion_matrix(metrics["confusion_matrix"])
                    st.plotly_chart(fig_cm, use_container_width=True)
            
            with col_viz2:
                if "feature_importance" in metrics:
                    fig_fi = plot_feature_importance(metrics["feature_importance"])
                    st.plotly_chart(fig_fi, use_container_width=True)
        
        # Cross-validation scores
        if use_cv and "cv_scores" in metrics:
            st.markdown("##### Cross-Validation Scores")
            cv_df = pd.DataFrame(
                {"Fold": range(1, len(metrics["cv_scores"]) + 1), "Score": metrics["cv_scores"]}
            )
            fig_cv = px.bar(cv_df, x="Fold", y="Score", title="Cross-Validation Scores by Fold")
            st.plotly_chart(fig_cv, use_container_width=True)
        
        # Model info
        with st.expander("Model Information"):
            st.json(
                {
                    "Model Path": metrics["model_path"],
                    "Train Samples": metrics["train_samples"],
                    "Test Samples": metrics["test_samples"],
                    "Features Used": metrics["features_used"],
                }
            )
    
    st.divider()
    
    # Predicción
    st.subheader("Make Predictions")
    
    model_files = list(Path("models").glob("*.pkl")) if Path("models").exists() else []
    
    if not model_files:
        st.info("No hay modelos entrenados disponibles")
    else:
        selected_model_path = st.selectbox(
            "Selecciona el modelo a utilizar:", [str(m) for m in model_files], key="predict_model_select"
        )
        
        # Load model info
        model_info = model.load_model_info(selected_model_path)
        
        with st.expander("Model Info"):
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("Model", model_info["model_name"])
            with col_info2:
                st.metric("Task", model_info["task_type"])
            with col_info3:
                st.metric("Target", model_info["target"])
        
        if st.button("Generate Predictions", use_container_width=True, key="predict_button"):
            with st.spinner("Generando predicciones..."):
                try:
                    predictions = model.predict(df, selected_model_path)
                    
                    st.success(f"✅ {len(predictions)} predicciones generadas")
                    
                    # Show predictions
                    st.dataframe(predictions.head(50), use_container_width=True, height=400)
                    
                    # Download button
                    csv = predictions.to_csv(index=False)
                    st.download_button(
                        "📥 Download Predictions (CSV)",
                        csv,
                        "predictions.csv",
                        "text/csv",
                        use_container_width=True,
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error al predecir: {str(e)}")


if __name__ == "__main__":
    show()