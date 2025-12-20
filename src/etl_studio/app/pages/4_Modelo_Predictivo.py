"""Streamlit page for supervised ML workflows with MLflow integration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from etl_studio.app import setup_page
from etl_studio.app.data import fetch, fetch_table_csv
from etl_studio.ai import model
from etl_studio.config import MLFLOW_TRACKING_URI_LOCAL

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


def render_encoding_section(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Render the encoding section and return the encoded dataframe."""
    st.subheader("Feature Encoding")
    
    encoding_key = f"encoding_config_{table_name}"
    if encoding_key not in st.session_state:
        st.session_state[encoding_key] = {}
    
    categorical_cols = model.get_categorical_columns(df)
    
    if not categorical_cols:
        st.info("✓ No hay columnas categóricas en el dataset")
        return df
    
    st.write(f"**Columnas categóricas detectadas:** {len(categorical_cols)}")
    
    col_config, col_preview = st.columns([1, 1])
    
    with col_config:
        st.markdown("##### Configurar Encoding")
        
        selected_col = st.selectbox(
            "Selecciona columna:",
            [""] + categorical_cols,
            key=f"enc_col_select_{table_name}"
        )
        
        if selected_col:
            encoding_type = st.selectbox(
                "Tipo de encoding:",
                ["One-Hot Encoding", "Label Encoding"],
                key=f"enc_type_select_{table_name}"
            )
            
            unique_values = df[selected_col].nunique()
            st.caption(f"Valores únicos: {unique_values}")
            
            if unique_values > 20:
                st.warning(f"Esta columna tiene {unique_values} valores únicos. One-Hot Encoding creará muchas columnas.")
            
            if st.button("Añadir Encoding", type="primary", key=f"add_enc_{table_name}"):
                enc_type = "onehot" if encoding_type == "One-Hot Encoding" else "label"
                st.session_state[encoding_key][selected_col] = enc_type
                st.rerun()
        
        st.divider()
        st.markdown("##### Encodings Aplicados")
        
        if st.session_state[encoding_key]:
            for col, enc_type in st.session_state[encoding_key].items():
                col_enc, col_del = st.columns([4, 1])
                with col_enc:
                    enc_name = "One-Hot" if enc_type == "onehot" else "Label"
                    st.text(f"{col}: {enc_name}")
                with col_del:
                    if st.button("", key=f"del_enc_{col}_{table_name}", help="Eliminar", icon=":material/delete:"):
                        del st.session_state[encoding_key][col]
                        st.rerun()
            
            if st.button("Limpiar todos", type="tertiary", use_container_width=True, key=f"clear_enc_{table_name}"):
                st.session_state[encoding_key] = {}
                st.rerun()
        else:
            st.caption("No hay encodings aplicados")
    
    with col_preview:
        st.markdown("##### Preview Columna Seleccionada")
        
        if selected_col:
            temp_config = {selected_col: "onehot" if encoding_type == "One-Hot Encoding" else "label"}
            
            try:
                df_preview, _ = model.apply_encoding(df, temp_config)
                
                tab_before, tab_after = st.tabs(["BEFORE", "AFTER"])
                
                with tab_before:
                    st.dataframe(
                        df[[selected_col]].head(15), 
                        use_container_width=True, 
                        height=300
                    )
                
                with tab_after:
                    if selected_col in df_preview.columns:
                        st.dataframe(
                            df_preview[[selected_col]].head(15),
                            use_container_width=True,
                            height=300
                        )
                    else:
                        encoded_cols = [c for c in df_preview.columns if c.startswith(f"{selected_col}_")]
                        st.dataframe(
                            df_preview[encoded_cols].head(15),
                            use_container_width=True,
                            height=300
                        )
                
            except Exception as e:
                st.error(f"Error en preview: {e}")
        else:
            st.info("Selecciona una columna para ver el preview")
    
    st.divider()
    
    if st.session_state[encoding_key]:
        st.markdown("##### Preview Dataset Completo con Encodings")
        
        try:
            df_encoded, encoders = model.apply_encoding(df, st.session_state[encoding_key])
            
            # Save encoders to session state for training
            st.session_state[f"encoders_{table_name}"] = encoders
            
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Columnas originales", len(df.columns))
            with col_m2:
                st.metric("Columnas después encoding", len(df_encoded.columns))
            with col_m3:
                delta = len(df_encoded.columns) - len(df.columns)
                st.metric("Diferencia", f"+{delta}" if delta > 0 else str(delta))
            
            tab_original, tab_encoded = st.tabs(["DATASET ORIGINAL", "DATASET ENCODED"])
            
            with tab_original:
                st.dataframe(df.head(20), use_container_width=True, height=400)
            
            with tab_encoded:
                st.dataframe(df_encoded.head(20), use_container_width=True, height=400)
            
            return df_encoded
            
        except Exception as e:
            st.error(f"Error al aplicar encodings: {e}")
            return df
    
    return df


def render_mlflow_section():
    """Render MLflow tracking section."""
    st.subheader("MLflow Tracking")
    
    # MLflow UI link
    st.markdown(f"""
    **[Open MLflow UI]({MLFLOW_TRACKING_URI_LOCAL})** - View all experiments, metrics, and models
    """)
    
    st.divider()
    
    # Recent runs
    st.markdown("##### Recent Training Runs")
    
    try:
        runs_df = model.get_mlflow_runs(limit=20)
        
        if runs_df.empty:
            st.info("No hay runs disponibles aún. Entrena un modelo primero.")
        else:
            # Display runs table
            display_cols = ["run_name", "model_name", "task_type", "accuracy_test", "r2_test", "mae_test", "start_time"]
            display_df = runs_df[display_cols].copy()
            
            # Format metrics
            for col in ["accuracy_test", "r2_test", "mae_test"]:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=300
            )
            
            # Run details
            with st.expander("Ver detalles de un run"):
                selected_run = st.selectbox(
                    "Selecciona un run:",
                    runs_df["run_id"].tolist(),
                    format_func=lambda x: runs_df[runs_df["run_id"] == x]["run_name"].values[0] or x[:8]
                )
                
                if selected_run:
                    run_info = runs_df[runs_df["run_id"] == selected_run].iloc[0]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Model", run_info["model_name"])
                    with col2:
                        st.metric("Task", run_info["task_type"])
                    with col3:
                        st.metric("Status", run_info["status"])
                    
                    st.code(f"Run ID: {selected_run}", language="text")
                    
                    if st.button("Use this model for predictions", type="primary"):
                        st.session_state["selected_mlflow_run"] = selected_run
                        st.success("Model selected! Go to predictions section below.")
    
    except Exception as e:
        st.error(f"Error loading MLflow runs: {e}")
    
    st.divider()
    
    # Registered models
    st.markdown("##### Registered Models")
    
    try:
        registered = model.get_registered_models()
        
        if not registered:
            st.info("No hay modelos registrados en el Model Registry.")
        else:
            reg_df = pd.DataFrame(registered)
            st.dataframe(
                reg_df[["name", "version", "stage", "creation_timestamp"]],
                use_container_width=True,
                hide_index=True,
                height=250
            )
    
    except Exception as e:
        st.error(f"Error loading registered models: {e}")


def show() -> None:
    """Render the model training and inference workspace."""
    
    st.header("Machine Learning Model Training")
    
    # Tabs for different sections
    tab_train, tab_mlflow, tab_predict = st.tabs([
        "Train Model",
        "MLflow Tracking",
        "Predictions"
    ])
    
    # ==================== TRAINING TAB ====================
    with tab_train:
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
        
        with st.spinner("Cargando dataset..."):
            df_original, _ = fetch_table_csv("gold", selected_table)
        
        if df_original is None:
            st.error("No se pudo cargar el dataset")
            return
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Filas", f"{len(df_original):,}")
        with col2:
            st.metric("Columnas", len(df_original.columns))
        
        with st.expander("Vista previa del dataset original", expanded=False):
            st.dataframe(df_original.head(20), use_container_width=True, height=300)
        
        st.divider()
        
        # Encoding section
        df = render_encoding_section(df_original, selected_table)
        
        st.divider()
        
        # Model configuration
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
        
        # Feature selection and target
        st.subheader("Feature Selection & Target")
        
        col_target, col_features = st.columns([1, 2])
        
        with col_target:
            target_column = st.selectbox("Variable objetivo (Target):", df.columns.tolist(), key="target_select")
            
            if task_type == "classification" and df[target_column].dtype == "object":
                st.info("Target categórico detectado. Se aplicará Label Encoding automáticamente durante el entrenamiento.")
        
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
                    st.warning("Debes seleccionar al menos una feature")
                    return
        
        st.divider()
        
        # Hyperparameters and advanced options
        col_hyper, col_advanced = st.columns([1, 1])
        
        with col_hyper:
            params = render_hyperparameters(model_name, task_type)
        
        with col_advanced:
            st.markdown("##### Opciones Avanzadas")
            test_size = st.slider("Proporción de test:", 0.1, 0.5, 0.2, 0.05, key="test_size_slider")
            use_cv = st.checkbox("Usar Cross-Validation (5-fold)", value=True, key="use_cv_checkbox")
            run_name = st.text_input("Nombre del experimento (opcional):", key="run_name_input", 
                                     placeholder=f"{model_name}_{task_type}")
        
        st.divider()
        
        # Train button
        st.subheader("Model Training")
        
        if st.button("Train Model", type="primary", use_container_width=True, key="train_button"):
            
            if not selected_features and not use_all_features:
                st.error("Debes seleccionar features o marcar 'Usar todas las features'")
                return
            
            with st.spinner("Entrenando modelo... Esto puede tardar unos minutos."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("Preparando datos...")
                    progress_bar.progress(20)
                    
                    status_text.text(f"Entrenando {model_name}...")
                    progress_bar.progress(40)
                    
                    # Get encoders if they exist
                    feature_encoders = st.session_state.get(f"encoders_{selected_table}", None)
                    
                    metrics = model.train_model(
                        df=df,
                        target_column=target_column,
                        model_name=model_name,
                        task_type=task_type,
                        params=params,
                        selected_features=selected_features,
                        test_size=test_size,
                        use_cross_validation=use_cv,
                        run_name=run_name or None,
                        feature_encoders=feature_encoders,
                    )
                    
                    progress_bar.progress(80)
                    status_text.text("Calculando métricas...")
                    
                    progress_bar.progress(100)
                    status_text.empty()
                    progress_bar.empty()
                    
                    st.success("Modelo entrenado correctamente")
                    st.info(f"Run ID: `{metrics['mlflow_run_id']}`")
                    st.info(f"[Ver en MLflow]({MLFLOW_TRACKING_URI_LOCAL})")
                    
                    st.session_state["last_metrics"] = metrics
                    st.session_state["last_task_type"] = task_type
                    
                except Exception as e:
                    st.error(f"Error al entrenar: {str(e)}")
                    return
        
        # Display results
        if "last_metrics" in st.session_state:
            st.divider()
            st.subheader("Training Results")
            
            metrics = st.session_state["last_metrics"]
            task = st.session_state["last_task_type"]
            
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
            
            else:
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
            
            if use_cv and "cv_scores" in metrics:
                st.markdown("##### Cross-Validation Scores")
                cv_df = pd.DataFrame({
                    "Fold": range(1, len(metrics["cv_scores"]) + 1),
                    "Score": metrics["cv_scores"]
                })
                fig_cv = px.bar(cv_df, x="Fold", y="Score", title="Cross-Validation Scores by Fold")
                st.plotly_chart(fig_cv, use_container_width=True)
    
    # ==================== MLFLOW TAB ====================
    with tab_mlflow:
        render_mlflow_section()
    
    # ==================== PREDICTIONS TAB ====================
    with tab_predict:
        st.subheader("Make Predictions")
        
        # Select prediction source
        pred_source = st.radio(
            "Selecciona el origen del modelo:",
            ["Local Models (Pickle)", "MLflow Run"],
            horizontal=True
        )
        
        if pred_source == "Local Models (Pickle)":
            model_files = list(Path("models").glob("*.pkl")) if Path("models").exists() else []
            
            if not model_files:
                st.info("No hay modelos locales entrenados disponibles")
            else:
                selected_model_path = st.selectbox(
                    "Selecciona el modelo:",
                    [str(m) for m in model_files],
                    key="predict_model_select"
                )
                
                model_info = model.load_model_info(selected_model_path)
                
                with st.expander("Model Info"):
                    col_info1, col_info2, col_info3 = st.columns(3)
                    with col_info1:
                        st.metric("Model", model_info["model_name"])
                    with col_info2:
                        st.metric("Task", model_info["task_type"])
                    with col_info3:
                        st.metric("Target", model_info["target"])
                
                # Select dataset for predictions
                gold_tables, _ = fetch("gold", "tables")
                if gold_tables:
                    table_names = [t["name"] for t in gold_tables]
                    pred_table = st.selectbox("Dataset para predicciones:", table_names, key="pred_table_select")
                    
                    if pred_table and st.button("Generate Predictions", use_container_width=True, key="predict_button"):
                        with st.spinner("Generando predicciones..."):
                            try:
                                df_pred, _ = fetch_table_csv("gold", pred_table)
                                predictions = model.predict(df_pred, selected_model_path)
                                
                                st.success(f"{len(predictions)} predicciones generadas")
                                
                                st.dataframe(predictions.head(50), use_container_width=True, height=400)
                                
                                csv = predictions.to_csv(index=False)
                                st.download_button(
                                    "Download Predictions (CSV)",
                                    csv,
                                    "predictions.csv",
                                    "text/csv",
                                    use_container_width=True,
                                )
                            except Exception as e:
                                st.error(f"Error al predecir: {str(e)}")
        
        else:  # MLflow Run
            if "selected_mlflow_run" in st.session_state:
                selected_run_id = st.session_state["selected_mlflow_run"]
                st.info(f"Modelo seleccionado: Run ID `{selected_run_id[:8]}...`")
            else:
                st.warning("Selecciona un run desde la pestaña 'MLflow Tracking'")
                return
            
            # Select dataset for predictions
            gold_tables, _ = fetch("gold", "tables")
            if gold_tables:
                table_names = [t["name"] for t in gold_tables]
                pred_table = st.selectbox("Dataset para predicciones:", table_names, key="pred_table_mlflow_select")
                
                if pred_table and st.button("Generate Predictions from MLflow", use_container_width=True, key="predict_mlflow_button"):
                    with st.spinner("Generando predicciones desde MLflow..."):
                        try:
                            df_pred, _ = fetch_table_csv("gold", pred_table)
                            
                            # Get features from the run
                            import mlflow
                            client = mlflow.tracking.MlflowClient()
                            run = client.get_run(selected_run_id)
                            features_str = run.data.params.get("features", "")
                            features = features_str.split(",") if features_str else df_pred.columns.tolist()
                            
                            # Select only the features used in training
                            df_for_pred = df_pred[features]
                            
                            predictions = model.predict_from_mlflow(df_for_pred, selected_run_id)
                            
                            st.success(f"{len(predictions)} predicciones generadas")
                            
                            st.dataframe(predictions.head(50), use_container_width=True, height=400)
                            
                            csv = predictions.to_csv(index=False)
                            st.download_button(
                                "Download Predictions (CSV)",
                                csv,
                                "predictions_mlflow.csv",
                                "text/csv",
                                use_container_width=True,
                            )
                        except Exception as e:
                            st.error(f"Error al predecir: {str(e)}")


if __name__ == "__main__":
    show()