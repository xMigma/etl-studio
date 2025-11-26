"""Streamlit page for supervised ML workflows."""

from __future__ import annotations

import streamlit as st

from etl_studio.ai import model


def show() -> None:
    """Render the model training and inference workspace."""

    st.header("ML Model")
    st.write(
        "Placeholder for dataset selection, hyperparameter configuration, and metrics."
    )
    st.info("TODO: call model.train_model / model.predict and visualize outputs.")

    st.caption(
        f"Training helper: {model.train_model.__name__} · Predict helper: {model.predict.__name__}"
    )


if __name__ == "__main__":
    show()
