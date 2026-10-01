import streamlit as st
import requests
import pandas as pd


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)


# ---------------------------------------------------------
# Backend API URL
# ---------------------------------------------------------

import os

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def format_label(label: str) -> str:
    """
    Convert model labels into cleaner display labels.
    """

    label_map = {
        "pants-fire": "Pants on Fire",
        "false": "False",
        "barely-true": "Barely True",
        "half-true": "Half True",
        "mostly-true": "Mostly True",
        "true": "True"
    }

    return label_map.get(label, label.replace("-", " ").title())


def get_prediction(claim: str):
    """
    Send claim to FastAPI backend.
    """

    response = requests.post(
        f"{API_URL}/predict",
        json={
            "claim": claim
        },
        timeout=30
    )

    return response


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("📰 Fake News Detector")

st.write(
    """
    Analyze a political or factual claim using our
    NLP-based text classification model.
    """
)

st.info(
    """
    This V1 model analyzes linguistic patterns in the text.
    It does not independently verify the claim against
    external evidence or live sources.
    """
)


# ---------------------------------------------------------
# Claim Input
# ---------------------------------------------------------

st.subheader("Enter a Claim")

claim = st.text_area(
    label="Claim",
    placeholder=(
        "Example: The unemployment rate doubled last year."
    ),
    height=150,
    label_visibility="collapsed"
)


# ---------------------------------------------------------
# Analyze Button
# ---------------------------------------------------------

analyze_button = st.button(
    "Analyze Claim",
    type="primary",
    use_container_width=True
)


# ---------------------------------------------------------
# Prediction Logic
# ---------------------------------------------------------

if analyze_button:

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    if not claim.strip():

        st.warning("Please enter a claim before analyzing.")

    else:

        try:

            with st.spinner("Analyzing claim..."):

                response = get_prediction(claim.strip())


            # -------------------------------------------------
            # Successful response
            # -------------------------------------------------

            if response.status_code == 200:

                result = response.json()

                prediction = result["prediction"]

                confidence = result["confidence"]

                probabilities = result["probabilities"]

                prediction_id = result.get(
                    "prediction_id"
                )


                # -------------------------------------------------
                # Main Result
                # -------------------------------------------------

                st.divider()

                st.subheader("Analysis Result")


                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        label="Prediction",
                        value=format_label(prediction)
                    )

                with col2:

                    st.metric(
                        label="Model Confidence",
                        value=f"{confidence * 100:.2f}%"
                    )


                # -------------------------------------------------
                # Confidence Progress Bar
                # -------------------------------------------------

                st.write("### Confidence")

                st.progress(
                    min(
                        max(float(confidence), 0.0),
                        1.0
                    )
                )


                # -------------------------------------------------
                # Probability Distribution
                # -------------------------------------------------

                st.write("### Class Probabilities")


                probability_data = []

                for label, probability in probabilities.items():

                    probability_data.append(
                        {
                            "Label": format_label(label),
                            "Probability": probability
                        }
                    )


                probability_df = pd.DataFrame(
                    probability_data
                )


                probability_df = probability_df.sort_values(
                    by="Probability",
                    ascending=False
                )


                # -------------------------------------------------
                # Bar Chart
                # -------------------------------------------------

                chart_df = probability_df.set_index(
                    "Label"
                )

                st.bar_chart(
                    chart_df["Probability"]
                )


                # -------------------------------------------------
                # Detailed Probability Table
                # -------------------------------------------------

                display_df = probability_df.copy()

                display_df["Probability"] = (
                    display_df["Probability"] * 100
                ).round(2)

                display_df["Probability"] = (
                    display_df["Probability"].astype(str)
                    + "%"
                )


                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )


                # -------------------------------------------------
                # Prediction Information
                # -------------------------------------------------

                with st.expander(
                    "View prediction details"
                ):

                    st.write(
                        "**Input claim:**"
                    )

                    st.write(claim)

                    st.write(
                        "**Predicted class:**",
                        format_label(prediction)
                    )

                    st.write(
                        "**Confidence:**",
                        f"{confidence * 100:.2f}%"
                    )

                    if prediction_id is not None:

                        st.write(
                            "**Prediction ID:**",
                            prediction_id
                        )


                # -------------------------------------------------
                # Important Disclaimer
                # -------------------------------------------------

                st.warning(
                    """
                    The prediction represents the model's
                    classification based on patterns learned
                    from the LIAR dataset.

                    It should not be interpreted as independent
                    factual verification of the claim.
                    """
                )


            # -------------------------------------------------
            # Backend returned error
            # -------------------------------------------------

            else:

                try:

                    error_data = response.json()

                    error_message = error_data.get(
                        "detail",
                        "Prediction failed."
                    )

                except Exception:

                    error_message = (
                        "The backend returned an unexpected error."
                    )


                st.error(
                    f"Backend Error: {error_message}"
                )


        # -----------------------------------------------------
        # Backend not running
        # -----------------------------------------------------

        except requests.exceptions.ConnectionError:

            st.error(
                """
                Unable to connect to the backend.

                Make sure FastAPI is running on:

                http://127.0.0.1:8000
                """
            )


        # -----------------------------------------------------
        # Backend timeout
        # -----------------------------------------------------

        except requests.exceptions.Timeout:

            st.error(
                """
                The backend took too long to respond.
                Please try again.
                """
            )


        # -----------------------------------------------------
        # Other errors
        # -----------------------------------------------------

        except Exception as e:

            st.error(
                f"Something went wrong: {e}"
            )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "Fake News Detection using NLP — V1"
)