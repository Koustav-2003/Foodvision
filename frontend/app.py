
import streamlit as st
import requests
import textwrap
from html import escape


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="FoodVision",
    page_icon="🍕",
    layout="wide"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* Main page */
.stApp {
    background-color: #F7F4EA;
}

[data-testid="stHeader"] {
    background-color: #F7F4EA;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-left: 3rem;
    padding-right: 3rem;
    padding-bottom: 1rem;
}

/* General text */
.stApp, .stApp p, .stApp label {
    color: #263B32;
}

/* Hero section */
.hero {
    background: linear-gradient(120deg, #173D32, #285B45);
    padding: 30px 38px;
    border-radius: 22px;
    margin-bottom: 25px;
}

.hero-tag {
    color: #F3B36B;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.hero h1 {
    color: white !important;
    font-size: 38px;
    font-weight: 800;
    line-height: 1.15;
    margin: 0;
    padding: 0;
}

.hero p {
    color: #E0E9E1 !important;
    font-size: 15px;
    margin-top: 10px;
    line-height: 1.6;
    max-width: 750px;
}

.hero-highlight {
    color: #F3B36B;
}

/* File uploader */
[data-testid="stFileUploader"] section {
    background-color: #FFFFFF;
    border: 2px dashed #A8B9A8;
    border-radius: 16px;
    padding: 18px;
}

[data-testid="stFileUploader"] button {
    background-color: #173D32 !important;
    border: none !important;
    border-radius: 9px;
    color: #FFFFFF !important;
}

[data-testid="stFileUploader"] button span,
[data-testid="stFileUploader"] button p {
    color: #FFFFFF !important;
}

[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] * {
    color: #52665C !important;
}

[data-testid="stFileUploader"] button:hover {
    background-color: #285B45 !important;
}

/* Buttons */
.stButton > button {
    background-color: #173D32 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px;
    padding: 10px 24px;
    font-weight: 600;
}

.stButton > button p,
.stButton > button span {
    color: #FFFFFF !important;
}

.stButton > button:hover {
    background-color: #285B45 !important;
}

/* Prediction card */
.prediction-card {
    background: linear-gradient(120deg, #173D32, #285B45);
    padding: 22px;
    border-radius: 16px;
    margin-top: 12px;
    margin-bottom: 15px;
}

.prediction-label {
    color: #F3B36B !important;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
}

.prediction-name {
    color: #FFFFFF !important;
    font-size: 25px;
    font-weight: 800;
    margin-top: 10px;
    overflow-wrap: anywhere;
}

/* Confidence metric */
[data-testid="stMetricLabel"] {
    color: #52665C !important;
}

[data-testid="stMetricValue"] {
    color: #173D32 !important;
}

/* Top 5 predictions */
.top5-container {
    background-color: #FFFFFF;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #E1E8DE;
    margin-top: 18px;
}

.top5-title {
    color: #173D32 !important;
    font-size: 19px;
    font-weight: 750;
    margin-bottom: 20px;
}

.prediction-row {
    margin-bottom: 15px;
}

.prediction-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
}

.food-name {
    color: #263B32 !important;
    font-weight: 600;
    font-size: 13px;
}

.food-confidence {
    color: #52665C !important;
    font-size: 12px;
    font-weight: 600;
    white-space: nowrap;
}

.bar-background {
    background-color: #DCE8D9;
    height: 8px;
    border-radius: 20px;
    overflow: hidden;
}

.bar-fill {
    background: linear-gradient(90deg, #285B45, #E88942);
    height: 100%;
    border-radius: 20px;
}

/* Information card */
.info-card {
    background-color: #FFFFFF;
    border: 1px solid #E1E8DE;
    border-radius: 16px;
    padding: 20px;
    margin-top: 15px;
}

.info-card h4 {
    color: #173D32 !important;
    margin-top: 0;
}

.info-card p {
    color: #52665C !important;
    font-size: 14px;
    line-height: 1.6;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# HERO SECTION
# =====================================================

st.markdown("""
<div class="hero">
    <div class="hero-tag">AI Powered Food Recognition</div>
    <h1>Discover What's<br>
    <span class="hero-highlight">On Your Plate.</span></h1>
    <p>
        Upload a food image and let our deep learning model
        identify what you're eating. Powered by a fine-tuned
        ConvNeXt Small model trained on the Food-101 dataset.
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# BACKEND CONNECTION
# =====================================================

API_URL = "http://127.0.0.1:8000/predict"


# =====================================================
# SESSION STATE
# =====================================================

if "predictions" not in st.session_state:
    st.session_state.predictions = None

if "uploaded_file_id" not in st.session_state:
    st.session_state.uploaded_file_id = None


# =====================================================
# IMAGE UPLOADER
# =====================================================

st.subheader("Identify Your Food")

uploaded_file = st.file_uploader(
    "Choose a food image",
    type=["jpeg", "jpg", "png", "webp"]
)


# Reset predictions when a new image is uploaded

if uploaded_file is not None:

    current_file_id = (
        uploaded_file.name,
        uploaded_file.size
    )

    if current_file_id != st.session_state.uploaded_file_id:

        st.session_state.predictions = None
        st.session_state.uploaded_file_id = current_file_id

else:

    st.session_state.predictions = None
    st.session_state.uploaded_file_id = None


# =====================================================
# TWO-COLUMN LAYOUT
# =====================================================

if uploaded_file is not None:

    image_col, result_col = st.columns(
        [0.9, 1.4],
        gap="large"
    )

    # -------------------------------------------------
    # LEFT COLUMN: IMAGE PREVIEW
    # -------------------------------------------------

    with image_col:

        st.markdown("### Your Image")

        st.image(
            uploaded_file,
            caption="Uploaded Image",
            width=350
        )

        predict_button = st.button(
            "🍽️ Predict Food",
            use_container_width=True
        )

    # -------------------------------------------------
    # RUN PREDICTION
    # -------------------------------------------------

    if predict_button:

        try:

            with st.spinner("Classifying your image..."):

                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )
                    },
                    timeout=120
                )

                response.raise_for_status()

                result = response.json()
                predictions = result["predictions"]

                if not predictions:
                    raise ValueError("No predictions returned by the model.")

                st.session_state.predictions = predictions

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to the backend. "
                "Please make sure FastAPI is running."
            )

        except requests.exceptions.Timeout:

            st.error("The prediction request timed out. Please try again.")

        except requests.exceptions.RequestException as e:

            st.error(f"API request failed: {e}")

        except (KeyError, ValueError, IndexError) as e:

            st.error(f"Invalid API response: {e}")

    # -------------------------------------------------
    # RIGHT COLUMN: PREDICTION RESULTS
    # -------------------------------------------------

    with result_col:

        st.markdown("### Food Recognition")

        if st.session_state.predictions is not None:

            predictions = st.session_state.predictions

            # Primary prediction
            predicted_food = predictions[0]["class"].replace(
                "_", " "
            ).title()

            confidence = float(predictions[0]["confidence"])

            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-label">
                    YOUR FOOD IDENTIFIED
                </div>
                <div class="prediction-name">
                    {escape(predicted_food)}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence
            st.metric(
                label="Model Confidence",
                value=f"{confidence:.2f}%"
            )

            # -----------------------------------------
            # TOP 5 PREDICTIONS
            # -----------------------------------------

            top5_rows = []

            for i, prediction in enumerate(predictions[:5]):

                food_name = prediction["class"].replace(
                    "_", " "
                ).title()

                confidence = float(prediction["confidence"])

                bar_width = max(0, min(confidence, 100))

                row_html = textwrap.dedent(f"""\
                    <div class="prediction-row">
                        <div class="prediction-info">
                            <span class="food-name">
                                {i + 1}. {escape(food_name)}
                            </span>
                            <span class="food-confidence">
                                {confidence:.2f}%
                            </span>
                        </div>
                        <div class="bar-background">
                            <div class="bar-fill"
                                 style="width: {bar_width}%">
                            </div>
                        </div>
                    </div>
                """)

                top5_rows.append(row_html)

            # Construct complete HTML without indentation issues
            top5_html = (
                '<div class="top5-container">'
                '<div class="top5-title">Top 5 Predictions</div>'
                + "".join(top5_rows)
                + '</div>'
            )

            st.markdown(
                top5_html,
                unsafe_allow_html=True
            )

        else:

            st.markdown("""
            <div class="info-card">
                <h4>Ready to identify?</h4>
                <p>
                    Click the Predict Food button after uploading
                    your image. Our fine-tuned ConvNeXt Small model
                    will analyse it and return its top five
                    predicted food categories.
                </p>
            </div>
            """, unsafe_allow_html=True)


else:

    # =================================================
    # INITIAL INFORMATION
    # =================================================

    upload_col, info_col = st.columns(
        [1.15, 1],
        gap="large"
    )

    with upload_col:

        st.markdown("### Upload your image")

        st.caption(
            "Supported formats: JPG, JPEG, PNG and WEBP"
        )

    with info_col:

        st.markdown("### What happens next?")

        st.write(
            "Your image is processed by our fine-tuned "
            "ConvNeXt Small model, which returns its top "
            "five predicted food categories."
        )
