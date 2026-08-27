
import gradio as gr
import pandas as pd
import requests

# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000/predict"

PC_COLUMNS = [f"PC_{i}" for i in range(1, 19)]


# ============================================================
# PREDICTION FUNCTION
# EXISTING WORKING LOGIC PRESERVED
# ============================================================

def predict_from_csv(file):

    if file is None:
        return """
        ## ⚠️ No File Uploaded

        Please upload a CSV file containing
        **PC_1 to PC_18**.
        """

    try:

        # ----------------------------------------------------
        # Read uploaded CSV
        # ----------------------------------------------------

        df = pd.read_csv(file)

        # ----------------------------------------------------
        # Check required PCA columns
        # ----------------------------------------------------

        missing = [
            c for c in PC_COLUMNS
            if c not in df.columns
        ]

        if missing:
            return (
                "## ❌ INVALID CSV\n\n"
                "The following required PCA features are missing:\n\n"
                + ", ".join(missing)
            )

        # ----------------------------------------------------
        # Check whether CSV contains records
        # ----------------------------------------------------

        if len(df) == 0:
            return """
            ## ❌ EMPTY CSV

            The uploaded CSV does not contain any records.
            """

        # ----------------------------------------------------
        # Use first record
        # ----------------------------------------------------

        row = df.iloc[0]

        # ----------------------------------------------------
        # FastAPI expects PC_1 ... PC_18 individually
        # ----------------------------------------------------

        payload = {
            c: float(row[c])
            for c in PC_COLUMNS
        }

        # ----------------------------------------------------
        # Send prediction request to FastAPI
        # ----------------------------------------------------

        response = requests.post(
            API_URL,
            json=payload,
            timeout=30
        )

        # ----------------------------------------------------
        # API error handling
        # ----------------------------------------------------

        if response.status_code != 200:

            return (
                "## ❌ API ERROR\n\n"
                f"**HTTP Status:** {response.status_code}\n\n"
                + response.text
            )

        # ----------------------------------------------------
        # Read API response
        # ----------------------------------------------------

        result = response.json()

        prediction = result.get("prediction")
        label = result.get("label")
        probability = result.get(
            "attacker_probability"
        )

        # ----------------------------------------------------
        # Format probability
        # ----------------------------------------------------

        if probability is not None:

            probability = float(probability)

            percentage = probability * 100

            if prediction == 1:

                status = "🔴 ATTACKER DETECTED"

            else:

                status = "🟢 NORMAL BEHAVIOUR"

            return f"""
# {status}

## Behaviour Classification

**{label}**

## Attacker Probability

**{percentage:.2f}%**

## Model Prediction

**{prediction}**

---

### Analysis Completed Successfully

The uploaded record was processed using the
**18 Phase-2 PCA features** and classified by the
deployed **Extra Trees model**.

**Pipeline:** CSV → Gradio → FastAPI → Model
"""

        # ----------------------------------------------------
        # Fallback if probability is unavailable
        # ----------------------------------------------------

        return f"""
# 🟢 PREDICTION COMPLETE

## Behaviour

**{label}**

## Prediction

**{prediction}**
"""

    except Exception as e:

        return (
            "## ❌ APPLICATION ERROR\n\n"
            f"**{type(e).__name__}:** {e}"
        )


# ============================================================
# PROFESSIONAL CUSTOM CSS
# VISUAL DESIGN ONLY
# ============================================================

CSS = """

/* ==========================================================
   MAIN APPLICATION
   ========================================================== */

.gradio-container {
    max-width: 1150px !important;
    margin: auto !important;
}


/* ==========================================================
   HERO HEADER
   ========================================================== */

.hero {
    padding: 34px;
    border-radius: 18px;
    margin-bottom: 24px;

    background: linear-gradient(
        135deg,
        #111827,
        #1f2937
    );

    border: 1px solid #374151;

    text-align: center;
}

.hero-title {
    font-size: 34px;
    font-weight: 800;
    letter-spacing: 1px;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 16px;
    opacity: 0.85;
}


/* ==========================================================
   CARDS
   ========================================================== */

.card {
    border-radius: 16px;
    padding: 22px;

    border: 1px solid #374151;

    background: rgba(
        31,
        41,
        55,
        0.55
    );
}


/* ==========================================================
   SECTION TITLES
   ========================================================== */

.section-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 12px;
}


/* ==========================================================
   UPLOAD BOX
   ========================================================== */

.upload-box {
    border-radius: 15px !important;
}


/* ==========================================================
   PREDICT BUTTON
   ========================================================== */

.predict-button {
    font-size: 18px !important;
    font-weight: 700 !important;

    min-height: 54px !important;

    border-radius: 12px !important;
}


/* ==========================================================
   RESULT BOX
   ========================================================== */

.result-box {
    border-radius: 16px !important;

    min-height: 250px;
}


/* ==========================================================
   INFORMATION CARDS
   ========================================================== */

.info-card {
    text-align: center;

    padding: 18px;

    border-radius: 14px;

    border: 1px solid #374151;
}


/* ==========================================================
   FOOTER
   ========================================================== */

.footer {
    text-align: center;

    margin-top: 25px;

    padding: 18px;

    opacity: 0.7;

    font-size: 13px;
}

"""


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(
    title="V2X Driver Behaviour Detection",
    css=CSS,
    theme=gr.themes.Soft()
) as demo:

    # ========================================================
    # HEADER
    # ========================================================

    gr.HTML(
        """
        <div class="hero">

            <div class="hero-title">
                🚗 V2X DRIVER BEHAVIOUR DETECTION
            </div>

            <div class="hero-subtitle">
                AI-Powered Detection of Attacker and Normal
                Driving Behaviour
            </div>

        </div>
        """
    )


    # ========================================================
    # INTRODUCTION
    # ========================================================

    gr.Markdown(
        """
        ## 🛡️ Intelligent V2X Behaviour Analysis

        Upload a CSV containing the **18 Phase 2 PCA features**
        (`PC_1` to `PC_18`) and the deployed machine-learning
        model will classify the driving behaviour.

        **No manual entry of the 18 PCA values is required.**
        """
    )


    # ========================================================
    # MAIN INPUT / OUTPUT AREA
    # ========================================================

    with gr.Row():

        # ----------------------------------------------------
        # LEFT — INPUT
        # ----------------------------------------------------

        with gr.Column(
            scale=1,
            elem_classes="card"
        ):

            gr.HTML(
                """
                <div class="section-title">
                    📂 Input Data
                </div>
                """
            )

            gr.Markdown(
                """
                Upload your Phase 2 PCA feature CSV.

                The CSV must contain:

                `PC_1, PC_2, ..., PC_18`
                """
            )

            file_input = gr.File(
                label="Upload PCA Feature CSV",
                file_types=[".csv"],
                type="filepath",
                elem_classes="upload-box"
            )

            predict_button = gr.Button(
                "🚀 RUN BEHAVIOUR DETECTION",
                variant="primary",
                elem_classes="predict-button"
            )


        # ----------------------------------------------------
        # RIGHT — RESULT
        # ----------------------------------------------------

        with gr.Column(
            scale=1,
            elem_classes="card"
        ):

            gr.HTML(
                """
                <div class="section-title">
                    📊 Detection Result
                </div>
                """
            )

            output = gr.Markdown(
                """
                ### Waiting for input

                Upload a CSV and click
                **RUN BEHAVIOUR DETECTION**.
                """,
                elem_classes="result-box"
            )


    # ========================================================
    # DEPLOYMENT INFORMATION
    # ========================================================

    gr.Markdown(
        "## ⚙️ Deployment Information"
    )

    with gr.Row():

        with gr.Column(
            elem_classes="info-card"
        ):

            gr.Markdown(
                """
                ### 🧠 Model

                **Extra Trees**

                Lightweight deployment model
                """)


        with gr.Column(
            elem_classes="info-card"
        ):

            gr.Markdown(
                """
                ### 📐 Features

                **18 PCA Components**

                PC_1 → PC_18
                """)


        with gr.Column(
            elem_classes="info-card"
        ):

            gr.Markdown(
                """
                ### ⚡ Backend

                **FastAPI**

                Real-time prediction API
                """)


    # ========================================================
    # HOW IT WORKS
    # ========================================================

    gr.Markdown(
        """
        ## 🔍 How the System Works

        **1. Upload CSV** → **2. Extract PC_1–PC_18** →
        **3. FastAPI prediction** → **4. Extra Trees classification**
        → **5. Normal / Attacker result**
        """
    )


    # ========================================================
    # BUTTON CONNECTION
    # ========================================================

    predict_button.click(
        fn=predict_from_csv,
        inputs=file_input,
        outputs=output
    )


    # ========================================================
    # FOOTER
    # ========================================================

    gr.HTML(
        """
        <div class="footer">

            V2X Driver Behaviour Detection |
            Phase 5 Application Deployment

            <br>

            AI-based driving behaviour classification

        </div>
        """
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":

    import os

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000)),
        share=False
    )
