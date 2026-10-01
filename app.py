import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import date


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="FutureTwin AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(75, 70, 180, 0.16), transparent 28%),
        radial-gradient(circle at 90% 10%, rgba(0, 190, 255, 0.10), transparent 25%),
        #070a12;
    color: #f4f7ff;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}


/* =========================================================
   HEADINGS
========================================================= */

.hero-title {
    font-size: 58px;
    font-weight: 800;
    line-height: 1.05;
    margin-bottom: 15px;
    background: linear-gradient(90deg, #ffffff, #9ea9ff, #6de7ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 19px;
    color: #aeb7cb;
    max-width: 760px;
    line-height: 1.7;
}

.section-title {
    font-size: 30px;
    font-weight: 750;
    margin-top: 25px;
    margin-bottom: 8px;
}

.section-subtitle {
    color: #8f99af;
    margin-bottom: 25px;
}


/* =========================================================
   CARDS
========================================================= */

.card {
    background: rgba(18, 23, 38, 0.88);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 25px;
    height: 100%;
    box-shadow: 0 12px 35px rgba(0,0,0,0.22);
}

.card h3 {
    margin-top: 0;
    font-size: 20px;
}

.card p {
    color: #9ba5bb;
    line-height: 1.6;
}


/* =========================================================
   METRICS
========================================================= */

.metric-card {
    background: linear-gradient(
        145deg,
        rgba(21,27,44,0.96),
        rgba(12,16,28,0.96)
    );

    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 23px;
    text-align: center;
}

.metric-value {
    font-size: 34px;
    font-weight: 800;
    margin: 8px 0;
}

.metric-label {
    color: #8f99af;
    font-size: 14px;
}


/* =========================================================
   PREDICTION
========================================================= */

.prediction-box {
    background: linear-gradient(
        135deg,
        rgba(72, 62, 170, 0.20),
        rgba(0, 178, 230, 0.10)
    );

    border: 1px solid rgba(124, 133, 255, 0.25);
    border-radius: 22px;
    padding: 30px;
    margin-top: 20px;
}

.prediction-title {
    font-size: 25px;
    font-weight: 750;
}

.prediction-text {
    color: #c0c8da;
    line-height: 1.7;
    font-size: 16px;
}


/* =========================================================
   BUTTONS
========================================================= */

.stButton > button {
    border-radius: 12px;
    border: 1px solid rgba(120,130,255,0.35);
    background: linear-gradient(90deg, #5148c9, #287fc9);
    color: white;
    font-weight: 700;
    min-height: 45px;
}

.stButton > button:hover {
    border-color: #7e8cff;
    box-shadow: 0 0 25px rgba(87, 91, 220, 0.25);
}


/* =========================================================
   TABS
========================================================= */

button[data-baseweb="tab"] {
    font-weight: 600;
    color: #9da7bc;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: white;
}


/* =========================================================
   INPUTS
========================================================= */

div[data-baseweb="input"] {
    background-color: #111624;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL FILES
# =========================================================

REVENUE_MODEL_FILE = "futuretwin_revenue_model.pkl"
DEMAND_MODEL_FILE = "futuretwin_demand_model.pkl"
NLP_MODEL_FILE = "futuretwin_nlp_model.pkl"
TFIDF_FILE = "futuretwin_tfidf.pkl"


required_files = [
    REVENUE_MODEL_FILE,
    DEMAND_MODEL_FILE,
    NLP_MODEL_FILE,
    TFIDF_FILE
]


missing_files = [
    file for file in required_files
    if not os.path.exists(file)
]


if missing_files:

    st.error(
        "Some required model files are missing."
    )

    st.write("Missing files:")

    for file in missing_files:
        st.write(f"- `{file}`")

    st.info(
        "Keep app.py and the four PKL files in the same folder."
    )

    st.stop()


# =========================================================
# LOAD MODELS
# =========================================================

revenue_model = joblib.load(REVENUE_MODEL_FILE)
demand_model = joblib.load(DEMAND_MODEL_FILE)
nlp_model = joblib.load(NLP_MODEL_FILE)
tfidf = joblib.load(TFIDF_FILE)


# =========================================================
# SESSION STATE
# =========================================================

if "company" not in st.session_state:
    st.session_state.company = "My Company"

if "history" not in st.session_state:
    st.session_state.history = []

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "input_data" not in st.session_state:
    st.session_state.input_data = None

if "product_description" not in st.session_state:
    st.session_state.product_description = ""


# =========================================================
# MODEL FEATURE PREPARATION
# =========================================================

def create_regression_features(
    quantity,
    price,
    selected_date
):
    """
    Create the 9 features expected by the trained
    FutureTwin revenue and demand models.

    The original models were trained on daily-level
    Online Retail II features.
    """

    quantity = float(quantity)
    price = float(price)

    previous_revenue = quantity * price

    previous_quantity = quantity

    rolling_7_revenue = previous_revenue

    rolling_7_quantity = previous_quantity

    average_price = price

    # One user scenario represents one transaction/day input.
    transactions = 1

    returns = 0

    month = selected_date.month

    day_of_week = selected_date.weekday()

    features = pd.DataFrame([{
        "PreviousRevenue": previous_revenue,
        "PreviousQuantity": previous_quantity,
        "Rolling7Revenue": rolling_7_revenue,
        "Rolling7Quantity": rolling_7_quantity,
        "AveragePrice": average_price,
        "Transactions": transactions,
        "Returns": returns,
        "Month": month,
        "DayOfWeek": day_of_week
    }])

    return features


# =========================================================
# REVENUE PREDICTION
# =========================================================

def predict_revenue(
    quantity,
    price,
    selected_date
):

    features = create_regression_features(
        quantity,
        price,
        selected_date
    )

    prediction = revenue_model.predict(features)[0]

    return max(float(prediction), 0)


# =========================================================
# DEMAND PREDICTION
# =========================================================

def predict_demand(
    quantity,
    price,
    selected_date
):

    features = create_regression_features(
        quantity,
        price,
        selected_date
    )

    prediction = demand_model.predict(features)[0]

    probability = None

    if hasattr(demand_model, "predict_proba"):

        probabilities = demand_model.predict_proba(features)[0]

        classes = list(demand_model.classes_)

        if 1 in classes:

            index = classes.index(1)

            probability = float(
                probabilities[index] * 100
            )

        else:

            probability = float(
                max(probabilities) * 100
            )

    if int(prediction) == 1:

        demand_status = "High Demand"

    else:

        demand_status = "Normal Demand"

    return demand_status, probability


# =========================================================
# NLP PRODUCT ANALYSIS
# =========================================================

def analyze_product_text(description):

    text = str(description).strip()

    if not text:

        return "No Product Description", 0.0

    vectorized_text = tfidf.transform([text])

    prediction = nlp_model.predict(
        vectorized_text
    )[0]

    probability = None

    if hasattr(nlp_model, "predict_proba"):

        probabilities = nlp_model.predict_proba(
            vectorized_text
        )[0]

        classes = list(nlp_model.classes_)

        if 1 in classes:

            index = classes.index(1)

            probability = float(
                probabilities[index] * 100
            )

    if int(prediction) == 1:

        status = "Return/Cancellation Risk"

    else:

        status = "Normal Product Pattern"

    return status, probability


# =========================================================
# FULL ANALYSIS
# =========================================================

def analyze_scenario(
    quantity,
    price,
    selected_date,
    description
):

    revenue = predict_revenue(
        quantity,
        price,
        selected_date
    )

    demand_status, demand_probability = predict_demand(
        quantity,
        price,
        selected_date
    )

    text_status, text_probability = analyze_product_text(
        description
    )

    return {
        "revenue": revenue,
        "demand_status": demand_status,
        "demand_probability": demand_probability,
        "text_status": text_status,
        "text_probability": text_probability
    }


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div style="text-align:center; padding:20px 0 10px 0;">

<div style="
display:inline-block;
padding:7px 15px;
border-radius:30px;
background:rgba(90,100,220,0.12);
border:1px solid rgba(120,130,255,0.25);
color:#aeb8ff;
font-size:13px;
font-weight:600;
">
AI-POWERED DIGITAL TWIN
</div>

<div class="hero-title">
FutureTwin AI
</div>

<div class="hero-subtitle" style="margin:auto;">
Test a possible future business scenario before making the real-world decision.
Create your digital twin, change conditions, and explore how revenue and demand
may change.
</div>

</div>
""", unsafe_allow_html=True)

st.write("")


# =========================================================
# MAIN TABS
# =========================================================

tabs = st.tabs([
    "🏠 Overview",
    "🏢 Company Setup",
    "🧬 Digital Twin",
    "🔮 Future Simulator",
    "⚠️ AI Insights",
    "📊 Compare Futures",
    "🕘 History"
])


# =========================================================
# TAB 1 — OVERVIEW
# =========================================================

with tabs[0]:

    st.markdown(
        '<div class="section-title">Your future starts with a scenario</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'FutureTwin converts your current business conditions into '
        'AI-based future scenario insights.'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("""
        <div class="card">

        <h3>🧬 Digital Twin</h3>

        <p>
        Create a simplified digital representation of your current
        product and business scenario.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class="card">

        <h3>🔮 What-If Simulation</h3>

        <p>
        Change demand, price and other scenario conditions to explore
        possible future outcomes.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown("""
        <div class="card">

        <h3>📊 AI Intelligence</h3>

        <p>
        Combine revenue prediction, demand classification and product
        text analysis into one decision-support view.
        </p>

        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
    <div class="prediction-box">

    <div class="prediction-title">
    How FutureTwin works
    </div>

    <div class="prediction-text">
    Current business data → Digital Twin → AI models →
    What-if simulation → Future comparison → Decision insights
    </div>

    </div>
    """, unsafe_allow_html=True)

    st.info(
        "FutureTwin provides model-based scenario insights. "
        "It does not guarantee future outcomes."
    )


# =========================================================
# TAB 2 — COMPANY SETUP
# =========================================================

with tabs[1]:

    st.markdown(
        '<div class="section-title">Create your Digital Twin</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Enter only the information required to simulate your product scenario.'
        '</div>',
        unsafe_allow_html=True
    )

    company = st.text_input(
        "Company / Project Name",
        value=st.session_state.company
    )

    st.session_state.company = company

    st.write("")

    c1, c2 = st.columns(2)

    with c1:

        product_description = st.text_input(
            "Product Description",
            value=st.session_state.product_description,
            placeholder="Example: wireless bluetooth headphones"
        )

        quantity = st.number_input(
            "Expected Quantity",
            min_value=1.0,
            value=100.0,
            step=1.0
        )

    with c2:

        price = st.number_input(
            "Unit Price",
            min_value=0.01,
            value=50.0,
            step=1.0
        )

        selected_date = st.date_input(
            "Scenario Date",
            value=date.today()
        )

    st.session_state.product_description = product_description

    st.write("")

    st.markdown("""
    <div class="card">

    <h3>What FutureTwin will calculate</h3>

    <p>
    You provide four simple inputs. FutureTwin internally creates the
    model features required by the trained machine learning models.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    if st.button(
        "🧬 Create My Digital Twin",
        use_container_width=True
    ):

        if not product_description.strip():

            st.warning(
                "Please enter a product description."
            )

        else:

            result = analyze_scenario(
                quantity,
                price,
                selected_date,
                product_description
            )

            st.session_state.input_data = {
                "product_description": product_description,
                "quantity": quantity,
                "price": price,
                "date": selected_date
            }

            st.session_state.prediction = result

            st.session_state.history.append({
                "Scenario": "Current Business State",
                "Revenue": round(result["revenue"], 2),
                "Demand": result["demand_status"],
                "Text Analysis": result["text_status"]
            })

            st.success(
                "Digital Twin created successfully. Open the Digital Twin tab."
            )


# =========================================================
# TAB 3 — DIGITAL TWIN
# =========================================================

with tabs[2]:

    st.markdown(
        '<div class="section-title">Your Digital Twin</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'AI-generated representation of your current product scenario.'
        '</div>',
        unsafe_allow_html=True
    )

    if st.session_state.prediction is None:

        st.warning(
            "First create your Digital Twin in Company Setup."
        )

    else:

        prediction = st.session_state.prediction

        input_data = st.session_state.input_data

        c1, c2, c3 = st.columns(3)

        with c1:

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-label">
                PREDICTED FUTURE REVENUE
                </div>

                <div class="metric-value">
                ₹{prediction["revenue"]:,.0f}
                </div>

                <div class="metric-label">
                Model prediction
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:

            probability = prediction["demand_probability"]

            demand_value = (
                f"{probability:.1f}%"
                if probability is not None
                else prediction["demand_status"]
            )

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-label">
                DEMAND SIGNAL
                </div>

                <div class="metric-value">
                {demand_value}
                </div>

                <div class="metric-label">
                {prediction["demand_status"]}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:

            text_probability = prediction["text_probability"]

            text_value = (
                f"{text_probability:.1f}%"
                if text_probability is not None
                else "Analyzed"
            )

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-label">
                PRODUCT TEXT SIGNAL
                </div>

                <div class="metric-value">
                {text_value}
                </div>

                <div class="metric-label">
                {prediction["text_status"]}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        st.markdown(
            f"""
            <div class="prediction-box">

            <div class="prediction-title">
            🔮 FutureTwin Outlook
            </div>

            <div class="prediction-text">

            For <b>{input_data["product_description"]}</b>,
            with an expected quantity of <b>{input_data["quantity"]:,.0f}</b>
            units at <b>₹{input_data["price"]:,.2f}</b> per unit,
            the revenue model estimates approximately

            <b>₹{prediction["revenue"]:,.0f}</b>
            in future revenue.

            The demand model indicates
            <b>{prediction["demand_status"]}</b>.

            The product description analysis indicates
            <b>{prediction["text_status"]}</b>.

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        st.subheader("Current Digital Twin State")

        twin_df = pd.DataFrame({
            "Parameter": [
                "Company",
                "Product",
                "Quantity",
                "Unit Price",
                "Date"
            ],

            "Value": [
                st.session_state.company,
                input_data["product_description"],
                input_data["quantity"],
                f"₹{input_data['price']:,.2f}",
                input_data["date"]
            ]
        })

        st.dataframe(
            twin_df,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# TAB 4 — FUTURE SIMULATOR
# =========================================================

with tabs[3]:

    st.markdown(
        '<div class="section-title">What happens if the future changes?</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Modify the expected demand or price and compare the simulated future.'
        '</div>',
        unsafe_allow_html=True
    )

    if st.session_state.prediction is None:

        st.warning(
            "Create your Digital Twin first."
        )

    else:

        input_data = st.session_state.input_data

        base_quantity = float(
            input_data["quantity"]
        )

        base_price = float(
            input_data["price"]
        )

        selected_date = input_data["date"]

        description = input_data["product_description"]

        c1, c2 = st.columns(2)

        with c1:

            demand_change = st.slider(
                "Future Demand Change (%)",
                -50,
                100,
                0,
                5
            )

        with c2:

            price_change = st.slider(
                "Future Price Change (%)",
                -30,
                50,
                0,
                5
            )

        future_quantity = (
            base_quantity *
            (1 + demand_change / 100)
        )

        future_price = (
            base_price *
            (1 + price_change / 100)
        )

        future_quantity = max(
            future_quantity,
            1
        )

        future_price = max(
            future_price,
            0.01
        )

        base_result = analyze_scenario(
            base_quantity,
            base_price,
            selected_date,
            description
        )

        future_result = analyze_scenario(
            future_quantity,
            future_price,
            selected_date,
            description
        )

        revenue_change = (
            future_result["revenue"] -
            base_result["revenue"]
        )

        revenue_change_percent = (
            revenue_change /
            base_result["revenue"] * 100
            if base_result["revenue"] != 0
            else 0
        )

        st.write("")

        c1, c2 = st.columns(2)

        with c1:

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-label">
                CURRENT REVENUE
                </div>

                <div class="metric-value">
                ₹{base_result["revenue"]:,.0f}
                </div>

                <div class="metric-label">
                Current scenario
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-label">
                SIMULATED FUTURE
                </div>

                <div class="metric-value">
                ₹{future_result["revenue"]:,.0f}
                </div>

                <div class="metric-label">
                Future scenario
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        if revenue_change > 0:

            st.success(
                f"Simulated revenue increases by "
                f"₹{revenue_change:,.0f} "
                f"({revenue_change_percent:.1f}%)."
            )

        elif revenue_change < 0:

            st.error(
                f"Simulated revenue decreases by "
                f"₹{abs(revenue_change):,.0f} "
                f"({abs(revenue_change_percent):.1f}%)."
            )

        else:

            st.info(
                "The simulated revenue remains approximately unchanged."
            )

        st.write("")

        comparison_df = pd.DataFrame({
            "State": [
                "Current",
                "Future"
            ],

            "Revenue": [
                base_result["revenue"],
                future_result["revenue"]
            ]
        })

        st.bar_chart(
            comparison_df.set_index("State")
        )

        st.markdown(
            f"""
            <div class="prediction-box">

            <div class="prediction-title">
            🔮 Future Scenario Result
            </div>

            <div class="prediction-text">

            Demand change:
            <b>{demand_change:+d}%</b>

            <br>

            Price change:
            <b>{price_change:+d}%</b>

            <br><br>

            Simulated quantity:
            <b>{future_quantity:,.0f}</b>

            <br>

            Simulated price:
            <b>₹{future_price:,.2f}</b>

            <br><br>

            Future demand signal:
            <b>{future_result["demand_status"]}</b>

            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        if st.button(
            "💾 Save This Future Scenario",
            use_container_width=True
        ):

            st.session_state.history.append({
                "Scenario": f"Demand {demand_change:+d}% / Price {price_change:+d}%",
                "Revenue": round(
                    future_result["revenue"],
                    2
                ),
                "Demand": future_result["demand_status"],
                "Text Analysis": future_result["text_status"]
            })

            st.success(
                "Future scenario saved."
            )


# =========================================================
# TAB 5 — AI INSIGHTS
# =========================================================

with tabs[4]:

    st.markdown(
        '<div class="section-title">FutureTwin AI Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Understand what each trained model is signalling.'
        '</div>',
        unsafe_allow_html=True
    )

    if st.session_state.prediction is None:

        st.warning(
            "Create your Digital Twin first."
        )

    else:

        prediction = st.session_state.prediction

        c1, c2 = st.columns(2)

        with c1:

            st.markdown("""
            <div class="card">

            <h3>📈 Revenue Prediction</h3>

            <p>
            The Random Forest Regressor uses nine engineered daily-level
            features to estimate future revenue.
            </p>

            </div>
            """, unsafe_allow_html=True)

            st.write("")

            st.metric(
                "Predicted Revenue",
                f"₹{prediction['revenue']:,.0f}"
            )

        with c2:

            st.markdown("""
            <div class="card">

            <h3>📊 Future Demand</h3>

            <p>
            The Random Forest Classifier estimates whether the scenario
            corresponds to a high-demand or normal-demand condition.
            </p>

            </div>
            """, unsafe_allow_html=True)

            st.write("")

            if prediction["demand_probability"] is not None:

                st.metric(
                    "High Demand Probability",
                    f"{prediction['demand_probability']:.1f}%"
                )

            else:

                st.metric(
                    "Demand Signal",
                    prediction["demand_status"]
                )

        st.write("")

        st.markdown("""
        <div class="card">

        <h3>📝 Product Text Analysis</h3>

        <p>
        The product description is converted into TF-IDF features and
        analyzed using Logistic Regression. The model identifies whether
        the product text resembles patterns associated with returns or
        cancellations in the training data.
        </p>

        </div>
        """, unsafe_allow_html=True)

        st.write("")

        st.metric(
            "Product Text Signal",
            prediction["text_status"]
        )

        if prediction["text_probability"] is not None:

            st.write(
                f"Model probability: "
                f"**{prediction['text_probability']:.1f}%**"
            )

        st.info(
            "These model outputs are learned from historical Online Retail II data "
            "and should be interpreted as predictive signals, not guaranteed outcomes."
        )


# =========================================================
# TAB 6 — COMPARE FUTURES
# =========================================================

with tabs[5]:

    st.markdown(
        '<div class="section-title">Compare Possible Futures</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Explore several predefined what-if scenarios using the trained models.'
        '</div>',
        unsafe_allow_html=True
    )

    if st.session_state.prediction is None:

        st.warning(
            "Create your Digital Twin first."
        )

    else:

        input_data = st.session_state.input_data

        base_quantity = float(
            input_data["quantity"]
        )

        base_price = float(
            input_data["price"]
        )

        selected_date = input_data["date"]

        description = input_data["product_description"]

        scenarios = {
            "Current": (0, 0),
            "Demand +10%": (10, 0),
            "Demand +20%": (20, 0),
            "Price +10%": (0, 10),
            "Demand +20% + Price +10%": (20, 10)
        }

        rows = []

        for name, changes in scenarios.items():

            demand_change = changes[0]
            price_change = changes[1]

            scenario_quantity = (
                base_quantity *
                (1 + demand_change / 100)
            )

            scenario_price = (
                base_price *
                (1 + price_change / 100)
            )

            result = analyze_scenario(
                scenario_quantity,
                scenario_price,
                selected_date,
                description
            )

            rows.append({
                "Future Scenario": name,
                "Predicted Revenue": round(
                    result["revenue"],
                    2
                ),
                "Demand Status": result["demand_status"]
            })

        comparison = pd.DataFrame(rows)

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

        st.write("")

        chart_df = comparison[
            [
                "Future Scenario",
                "Predicted Revenue"
            ]
        ].set_index(
            "Future Scenario"
        )

        st.bar_chart(
            chart_df
        )


# =========================================================
# TAB 7 — HISTORY
# =========================================================

with tabs[6]:

    st.markdown(
        '<div class="section-title">Scenario History</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Your simulated business futures from this session.'
        '</div>',
        unsafe_allow_html=True
    )

    if len(st.session_state.history) == 0:

        st.info(
            "No scenarios saved yet. Create your Digital Twin and run a simulation."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# FOOTER
# =========================================================

st.write("")

st.markdown("""
<hr style="
border:0;
border-top:1px solid rgba(255,255,255,0.08);
">

<div style="
text-align:center;
color:#69738a;
font-size:13px;
padding:15px;
">

FutureTwin AI • AI-powered future scenario simulation

<br><br>

Model-based simulation only. Results are not guaranteed future outcomes
and should be validated before real-world implementation.

</div>
""", unsafe_allow_html=True)