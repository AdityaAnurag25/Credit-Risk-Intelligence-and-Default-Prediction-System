import json

import streamlit as st

from credit_risk.api.decision import assign_decision
from credit_risk.api.features import build_feature_row, load_category_vocab, load_feature_defaults
from credit_risk.api.risk_bands import assign_risk_band
from credit_risk.api.schemas import LoanApplication
from credit_risk.config import settings
from credit_risk.models.explain import local_explanation
from credit_risk.models.load import load_champion

TOP_REASONS_COUNT = 5

GRADE_OPTIONS = ["A", "B", "C", "D", "E", "F", "G"]
SUB_GRADE_OPTIONS = [f"{grade}{n}" for grade in GRADE_OPTIONS for n in range(1, 6)]
TERM_OPTIONS = ["36 months", "60 months"]
EMP_LENGTH_OPTIONS = ["< 1 year", *[f"{n} years" for n in range(1, 10)], "10+ years"]
HOME_OWNERSHIP_OPTIONS = ["RENT", "MORTGAGE", "OWN", "OTHER"]
VERIFICATION_OPTIONS = ["Verified", "Source Verified", "Not Verified"]
INITIAL_LIST_STATUS_OPTIONS = ["w", "f"]
APPLICATION_TYPE_OPTIONS = ["Individual", "Joint App"]


def _unwrap_tree_estimator(calibrated_model):
    """Get back the raw XGBoost estimator wrapped inside the champion model.

    Mirrors `credit_risk.api.main._unwrap_tree_estimator`: `run_training()` only
    ever registers a `CalibratedClassifierCV(FrozenEstimator(xgb))` as champion,
    so this structure is a safe assumption here too. Explaining the raw tree model
    gets the fast, exact `TreeExplainer` path instead of a slow model-agnostic one.
    """
    return calibrated_model.calibrated_classifiers_[0].estimator.estimator


@st.cache_resource
def _load_model():
    return load_champion()


@st.cache_data
def _load_metadata() -> dict:
    metadata_path = settings.models_dir / "champion_metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text())


@st.cache_data
def _load_vocab_and_defaults() -> tuple[dict[str, list[str]], dict[str, float]]:
    # Committed alongside champion.joblib in models/ (not the gitignored
    # outputs/models/ the local/Docker API workflow uses) so the encoding these
    # rely on survives a plain `git clone` with no training run or volume mount.
    vocab = load_category_vocab(settings.models_dir / "category_vocab.json")
    defaults = load_feature_defaults(settings.models_dir / "feature_defaults.json")
    return vocab, defaults


def _render_sidebar(metadata: dict) -> None:
    st.sidebar.header("Champion model")
    if not metadata:
        st.sidebar.warning("No models/champion_metadata.json found.")
        return

    st.sidebar.metric("Version", metadata.get("model_version", "unknown"))
    st.sidebar.metric("Test ROC-AUC", f"{metadata.get('test_auc', 0):.3f}")
    st.sidebar.metric("Test KS", f"{metadata.get('test_ks', 0):.3f}")
    st.sidebar.metric("Test Gini", f"{metadata.get('test_gini', 0):.3f}")
    st.sidebar.metric("Brier score", f"{metadata.get('brier', 0):.3f}")
    st.sidebar.caption(f"Trained: {metadata.get('training_date', 'unknown')}")

    thresholds = metadata.get("decision_thresholds", {})
    if thresholds:
        st.sidebar.caption(
            f"Approve below {thresholds.get('approve_below')}, "
            f"reject above {thresholds.get('reject_above')}"
        )


def _application_form() -> LoanApplication | None:
    with st.form("loan_application"):
        col1, col2 = st.columns(2)
        with col1:
            loan_amnt = st.number_input(
                "Loan amount ($)", min_value=1000.0, max_value=40000.0, value=15000.0, step=500.0
            )
            term = st.selectbox("Term", TERM_OPTIONS)
            int_rate = st.number_input(
                "Interest rate (%)", min_value=0.0, max_value=40.0, value=13.5
            )
            installment = st.number_input("Monthly installment ($)", min_value=1.0, value=450.0)
            grade = st.selectbox("Grade", GRADE_OPTIONS, index=1)
            sub_grade = st.selectbox(
                "Sub-grade", SUB_GRADE_OPTIONS, index=SUB_GRADE_OPTIONS.index("B3")
            )
            emp_length = st.selectbox("Employment length", EMP_LENGTH_OPTIONS, index=5)
            home_ownership = st.selectbox("Home ownership", HOME_OWNERSHIP_OPTIONS)
            annual_inc = st.number_input("Annual income ($)", min_value=1.0, value=65000.0)
            verification_status = st.selectbox("Verification status", VERIFICATION_OPTIONS)
            purpose = st.text_input("Purpose", value="debt_consolidation")
            dti = st.number_input("Debt-to-income ratio", min_value=0.0, value=18.5)
        with col2:
            delinq_2yrs = st.number_input("Delinquencies (2yr)", min_value=0, value=0, step=1)
            earliest_cr_line = st.text_input("Earliest credit line (Mon-YYYY)", value="Jan-2010")
            inq_last_6mths = st.number_input("Inquiries (6mo)", min_value=0, value=1, step=1)
            open_acc = st.number_input("Open credit lines", min_value=0, value=8, step=1)
            pub_rec = st.number_input("Derogatory public records", min_value=0, value=0, step=1)
            revol_bal = st.number_input("Revolving balance ($)", min_value=0.0, value=12000.0)
            revol_util = st.number_input(
                "Revolving utilization (%)", min_value=0.0, max_value=150.0, value=45.0
            )
            total_acc = st.number_input("Total credit lines", min_value=0, value=20, step=1)
            initial_list_status = st.selectbox("Initial list status", INITIAL_LIST_STATUS_OPTIONS)
            application_type = st.selectbox("Application type", APPLICATION_TYPE_OPTIONS)

        submitted = st.form_submit_button("Score application")

    if not submitted:
        return None

    return LoanApplication(
        loan_amnt=loan_amnt,
        term=term,
        int_rate=int_rate,
        installment=installment,
        grade=grade,
        sub_grade=sub_grade,
        emp_length=emp_length,
        home_ownership=home_ownership,
        annual_inc=annual_inc,
        verification_status=verification_status,
        purpose=purpose,
        dti=dti,
        delinq_2yrs=delinq_2yrs,
        earliest_cr_line=earliest_cr_line,
        inq_last_6mths=inq_last_6mths,
        open_acc=open_acc,
        pub_rec=pub_rec,
        revol_bal=revol_bal,
        revol_util=revol_util,
        total_acc=total_acc,
        initial_list_status=initial_list_status,
        application_type=application_type,
    )


def _render_result(model, application: LoanApplication) -> None:
    feature_names = list(model.feature_names_in_)
    category_vocab, feature_defaults = _load_vocab_and_defaults()

    row = build_feature_row(application, feature_names, category_vocab, feature_defaults)
    probability_default = float(model.predict_proba(row)[0, 1])
    risk_band = assign_risk_band(probability_default)
    decision = assign_decision(probability_default)

    st.subheader("Result")
    col1, col2, col3 = st.columns(3)
    col1.metric("Probability of default", f"{probability_default:.1%}")
    col2.metric("Risk band", risk_band)
    col3.metric("Decision", decision)

    explainer_model = _unwrap_tree_estimator(model)
    reasons = local_explanation(explainer_model, row, n=TOP_REASONS_COUNT)

    # Show the value the applicant actually submitted where available (e.g.
    # sub_grade="B3"), falling back to the model's encoded numeric value for
    # the ~90 columns the form doesn't cover — same rule as the API's /score.
    submitted_values = application.model_dump()
    reasons["value"] = [
        submitted_values.get(feature, value)
        for feature, value in zip(reasons["feature"], reasons["value"], strict=True)
    ]

    st.subheader("Top reasons")
    st.dataframe(
        reasons[["feature", "value", "direction", "magnitude"]],
        hide_index=True,
        width="stretch",
    )


def main() -> None:
    st.set_page_config(page_title="Credit Risk PD Scoring", page_icon="💳", layout="wide")
    st.title("Credit Risk Intelligence — PD Scoring Dashboard")
    st.caption(
        "Scores a single loan application against the committed champion model "
        "(models/champion.joblib). No MLflow or database connection required."
    )

    metadata = _load_metadata()
    _render_sidebar(metadata)

    champion_path = settings.models_dir / "champion.joblib"
    if not champion_path.exists():
        st.error(
            f"No champion model found at {champion_path}. Run `make train MODEL=xgboost` "
            "or `make export-champion` to produce one, then commit models/champion.joblib "
            "and models/champion_metadata.json."
        )
        return

    model = _load_model()
    application = _application_form()
    if application is not None:
        _render_result(model, application)


if __name__ == "__main__":
    main()
