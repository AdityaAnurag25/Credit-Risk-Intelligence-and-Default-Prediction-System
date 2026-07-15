import pytest
from fastapi.testclient import TestClient

from credit_risk.api.main import app

VALID_APPLICATION = {
    "loan_amnt": 15000,
    "term": "36 months",
    "int_rate": 13.5,
    "installment": 450.0,
    "grade": "B",
    "sub_grade": "B3",
    "emp_length": "5 years",
    "home_ownership": "RENT",
    "annual_inc": 65000,
    "verification_status": "Verified",
    "purpose": "debt_consolidation",
    "dti": 18.5,
    "delinq_2yrs": 0,
    "earliest_cr_line": "Jan-2010",
    "inq_last_6mths": 1,
    "open_acc": 8,
    "pub_rec": 0,
    "revol_bal": 12000,
    "revol_util": 45.0,
    "total_acc": 20,
    "initial_list_status": "w",
    "application_type": "Individual",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_ok_and_model_version(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_version"]


def test_score_returns_a_well_formed_response(client):
    response = client.post("/score", json=VALID_APPLICATION)

    assert response.status_code == 200
    body = response.json()

    assert 0.0 <= body["pd"] <= 1.0
    assert body["risk_band"] in {"Very Low", "Low", "Medium", "High", "Very High"}
    assert body["decision"] in {"Approve", "Manual Review", "Reject"}
    assert len(body["top_reasons"]) == 5
    for reason in body["top_reasons"]:
        assert reason["direction"] in {"increases_risk", "decreases_risk"}
        assert reason["magnitude"] >= 0


def test_score_reports_submitted_values_in_top_reasons_where_covered(client):
    response = client.post("/score", json=VALID_APPLICATION)

    body = response.json()
    reasons_by_feature = {r["feature"]: r["value"] for r in body["top_reasons"]}
    for feature, value in reasons_by_feature.items():
        if feature in VALID_APPLICATION:
            assert value == VALID_APPLICATION[feature]


def test_score_rejects_malformed_request_with_422(client):
    response = client.post("/score", json={"loan_amnt": -5})

    assert response.status_code == 422


def test_score_rejects_invalid_grade_with_422(client):
    malformed = {**VALID_APPLICATION, "grade": "Z"}

    response = client.post("/score", json=malformed)

    assert response.status_code == 422


def test_score_rejects_missing_required_field_with_422(client):
    malformed = dict(VALID_APPLICATION)
    del malformed["dti"]

    response = client.post("/score", json=malformed)

    assert response.status_code == 422
