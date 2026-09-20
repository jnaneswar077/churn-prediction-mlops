import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from fastapi.testclient import TestClient
from api.main import app

VALID_PAYLOAD = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": "29.85"
}


def with_override(**overrides):
    d = dict(VALID_PAYLOAD)
    d.update(overrides)
    return d


def without_field(field):
    d = dict(VALID_PAYLOAD)
    del d[field]
    return d


# ---------------------------------------------------------------------
# The required test matrix: valid inputs, missing fields, invalid
# datatypes, corrupted payloads, and extreme values -- against both the
# single and batch endpoints.
# ---------------------------------------------------------------------
TEST_CASES = [
    {"name": "valid_request", "method": "post", "endpoint": "/predict",
     "payload": VALID_PAYLOAD, "expected_status": 200},

    {"name": "missing_required_field", "method": "post", "endpoint": "/predict",
     "payload": without_field("Contract"), "expected_status": 422},

    {"name": "invalid_datatype_tenure_as_string", "method": "post", "endpoint": "/predict",
     "payload": with_override(tenure="not-a-number"), "expected_status": 422},

    {"name": "invalid_categorical_value", "method": "post", "endpoint": "/predict",
     "payload": with_override(InternetService="Satellite"), "expected_status": 422},

    {"name": "extreme_value_negative_tenure", "method": "post", "endpoint": "/predict",
     "payload": with_override(tenure=-5), "expected_status": 422},

    {"name": "extreme_value_tenure_over_max", "method": "post", "endpoint": "/predict",
     "payload": with_override(tenure=9999), "expected_status": 422},

    {"name": "extreme_value_absurd_monthly_charges", "method": "post", "endpoint": "/predict",
     "payload": with_override(MonthlyCharges=999999999), "expected_status": 422},

    {"name": "corrupted_payload_wrong_type_senior_citizen", "method": "post", "endpoint": "/predict",
     "payload": with_override(SeniorCitizen="yes-please"), "expected_status": 422},

    {"name": "corrupted_payload_wrong_structure", "method": "post", "endpoint": "/predict",
     "payload": ["this", "is", "not", "a", "customer", "object"], "expected_status": 422},

    {"name": "batch_empty", "method": "post", "endpoint": "/predict/batch",
     "payload": [], "expected_status": 400},

    {"name": "batch_one_bad_record_among_good", "method": "post", "endpoint": "/predict/batch",
     "payload": [VALID_PAYLOAD, with_override(tenure=-1)], "expected_status": 422},

    {"name": "batch_all_valid", "method": "post", "endpoint": "/predict/batch",
     "payload": [VALID_PAYLOAD, VALID_PAYLOAD], "expected_status": 200},

    {"name": "unmatched_route", "method": "get", "endpoint": "/does-not-exist",
     "payload": None, "expected_status": 404},
]


def run_test(client, case):
    if case["method"] == "get":
        response = client.get(case["endpoint"])
    else:
        response = client.post(case["endpoint"], json=case["payload"])

    status_correct = (response.status_code == case["expected_status"])

    # Every error response (>=400) must ALSO carry the standardized
    # shape from Stage 6 -- not just the right status code.
    error_shape_consistent = "n/a (success response)"
    if response.status_code >= 400:
        body = response.json()
        error_shape_consistent = all(k in body for k in ("error", "status_code", "message", "details"))

    overall_pass = status_correct and (error_shape_consistent in (True, "n/a (success response)"))

    return {
        "name": case["name"],
        "endpoint": case["endpoint"],
        "expected_status": case["expected_status"],
        "actual_status": response.status_code,
        "status_correct": status_correct,
        "error_shape_consistent": error_shape_consistent,
        "overall_pass": overall_pass,
    }


def run_all_tests():
    print("[INFO] Starting Lab 10 Stage 7: API Test Matrix...")

    # Using TestClient as a context manager ensures the app's lifespan
    # (model loading at startup) actually runs -- without "with", the
    # model never loads and every endpoint returns 503.
    with TestClient(app) as client:
        results = [run_test(client, c) for c in TEST_CASES]

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/api_test_report.json", "w") as f:
        json.dump({"test_cases": results}, f, indent=4)

    print(f"\n{'='*70}")
    print("API TEST MATRIX SUMMARY")
    print('='*70)
    all_passed = True
    for r in results:
        status = "PASS" if r["overall_pass"] else "**FAIL**"
        print(f"  {r['name']:42s} expected={r['expected_status']:4d} "
              f"actual={r['actual_status']:4d} -> {status}")
        all_passed = all_passed and r["overall_pass"]

    print(f"\nReport saved to artifacts/api_test_report.json")
    return all_passed


if __name__ == "__main__":
    ok = run_all_tests()
    sys.exit(0 if ok else 1)