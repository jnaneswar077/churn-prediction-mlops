import os
import sys
import json
import time
import statistics
import requests

BASE_URL = "http://localhost:8000"
N_REQUESTS = 50
THROUGHPUT_DURATION_SECONDS = 5

VALID_PAYLOAD = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": "29.85"
}


def measure_latency_and_consistency():
    """Fires the SAME request N times. Latency should vary a little
    (normal system noise); the actual prediction output should NEVER
    vary, since the model and input are both fixed."""
    latencies_ms = []
    responses = []
    for _ in range(N_REQUESTS):
        start = time.perf_counter()
        r = requests.post(f"{BASE_URL}/predict", json=VALID_PAYLOAD)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)
        responses.append(r.json())
    return latencies_ms, responses


def measure_throughput(duration_seconds):
    """How many requests can the API actually handle per second,
    sustained over a real time window (not a single best-case call)."""
    count = 0
    start = time.perf_counter()
    while time.perf_counter() - start < duration_seconds:
        requests.post(f"{BASE_URL}/predict", json=VALID_PAYLOAD)
        count += 1
    elapsed = time.perf_counter() - start
    return count, elapsed


def run_performance_tests():
    print(f"[INFO] Checking API is reachable at {BASE_URL}...")
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Could not reach {BASE_URL}.")
        print("[ERROR] Start the server first in another terminal: uvicorn api.main:app")
        return False
    print(f"[INFO] Health check: {health.json()}")

    print(f"\n[INFO] Firing {N_REQUESTS} repeated identical requests to /predict...")
    latencies_ms, responses = measure_latency_and_consistency()

    first_response = responses[0]
    all_identical = all(r == first_response for r in responses)

    print(f"[INFO] Measuring sustained throughput for {THROUGHPUT_DURATION_SECONDS} seconds...")
    req_count, elapsed = measure_throughput(THROUGHPUT_DURATION_SECONDS)
    rps = req_count / elapsed

    report = {
        "requests_sent": N_REQUESTS,
        "latency_ms": {
            "min": round(min(latencies_ms), 2),
            "max": round(max(latencies_ms), 2),
            "mean": round(statistics.mean(latencies_ms), 2),
            "median": round(statistics.median(latencies_ms), 2),
            "stdev": round(statistics.stdev(latencies_ms), 2) if len(latencies_ms) > 1 else 0.0,
        },
        "response_consistency": {
            "identical_input_produces_identical_output": all_identical,
            "sample_response": first_response
        },
        "throughput": {
            "duration_seconds": round(elapsed, 2),
            "requests_completed": req_count,
            "requests_per_second": round(rps, 2)
        }
    }

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/api_performance_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print(f"\n{'='*70}")
    print("API PERFORMANCE SUMMARY")
    print('='*70)
    lat = report["latency_ms"]
    print(f"Latency (ms)  min={lat['min']}  max={lat['max']}  mean={lat['mean']}  "
          f"median={lat['median']}  stdev={lat['stdev']}")
    print(f"Consistency   identical input -> identical output every time: {all_identical}")
    print(f"Throughput    {rps:.2f} requests/sec sustained over {elapsed:.2f}s "
          f"({req_count} requests)")
    print(f"\nReport saved to artifacts/api_performance_report.json")

    # Consistency is the one genuine pass/fail here -- latency and
    # throughput are measurements, not correctness checks, and their
    # "acceptable" range depends on hardware, not a fixed threshold.
    return all_identical


if __name__ == "__main__":
    ok = run_performance_tests()
    sys.exit(0 if ok else 1)