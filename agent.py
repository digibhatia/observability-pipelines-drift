"""Terminal client (Terminal 2) - while app.py (Terminal 1) is running."""
import time
import requests

BASE = "http://127.0.0.1:8000"

QUESTIONS = [
    "What is the overdraft EAR on a Premier Current Account?",
    "What LTV can I get on a mortgage with the Standard product?",
    "What savings rate does the Premier Fixed Saver pay?",
]


def ask(q: str):
    r = requests.get(f"{BASE}/ask", params={"q": q})
    data = r.json()
    print(f"Q: {q}")
    print(f"A: {data['answer']}")
    print(f"   steps: {data['steps']}  |  tools: {data['tool_sequence']}  |  cost: ${float(data['cost_usd']):.8f}  "
          f"|  model: {data['requested_model']}->{data['served_model']}  |  trace_id: {data['trace_id']}\n")
    return data


def run_domain_questions(n_loops: int = 1):
    """Looping N times matters here in a way it didn't in Nidan: a single
    /ask only ever shows you one trace. Repeated traffic is what gives
    online_eval.py's sampling, drift.py's rolling windows, and alerting.py's
    cost baseline something to actually operate on."""
    print("=== Domain questions ===\n")
    for _ in range(n_loops):
        for q in QUESTIONS:
            ask(q)


def bonus_1_observability_dashboard():
    print("=== M4: build the baseline the dashboard will show ===\n")
    print("Run `docker compose up` (Terminal 1), then open:")
    print("  Jaeger    -> http://localhost:16686")
    print("  Grafana   -> http://localhost:3000  (admin/admin)")
    print("  Prometheus-> http://localhost:9090")
    print("Generating 20 requests so the dashboard panels have something to show...\n")
    run_domain_questions(n_loops=7)  # 7 * 3 = 21 requests


def bonus_2_online_eval_and_drift():
    print("=== M5: online eval + drift status ===\n")
    print("eval/recent:", requests.get(f"{BASE}/eval/recent").json(), "\n")
    print("drift/status:", requests.get(f"{BASE}/drift/status").json(), "\n")


def bonus_3_feedback_and_annotation_queue():
    print("=== M5: human feedback + annotation queue ===\n")
    r = ask("What is the overdraft EAR on a Premier Current Account?")
    requests.post(f"{BASE}/feedback", params={
        "trace_id": r["trace_id"], "rating": "down",
        "note": "Answer was technically correct but did not mention the arranged-overdraft caveat.",
        "question": "What is the overdraft EAR on a Premier Current Account?", "answer": r["answer"],
    })
    print("annotation-queue:", requests.get(f"{BASE}/annotation-queue").json(), "\n")


def run_incident_5_exercise():
    """Simulates the facilitator seeding incident 5 (silent model swap),
    then a learner diagnosing it NOT from a single trace (that's Nidan's
    exercise) but from the dashboard/alerts - the point being that this
    incident genuinely cannot be caught by reading one trace_id, only by
    watching enough traffic that cost, eval score, and drift all have
    something to move against."""
    print("=== Incident exercise: seeding incident 5 (facilitator-only in a real run) ===\n")
    requests.post(f"{BASE}/seed-incident/5")
    print("incident-status (all a learner would see):", requests.get(f"{BASE}/incident-status").json(), "\n")

    print("Learner's view - generating traffic (a real learner would just use the app normally):")
    for _ in range(12):
        ask("What is the overdraft EAR on a Premier Current Account?")
        time.sleep(0.2)

    print("Learner's view - now check /alerts, /eval/recent, /drift/status, and the Grafana cost panel:")
   
response = requests.get(f"{BASE}/alerts", timeout=10)

print("Alerts HTTP status:", response.status_code)

if response.ok and response.text.strip():
    try:
        print("alerts:", response.json(), "\n")
    except requests.exceptions.JSONDecodeError:
        print("Alerts endpoint returned non-JSON content:")
        print(response.text[:500])
else:
    print("Alerts endpoint returned an empty response or HTTP error.")
    print(response.text[:500])


    print("--- write your diagnosis now, THEN check the answer key ---")
    print("reveal-incident:", requests.get(f"{BASE}/reveal-incident/5").json(), "\n")
    requests.post(f"{BASE}/clear-incident")
    requests.post(f"{BASE}/alerts/clear")
    requests.post(f"{BASE}/drift/reset-baseline")


if __name__ == "__main__":
    run_domain_questions()
    bonus_1_observability_dashboard()
    bonus_2_online_eval_and_drift()
    bonus_3_feedback_and_annotation_queue()
    run_incident_5_exercise()
