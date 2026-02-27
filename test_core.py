"""
test_core.py
------------
Enterprise unit test suite for AegisML.

Covers: metrics, calibration, drift, risk, report, bias, governance.

Run with:
  python -m pytest test_core.py -v
  python test_core.py
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from reliability.metrics import compute_metrics, format_confusion_matrix, compute_threshold_analysis
from reliability.calibration import compute_ece, reliability_diagram, confidence_histogram
from reliability.drift import compute_prediction_drift, compute_feature_drift, drift_heatmap
from reliability.risk import compute_risk_score
from reliability.report import generate_report
from verification.metrics import calculate_psi, calculate_ks, interpret_psi, interpret_ks
from verification.risk_scoring import calculate_risk_score
from verification.governance import map_risk_tier, eu_ai_act_compliance_check
from verification.bias import (
    statistical_parity_difference,
    disparate_impact_ratio,
    equal_opportunity_difference,
    compute_full_bias_report,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _perfect_preds(n=200):
    np.random.seed(42)
    y_true = np.random.randint(0, 2, n)
    y_prob = np.where(y_true == 1,
                      np.random.uniform(0.85, 0.99, n),
                      np.random.uniform(0.01, 0.15, n))
    return y_true, y_prob


def _random_preds(n=200):
    np.random.seed(7)
    y_true = np.random.randint(0, 2, n)
    y_prob = np.random.uniform(0, 1, n)
    return y_true, y_prob


def _overconfident_preds(n=200):
    np.random.seed(13)
    y_true = np.random.randint(0, 2, n)
    y_prob = np.where(y_true == 1,
                      np.random.uniform(0.75, 0.98, n),
                      np.random.uniform(0.35, 0.65, n))
    return y_true, y_prob


# ---------------------------------------------------------------------------
# reliability/metrics.py
# ---------------------------------------------------------------------------

def test_compute_metrics_perfect():
    y_true, y_prob = _perfect_preds()
    result = compute_metrics(y_true, y_prob)
    assert result["accuracy"] > 0.90,   f"Expected accuracy > 0.90, got {result['accuracy']}"
    assert result["f1_weighted"] > 0.85, f"Expected f1 > 0.85"
    assert result["brier_score"] < 0.10, f"Expected brier < 0.10"
    assert result["precision"] > 0.85
    assert result["recall"] > 0.85
    assert len(result["confusion_matrix"]) == 2
    assert "auc_roc" in result
    assert "mcc" in result
    print(f"  [PASS] Perfect preds — acc={result['accuracy']:.3f}, auc={result['auc_roc']}, brier={result['brier_score']:.4f}")


def test_compute_metrics_random():
    y_true, y_prob = _random_preds()
    result = compute_metrics(y_true, y_prob)
    assert 0.3 < result["accuracy"] < 0.7
    assert result["brier_score"] > 0.15
    print(f"  [PASS] Random preds — acc={result['accuracy']:.3f}")


def test_compute_metrics_flags():
    y_true, y_prob = _random_preds()
    result = compute_metrics(y_true, y_prob)
    assert "flags" in result
    assert isinstance(result["flags"]["low_accuracy"], bool)
    print("  [PASS] Metric flags present")


def test_format_confusion_matrix():
    cm = [[90, 10], [5, 95]]
    s = format_confusion_matrix(cm)
    assert "90" in s and "95" in s
    print("  [PASS] Confusion matrix formatting")


def test_threshold_analysis():
    y_true, y_prob = _perfect_preds()
    results = compute_threshold_analysis(y_true, y_prob)
    assert len(results) > 5
    assert "threshold" in results[0]
    assert "f1_weighted" in results[0]
    print(f"  [PASS] Threshold analysis — {len(results)} thresholds tested")


# ---------------------------------------------------------------------------
# reliability/calibration.py
# ---------------------------------------------------------------------------

def test_ece_perfect():
    y_true, y_prob = _perfect_preds()
    result = compute_ece(y_true, y_prob)
    assert result["ece"] < 0.12, f"Expected low ECE, got {result['ece']}"
    assert result["mce"] < 0.20
    assert len(result["bins"]) == 10
    print(f"  [PASS] ECE perfect — ece={result['ece']:.4f}")


def test_ece_overconfident():
    y_true, y_prob = _overconfident_preds()
    result = compute_ece(y_true, y_prob)
    assert result["overconfidence_gap"] > 0
    print(f"  [PASS] ECE overconfident — ece={result['ece']:.4f}, gap={result['overconfidence_gap']:.4f}")


def test_reliability_diagram_bytes():
    y_true, y_prob = _perfect_preds()
    img_bytes = reliability_diagram(y_true, y_prob, return_bytes=True)
    assert isinstance(img_bytes, bytes)
    assert len(img_bytes) > 1000
    print(f"  [PASS] Reliability diagram — {len(img_bytes):,} bytes")


def test_confidence_histogram_bytes():
    _, y_prob = _perfect_preds()
    img_bytes = confidence_histogram(y_prob, return_bytes=True)
    assert isinstance(img_bytes, bytes)
    assert len(img_bytes) > 1000
    print(f"  [PASS] Confidence histogram — {len(img_bytes):,} bytes")


# ---------------------------------------------------------------------------
# reliability/drift.py
# ---------------------------------------------------------------------------

def test_prediction_drift_no_drift():
    np.random.seed(1)
    probs  = np.random.beta(2, 2, 200)
    result = compute_prediction_drift(probs, probs + np.random.normal(0, 0.01, 200).clip(-0.05, 0.05))
    assert result["psi"] < 0.10, f"Minimal drift expected, got PSI={result['psi']}"
    print(f"  [PASS] Prediction drift (no shift) — psi={result['psi']:.4f}")


def test_prediction_drift_significant():
    np.random.seed(2)
    ref_probs  = np.random.beta(2, 5, 200)
    prod_probs = np.random.beta(5, 2, 200)
    result = compute_prediction_drift(ref_probs, prod_probs)
    assert result["psi"] > 0.1
    print(f"  [PASS] Prediction drift (significant) — psi={result['psi']:.4f}")


def test_feature_drift():
    np.random.seed(3)
    ref_df  = pd.DataFrame({"f1": np.random.normal(0, 1, 100), "f2": np.random.normal(5, 2, 100), "f3": np.random.normal(-1, 0.5, 100)})
    prod_df = pd.DataFrame({"f1": np.random.normal(3, 1.5, 100), "f2": np.random.normal(5.5, 2, 100), "f3": np.random.normal(-1, 0.5, 100)})
    result  = compute_feature_drift(ref_df, prod_df)
    assert result["n_features"] == 3
    assert result["features"]["f1"]["psi"] > result["features"]["f3"]["psi"]
    print(f"  [PASS] Feature drift — f1_psi={result['features']['f1']['psi']:.4f}, f3_psi={result['features']['f3']['psi']:.4f}")


def test_drift_heatmap_bytes():
    np.random.seed(4)
    ref_df  = pd.DataFrame({"a": np.random.normal(0, 1, 100), "b": np.random.normal(0, 1, 100)})
    prod_df = pd.DataFrame({"a": np.random.normal(2, 1, 100), "b": np.random.normal(0.1, 1, 100)})
    drift   = compute_feature_drift(ref_df, prod_df)
    img_bytes = drift_heatmap(drift, return_bytes=True)
    assert isinstance(img_bytes, bytes) and len(img_bytes) > 500
    print(f"  [PASS] Drift heatmap — {len(img_bytes):,} bytes")


# ---------------------------------------------------------------------------
# reliability/risk.py
# ---------------------------------------------------------------------------

def test_risk_low():
    risk = compute_risk_score(ece=0.02, brier_score=0.05, accuracy=0.95, drift_score=0.03)
    assert risk["overall_risk_level"] == "Low"
    assert risk["overall_score"] == 0
    assert "eu_ai_act_category" in risk
    assert "uk_ai_framework" in risk
    assert not risk["requires_human_oversight"]
    print(f"  [PASS] Risk low — level={risk['overall_risk_level']}, eu_act={risk['eu_ai_act_category']}")


def test_risk_high():
    risk = compute_risk_score(ece=0.18, brier_score=0.25, accuracy=0.62, drift_score=0.35)
    assert risk["overall_risk_level"] in ("High", "Critical")
    assert risk["requires_human_oversight"]
    print(f"  [PASS] Risk high — level={risk['overall_risk_level']}")


def test_risk_no_drift():
    risk = compute_risk_score(ece=0.08, brier_score=0.14, accuracy=0.84, drift_score=None)
    assert risk["overall_risk_level"] in ("Low", "Medium")
    assert isinstance(risk["recommendations"], list)
    print(f"  [PASS] Risk no drift — level={risk['overall_risk_level']}, recs={len(risk['recommendations'])}")


def test_risk_verdict_present():
    risk = compute_risk_score(ece=0.05, brier_score=0.10, accuracy=0.88)
    assert isinstance(risk["deployment_verdict"], str) and len(risk["deployment_verdict"]) > 10
    print("  [PASS] Risk verdict present")


# ---------------------------------------------------------------------------
# verification/metrics.py + risk_scoring.py
# ---------------------------------------------------------------------------

def test_calculate_psi_no_drift():
    np.random.seed(10)
    a = np.random.normal(0.5, 0.1, 500)
    b = np.random.normal(0.5, 0.1, 500)
    psi = calculate_psi(a, b)
    assert psi < 0.10, f"Expected low PSI, got {psi}"
    print(f"  [PASS] PSI no drift — {psi:.5f}")


def test_calculate_psi_drift():
    np.random.seed(11)
    a = np.random.normal(0.5, 0.1, 500)
    b = np.random.normal(0.8, 0.1, 500)
    psi = calculate_psi(a, b)
    assert psi > 0.20
    print(f"  [PASS] PSI significant drift — {psi:.5f}")


def test_calculate_ks():
    np.random.seed(12)
    a = np.random.normal(0, 1, 200)
    b = np.random.normal(0, 1, 200)
    stat, p = calculate_ks(a, b)
    assert p > 0.05
    print(f"  [PASS] KS no drift — stat={stat:.4f}, p={p:.4f}")


def test_risk_score_low():
    score = calculate_risk_score(psi=0.02, p_value=0.80, bias_score=0.0)
    assert score < 30
    print(f"  [PASS] Risk score low — {score}")


def test_risk_score_high():
    score = calculate_risk_score(psi=0.45, p_value=0.001, bias_score=0.3)
    assert score > 60
    print(f"  [PASS] Risk score high — {score}")


def test_map_risk_tier():
    assert map_risk_tier(15)[0] == "LOW"
    assert map_risk_tier(50)[0] == "MEDIUM"
    assert map_risk_tier(75)[0] == "HIGH"
    assert map_risk_tier(92)[0] == "CRITICAL"
    print("  [PASS] Risk tier mapping — all 4 tiers correct")


# ---------------------------------------------------------------------------
# verification/governance.py
# ---------------------------------------------------------------------------

def test_eu_ai_act_compliance_low_risk():
    result = eu_ai_act_compliance_check(risk_score=20, psi=0.03, p_value=0.80)
    assert result["overall_status"] == "COMPLIANT"
    assert not result["conformity_assessment_required"]
    print(f"  [PASS] EU AI Act compliance (low risk) — {result['overall_status']}")


def test_eu_ai_act_compliance_high_risk():
    result = eu_ai_act_compliance_check(risk_score=80, psi=0.35, p_value=0.001)
    assert result["overall_status"] == "ACTION_REQUIRED"
    assert result["conformity_assessment_required"]
    print(f"  [PASS] EU AI Act compliance (high risk) — {result['overall_status']}")


# ---------------------------------------------------------------------------
# verification/bias.py
# ---------------------------------------------------------------------------

def test_bias_spd_fair():
    np.random.seed(20)
    n = 500
    protected = np.random.randint(0, 2, n)
    y_pred    = np.random.randint(0, 2, n)   # random, should be ~fair
    spd = statistical_parity_difference(y_pred, protected)
    assert abs(spd) < 0.15, f"Expected small SPD, got {spd}"
    print(f"  [PASS] SPD fair — {spd:.4f}")


def test_bias_spd_unfair():
    np.random.seed(21)
    n = 500
    protected = np.random.randint(0, 2, n)
    # Deliberately bias group 1
    y_pred = np.where(protected == 1,
                      np.random.choice([0, 1], n, p=[0.1, 0.9]),
                      np.random.choice([0, 1], n, p=[0.6, 0.4]))
    spd = statistical_parity_difference(y_pred, protected)
    assert abs(spd) > 0.3, f"Expected large SPD, got {spd}"
    print(f"  [PASS] SPD unfair — {spd:.4f}")


def test_full_bias_report():
    np.random.seed(22)
    n = 500
    protected = np.random.randint(0, 2, n)
    y_true = np.random.randint(0, 2, n)
    y_pred = np.random.randint(0, 2, n)
    report = compute_full_bias_report(y_true, y_pred, protected)
    assert "statistical_parity_difference" in report
    assert "disparate_impact_ratio" in report
    assert "compliance" in report
    assert "eu_ai_act_bias_obligation" in report
    print(f"  [PASS] Full bias report — compliant={report['compliance']['overall_compliant']}")


# ---------------------------------------------------------------------------
# reliability/report.py
# ---------------------------------------------------------------------------

def test_generate_report():
    y_true, y_prob = _perfect_preds(300)
    metrics     = compute_metrics(y_true, y_prob)
    calibration = compute_ece(y_true, y_prob)
    risk        = compute_risk_score(
        ece=calibration["ece"],
        brier_score=metrics["brier_score"],
        accuracy=metrics["accuracy"],
        drift_score=0.07,
        auc_roc=metrics.get("auc_roc"),
    )
    rel_bytes   = reliability_diagram(y_true, y_prob, return_bytes=True)
    hist_bytes  = confidence_histogram(y_prob, return_bytes=True)

    np.random.seed(5)
    ref_df  = pd.DataFrame({f"feat{i}": np.random.normal(i, 1, 100) for i in range(4)})
    prod_df = pd.DataFrame({f"feat{i}": np.random.normal(i + 0.5, 1.2, 100) for i in range(4)})
    drift   = compute_feature_drift(ref_df, prod_df)
    pred_drift  = compute_prediction_drift(y_prob, y_prob + np.random.normal(0, 0.05, len(y_prob)))
    drift_chart = drift_heatmap(drift, return_bytes=True)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        out_path = f.name

    result_path = generate_report(
        output_path=out_path,
        model_name="AegisML-TestModel-v5.0",
        metrics=metrics,
        calibration=calibration,
        risk=risk,
        drift=drift,
        prediction_drift=pred_drift,
        reliability_diagram_bytes=rel_bytes,
        confidence_hist_bytes=hist_bytes,
        drift_chart_bytes=drift_chart,
        n_samples=len(y_true),
    )

    assert os.path.exists(result_path)
    file_size = os.path.getsize(result_path)
    assert file_size > 50_000, f"PDF too small: {file_size} bytes"
    os.unlink(result_path)
    print(f"  [PASS] PDF report generated — {file_size:,} bytes")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    tests = [
        # metrics
        ("compute_metrics (perfect)",     test_compute_metrics_perfect),
        ("compute_metrics (random)",      test_compute_metrics_random),
        ("compute_metrics (flags)",       test_compute_metrics_flags),
        ("format_confusion_matrix",       test_format_confusion_matrix),
        ("threshold_analysis",            test_threshold_analysis),
        # calibration
        ("ECE (perfect)",                 test_ece_perfect),
        ("ECE (overconfident)",           test_ece_overconfident),
        ("reliability_diagram bytes",     test_reliability_diagram_bytes),
        ("confidence_histogram bytes",    test_confidence_histogram_bytes),
        # drift
        ("prediction_drift (no drift)",   test_prediction_drift_no_drift),
        ("prediction_drift (significant)",test_prediction_drift_significant),
        ("feature_drift",                 test_feature_drift),
        ("drift_heatmap bytes",           test_drift_heatmap_bytes),
        # risk
        ("risk (low)",                    test_risk_low),
        ("risk (high)",                   test_risk_high),
        ("risk (no drift)",               test_risk_no_drift),
        ("risk verdict",                  test_risk_verdict_present),
        # verification
        ("PSI no drift",                  test_calculate_psi_no_drift),
        ("PSI drift",                     test_calculate_psi_drift),
        ("KS test",                       test_calculate_ks),
        ("risk score low",                test_risk_score_low),
        ("risk score high",               test_risk_score_high),
        ("risk tier mapping",             test_map_risk_tier),
        # governance
        ("EU AI Act (low risk)",          test_eu_ai_act_compliance_low_risk),
        ("EU AI Act (high risk)",         test_eu_ai_act_compliance_high_risk),
        # bias
        ("SPD fair",                      test_bias_spd_fair),
        ("SPD unfair",                    test_bias_spd_unfair),
        ("full bias report",              test_full_bias_report),
        # report
        ("generate_report PDF",           test_generate_report),
    ]

    passed = failed = 0
    print("\n" + "=" * 65)
    print("  AegisML Enterprise v5.0.0 — Test Suite")
    print("=" * 65)

    for name, test_fn in tests:
        print(f"\n▶ {name}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    print("\n" + "=" * 65)
    print(f"  Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 65 + "\n")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
