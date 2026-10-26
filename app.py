import io
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, Response, send_file
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
import xgboost as xgb
import shap
from waitress import serve

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as rl_colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

app = Flask(__name__)

# ==========================================
# 1. محرك الذكاء الاصطناعي (الخلفية)
# ==========================================

NASA_FEATURES = ["s_2", "s_3", "s_4", "s_6", "s_7", "s_8", "s_9",
                 "s_11", "s_12", "s_13", "s_14", "s_15", "s_17", "s_20", "s_21"]
WINDOW = 10
RUL_CAP = 130
FAILURE_HORIZON = 30


class PredictiveEngine:
    def __init__(self):
        self.regressor = None
        self.classifier = None
        self.scaler = None
        self.feature_cols = []
        self.explainer_reg = None
        self.explainer_clf = None
        self.metrics = {}
        self.trained = False
        self.data_source = "none"
        self.demo_samples = {}

    def _synthetic_fallback(self):
        print("[AEGIS] Generating synthetic data for fallback...")
        rng = np.random.default_rng(42)
        rows = []
        for unit in range(1, 81):
            life = int(rng.integers(150, 320))
            for cycle in range(1, life + 1):
                progress = cycle / life
                deg = progress ** 1.7
                row = {"unit": unit, "cycle": cycle}
                for i in range(1, 22):
                    coef = rng.normal(0.55, 0.35) if f"s_{i}" in NASA_FEATURES else rng.normal(0.05, 0.15)
                    row[f"s_{i}"] = float(rng.normal(0, 0.4) + deg * coef * (1 + 0.05 * i) + rng.normal(0, 0.15))
                row["RUL"] = life - cycle
                rows.append(row)
        return pd.DataFrame(rows)

    def _extract_slice(self, df, unit, mode="start", length=50):
        unit_df = df[df["unit"] == unit].sort_values("cycle").reset_index(drop=True)
        if len(unit_df) < length:
            length = len(unit_df)
        if mode == "end":
            slice_df = unit_df.tail(length)
        elif mode == "mid":
            start = max(0, len(unit_df) // 2 - length // 2)
            slice_df = unit_df.iloc[start:start + length]
        else:
            slice_df = unit_df.head(length)
        cols = ["cycle"] + NASA_FEATURES
        out = slice_df[cols].copy()
        rename = {f"s_{i}": f"sensor_{i}" for i in range(1, 22)}
        return out.rename(columns=rename).round(4).to_dict("records")

    def _engineer(self, df):
        df = df.copy().sort_values(["unit", "cycle"]).reset_index(drop=True)
        grp = df.groupby("unit")
        feats = []
        for col in NASA_FEATURES:
            df[f"{col}_m"] = grp[col].transform(lambda s: s.rolling(WINDOW, min_periods=1).mean())
            df[f"{col}_s"] = grp[col].transform(lambda s: s.rolling(WINDOW, min_periods=1).std()).fillna(0.0)
            df[f"{col}_l"] = grp[col].shift(5).bfill()
            feats += [col, f"{col}_m", f"{col}_s", f"{col}_l"]
        return df.fillna(0.0), feats

    def train(self):
        raw = self._synthetic_fallback()
        self.data_source = "Synthetic Data (NASA dataset not found)"

        for i in range(1, 22):
            if f"s_{i}" not in raw.columns:
                raw[f"s_{i}"] = 0.0

        self.demo_samples = {
            "healthy": self._extract_slice(raw, unit=1, mode="start", length=50),
            "degrading": self._extract_slice(raw, unit=50, mode="mid", length=50),
            "critical": self._extract_slice(raw, unit=95, mode="end", length=50),
        }

        df, feats = self._engineer(raw)
        self.feature_cols = feats
        df["RUL_capped"] = df["RUL"].clip(upper=RUL_CAP)
        df["failure"] = (df["RUL"] <= FAILURE_HORIZON).astype(int)

        X, y_reg, y_clf = df[feats].values, df["RUL_capped"].values, df["failure"].values
        split = int(len(df) * 0.8)
        X_tr, X_te = X[:split], X[split:]

        self.scaler = StandardScaler().fit(X_tr)
        Xs_tr, Xs_te = self.scaler.transform(X_tr), self.scaler.transform(X_te)

        self.regressor = xgb.XGBRegressor(n_estimators=250, max_depth=6, learning_rate=0.07,
                                          subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1)
        self.regressor.fit(Xs_tr, y_reg[:split])

        self.classifier = xgb.XGBClassifier(n_estimators=220, max_depth=6, learning_rate=0.07,
                                            subsample=0.9, colsample_bytree=0.9, random_state=42,
                                            n_jobs=-1, eval_metric="logloss")
        self.classifier.fit(Xs_tr, y_clf[:split])

        pred_rul = self.regressor.predict(Xs_te)
        pred_fail = self.classifier.predict(Xs_te)

        self.metrics = {
            "rmse": round(float(np.sqrt(mean_squared_error(y_reg[split:], pred_rul))), 3),
            "r2": round(float(r2_score(y_reg[split:], pred_rul)), 4),
            "accuracy": round(float(accuracy_score(y_clf[split:], pred_fail)), 4),
            "f1": round(float(f1_score(y_clf[split:], pred_fail, zero_division=0)), 4),
            "samples": int(len(df)),
            "features": len(feats),
            "data_source": self.data_source,
        }

        self.explainer_reg = shap.TreeExplainer(self.regressor)
        self.explainer_clf = shap.TreeExplainer(self.classifier)
        self.trained = True
        return self.metrics

    def _build_features(self, history):
        df = pd.DataFrame(history)
        for i in range(1, 22):
            sc, s = f"sensor_{i}", f"s_{i}"
            if sc in df.columns and s not in df.columns:
                df[s] = pd.to_numeric(df[sc], errors="coerce")
        if "unit" not in df.columns: df["unit"] = 1
        if "cycle" not in df.columns: df["cycle"] = np.arange(1, len(df) + 1)
        for col in NASA_FEATURES:
            if col not in df.columns: df[col] = 0.0
        df_eng, _ = self._engineer(df)
        return df_eng.iloc[[-1]][self.feature_cols].values

    def predict(self, history, unit_id="UNIT-001"):
        if not self.trained:
            raise RuntimeError("Engine not trained.")
        Xs = self.scaler.transform(self._build_features(history))
        rul = float(np.clip(self.regressor.predict(Xs)[0], 0, RUL_CAP))
        prob = float(self.classifier.predict_proba(Xs)[0][1])

        sv_reg = self.explainer_reg.shap_values(Xs)[0]
        sv_clf = self.explainer_clf.shap_values(Xs)[0]

        importance = []
        for sensor in NASA_FEATURES:
            idxs = [self.feature_cols.index(v) for v in
                    [sensor, f"{sensor}_m", f"{sensor}_s", f"{sensor}_l"]
                    if v in self.feature_cols]
            impact = max(abs(sv_clf[i]) for i in idxs)
            signed = float(sv_reg[idxs[1]] if len(idxs) > 1 else sv_reg[idxs[0]])
            importance.append({"sensor": f"sensor_{sensor[2:]}", "impact": round(float(impact), 4),
                               "shap_rul": round(signed, 4)})
        importance.sort(key=lambda x: x["impact"], reverse=True)

        if prob >= 0.75 or rul <= 15:
            risk = "critical"
        elif prob >= 0.5 or rul <= 45:
            risk = "high"
        elif prob >= 0.25 or rul <= 90:
            risk = "medium"
        else:
            risk = "low"

        return {
            "unit_id": unit_id, "rul": round(rul, 2),
            "failure_probability": round(prob, 4), "risk_level": risk,
            "top_sensors": importance[:5],
            "sensor_snapshot": {k: round(float(v), 4) for k, v in history[-1].items() if k.startswith("sensor_")},
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data_source": self.data_source,
        }


engine = PredictiveEngine()


# ==========================================
# 2. معالجة البيانات والمساعد الذكي (API)
# ==========================================

def _parse_uploaded_file(file_storage):
    filename = (file_storage.filename or "").lower()
    if filename.endswith(".csv"):
        df = pd.read_csv(file_storage)
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_storage)
    else:
        raise ValueError("Unsupported format. Upload .csv or .xlsx")
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    mapping = {}
    for col in df.columns:
        if col in ("unit", "unit_id", "engine", "engine_id", "asset_id"):
            mapping[col] = "unit"
        elif col in ("cycle", "time_cycles", "time", "cycles"):
            mapping[col] = "cycle"
        elif col.startswith("sensor_"):
            mapping[col] = col
        elif col.startswith("s_") and col[2:].isdigit():
            mapping[col] = f"sensor_{col[2:]}"
    df = df.rename(columns=mapping)
    if "cycle" not in df.columns: df["cycle"] = np.arange(1, len(df) + 1)
    if "unit" not in df.columns: df["unit"] = 1
    return df


def _history_from_df(df):
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    records = []
    for _, row in df.iterrows():
        rec = {"cycle": int(row["cycle"])}
        for i in range(1, 22):
            col = f"sensor_{i}"
            if col in df.columns:
                val = pd.to_numeric(row[col], errors="coerce")
                if pd.notna(val): rec[col] = float(val)
        records.append(rec)
    return records


def _smart_assistant(question, prediction, lang="ar"):
    q = question.strip().lower()
    unit = prediction.get("unit_id", "—")
    risk = prediction.get("risk_level", "unknown")
    rul = float(prediction.get("rul", 0))
    prob = float(prediction.get("failure_probability", 0)) * 100
    top = prediction.get("top_sensors", [])
    snapshot = prediction.get("sensor_snapshot", {})

    risk_ar = {"low": "منخفض", "medium": "متوسط", "high": "مرتفع", "critical": "حرج"}
    risk_en = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH", "critical": "CRITICAL"}

    sensor_match = re.search(r"s(?:ensor)?[_\s]?(\d+)", q)
    mentioned = f"sensor_{sensor_match.group(1)}" if sensor_match else None

    parts = []
    if lang == "ar":
        is_summary = any(k in q for k in ["ملخص", "حالة", "الوضع", "كل شي", "ايش", "وش", "عام", "شامل"])
        is_risk = any(k in q for k in ["خطر", "خطورة", "احتمال", "فشل", "risk", "probab", "failure"])
        is_rul = any(k in q for k in ["rul", "عمر", "متبقي", "باقي", "life", "remaining", "كم"])
        is_sensor = any(k in q for k in ["حساس", "sensor", "سبب", "cause"]) or mentioned
        is_maint = any(k in q for k in ["توصي", "صيانة", "mainten", "recommend", "افعل", "action", "خطة", "نصيحة"])

        if is_summary:
            parts.append(
                f"<b>📊 ملخص حالة الآلة {unit}:</b>\n• مستوى الخطورة: <b>{risk_ar.get(risk, risk).upper()}</b>\n• احتمال الفشل: <b>{prob:.1f}%</b>\n• العمر المتبقي: <b>{rul:.1f} دورة</b>")
        if is_risk and not is_summary:
            adv = {"critical": "الوضع حرج ويستوجب تدخلاً فورياً.", "high": "يُنصح بجدولة فحص وقائي خلال 24-48 ساعة.",
                   "medium": "الوضع تحت السيطرة مع رفع وتيرة المراقبة.", "low": "الوضع مستقر."}.get(risk, "")
            parts.append(
                f"<b>تقييم الخطورة:</b> مستوى <b>{risk_ar.get(risk, risk)}</b> مع احتمال فشل <b>{prob:.1f}%</b>. {adv}")
        if is_rul and not is_summary:
            parts.append(f"<b>العمر الافتراضي المتبقي (RUL):</b> <b>{rul:.1f}</b> دورة تشغيلية.")
        if is_sensor:
            if mentioned and mentioned in snapshot:
                parts.append(f"<b>{mentioned}:</b> قراءته الحالية <b>{snapshot[mentioned]}</b>.")
            else:
                lines = "\n".join(
                    [f"  {i + 1}. <b>{s['sensor']}</b> — تأثير {s['impact']:.3f}" for i, s in enumerate(top[:3])])
                parts.append(f"<b>الحساسات الأكثر تأثيراً في قرار النموذج:</b>\n{lines}")
        if is_maint:
            parts.append(f"<b>توصية الصيانة:</b> يُنصح بفحص الحساسات المذكورة أعلاه والتحقق من معايرتها.")
        if not parts:
            parts.append(
                f"يمكنني مساعدتك في تحليل الآلة <b>{unit}</b>. جرّب أسئلة مثل: 'ما مستوى الخطورة؟'، 'كم العمر المتبقي؟'.")
    else:
        is_summary = any(k in q for k in ["summary", "status", "overview", "state", "everything"])
        is_risk = any(k in q for k in ["risk", "probab", "failure", "danger"])
        is_rul = any(k in q for k in ["rul", "life", "remaining", "how much", "left"])
        is_sensor = any(k in q for k in ["sensor", "cause", "which"]) or mentioned
        is_maint = any(k in q for k in ["maint", "recommend", "action", "plan", "advice", "do"])

        if is_summary:
            parts.append(
                f"<b>📊 Unit {unit} summary:</b>\n• Risk level: <b>{risk_en.get(risk, risk)}</b>\n• Failure probability: <b>{prob:.1f}%</b>\n• RUL: <b>{rul:.1f} cycles</b>")
        if is_risk and not is_summary:
            parts.append(
                f"<b>Risk assessment:</b> Level <b>{risk_en.get(risk, risk)}</b> with failure probability <b>{prob:.1f}%</b>.")
        if is_rul and not is_summary:
            parts.append(f"<b>Remaining Useful Life (RUL):</b> <b>{rul:.1f}</b> cycles.")
        if is_sensor:
            if mentioned and mentioned in snapshot:
                parts.append(f"<b>{mentioned}:</b> Current reading <b>{snapshot[mentioned]}</b>.")
            else:
                lines = "\n".join(
                    [f"  {i + 1}. <b>{s['sensor']}</b> — impact {s['impact']:.3f}" for i, s in enumerate(top[:3])])
                parts.append(f"<b>Top contributing sensors:</b>\n{lines}")
        if is_maint:
            parts.append(f"<b>Maintenance recommendation:</b> Inspect flagged sensors and verify calibration.")
        if not parts:
            parts.append(
                f"I can help you analyze unit <b>{unit}</b>. Try asking: 'What is the risk?' or 'How much RUL remains?'.")

    return "\n\n".join(parts)


# ==========================================
# 3. مسارات الويب (Web Routes & API)
# ==========================================

@app.route('/')
def index():
    return Response(HTML_CONTENT, mimetype='text/html')


@app.route("/api/demo", methods=["POST"])
def demo():
    body = request.get_json(silent=True) or {}
    scenario = body.get("scenario", "healthy")
    if scenario not in engine.demo_samples or not engine.demo_samples[scenario]:
        return jsonify({"error": "Demo samples unavailable."}), 400
    history = engine.demo_samples[scenario]
    unit_id = f"DEMO-{scenario[:3].upper()}-{np.random.randint(100, 999)}"
    prediction = engine.predict(history, unit_id=unit_id)
    return jsonify({"unit_id": unit_id, "scenario": scenario, "sensor_history": history, "prediction": prediction,
                    "is_demo": True})


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files: return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if not file.filename: return jsonify({"error": "Empty filename"}), 400
    try:
        df = _parse_uploaded_file(file)
        if len(df) < 2: return jsonify({"error": "File must contain at least 2 rows"}), 400
        history = _history_from_df(df)
        unit_id = request.form.get("unit_id") or "UPLOADED-UNIT"
        prediction = engine.predict(history, unit_id=unit_id)
        return jsonify({"unit_id": unit_id, "sensor_history": history, "prediction": prediction, "rows": len(history),
                        "is_demo": False})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/manual", methods=["POST"])
def manual():
    body = request.get_json(silent=True) or {}
    readings = body.get("readings", [])
    if not isinstance(readings, list) or len(readings) < 1:
        return jsonify({"error": "Provide at least 1 reading row"}), 400
    unit_id = body.get("unit_id", "MANUAL-UNIT")
    history = []
    for idx, row in enumerate(readings, start=1):
        rec = {"cycle": int(row.get("cycle", idx))}
        for key, val in row.items():
            if key.startswith("sensor_"):
                try:
                    rec[key] = float(val)
                except (TypeError, ValueError):
                    continue
        history.append(rec)
    try:
        prediction = engine.predict(history, unit_id=unit_id)
        return jsonify({"unit_id": unit_id, "sensor_history": history, "prediction": prediction, "is_demo": False})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/assistant", methods=["POST"])
def assistant():
    body = request.get_json(silent=True) or {}
    question = body.get("question", "")
    prediction = body.get("prediction", {})
    lang = body.get("lang", "ar")
    if not question.strip(): return jsonify({"error": "Empty question"}), 400
    return jsonify({"answer": _smart_assistant(question, prediction, lang=lang), "lang": lang})


@app.route("/api/report", methods=["POST"])
def report():
    data = request.get_json(silent=True) or {}
    pred = data.get("prediction", {})
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm,
                            bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=20,
                           textColor=rl_colors.HexColor("#0f3d2e"), spaceAfter=6)
    sub = ParagraphStyle("s", parent=styles["Normal"], fontName="Helvetica", fontSize=9,
                         textColor=rl_colors.HexColor("#3d6b58"), spaceAfter=16)
    h = ParagraphStyle("h", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12,
                       textColor=rl_colors.HexColor("#0f3d2e"), spaceBefore=12, spaceAfter=6)
    normal = ParagraphStyle("n", parent=styles["Normal"], fontName="Helvetica", fontSize=10,
                            textColor=rl_colors.HexColor("#1e3a2e"), leading=15)

    story = [
        Paragraph("AEGIS — Predictive Maintenance Report", title),
        Paragraph(f"Unit: {pred.get('unit_id', '—')} · Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
                  sub),
        Paragraph("Executive Summary", h),
        Paragraph(
            f"Unit <b>{pred.get('unit_id', '—')}</b> shows a failure probability of <b>{pred.get('failure_probability', 0) * 100:.2f}%</b> with an estimated RUL of <b>{pred.get('rul', 0)} cycles</b>. Risk: <b>{pred.get('risk_level', '—').upper()}</b>.",
            normal),
    ]
    table_data = [["Metric", "Value"], ["RUL", f"{pred.get('rul', 0)} cycles"],
                  ["Failure Probability", f"{pred.get('failure_probability', 0) * 100:.2f}%"],
                  ["Risk Level", pred.get("risk_level", "—").upper()]]
    t = Table(table_data, colWidths=[8 * cm, 8 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#0f3d2e")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
                           ("GRID", (0, 0), (-1, -1), 0.4, rl_colors.HexColor("#a7d4bd"))]))
    story += [t, Paragraph("Root-Cause Sensors", h)]
    rows = [["Sensor", "SHAP Impact"]]
    for s in pred.get("top_sensors", [])[:5]:
        rows.append([s["sensor"], f"{s['impact']:.4f}"])
    st = Table(rows, colWidths=[8 * cm, 8 * cm])
    st.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#0f3d2e")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
                            ("GRID", (0, 0), (-1, -1), 0.4, rl_colors.HexColor("#a7d4bd"))]))
    story.append(st)
    doc.build(story)
    buf.seek(0)
    return send_file(buf, mimetype="application/pdf", as_attachment=True,
                     download_name=f"AEGIS_Report_{pred.get('unit_id', 'UNIT')}.pdf")


# ==========================================
# 4. الواجهة الأمامية (HTML + CSS + JS)
# ==========================================

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en" dir="ltr">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AEGIS · Predictive Maintenance</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Tajawal:wght@400;500;700&display=swap" rel="stylesheet" />
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    :root {
      --bg: #050811; --card: rgba(14, 20, 40, 0.55); --border: rgba(99, 130, 246, 0.2);
      --border-strong: rgba(99, 130, 246, 0.5); --primary: #4f7cff; --primary-bright: #7ba3ff;
      --violet: #a78bfa; --cyan: #22d3ee; --gold: #fbbf24; --danger: #f43f5e; --critical: #ef4444;
      --text: #e8ecf8; --muted: #8896b8; --dim: #4f5a7a;
    }
    html, body { min-height: 100vh; }
    body { font-family: 'Inter', 'Tajawal', sans-serif; color: var(--text); background: var(--bg); padding: 24px clamp(16px, 4vw, 56px) 40px; position: relative; overflow-x: hidden; }

    /* خلفية النجوم والفضاء */
    #starsCanvas { position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -2; pointer-events: none; }
    .aurora { position: fixed; inset: 0; background: radial-gradient(ellipse 70% 50% at 15% 0%, rgba(79, 124, 255, 0.15), transparent 60%), radial-gradient(ellipse 60% 40% at 85% 15%, rgba(167, 139, 250, 0.1), transparent 60%), radial-gradient(ellipse 50% 40% at 50% 100%, rgba(34, 211, 238, 0.05), transparent 60%); pointer-events: none; z-index: -1; }
    .grid-overlay { position: fixed; inset: 0; background-image: linear-gradient(rgba(99, 130, 246, 0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(99, 130, 246, 0.03) 1px, transparent 1px); background-size: 56px 56px; pointer-events: none; z-index: -1; mask-image: radial-gradient(ellipse at center, black 40%, transparent 90%); }

    header, main, footer { position: relative; z-index: 1; }
    .topbar { display: flex; justify-content: space-between; align-items: center; padding-bottom: 20px; border-bottom: 1px solid var(--border); margin-bottom: 40px; flex-wrap: wrap; gap: 16px; }
    .brand { display: flex; align-items: center; gap: 16px; }
    .logo-mark { 
      width: 50px; height: 50px; border-radius: 14px; 
      background: linear-gradient(135deg, rgba(79, 124, 255, 0.4), rgba(167, 139, 250, 0.3)); 
      border: 1px solid var(--border-strong); display: grid; place-items: center; 
      color: #fff; box-shadow: 0 0 30px rgba(79, 124, 255, 0.5), inset 0 0 15px rgba(255,255,255,0.2); 
      transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .brand:hover .logo-mark { transform: rotate(10deg) scale(1.08); box-shadow: 0 0 45px rgba(79, 124, 255, 0.8), inset 0 0 20px rgba(255,255,255,0.4); }
    .brand-text h1 { font-family: 'Outfit', sans-serif; font-size: 26px; font-weight: 800; letter-spacing: 5px; background: linear-gradient(135deg, #ffffff 0%, #7ba3ff 50%, #a78bfa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 0 20px rgba(79, 124, 255, 0.3); }

    .lang-toggle { display: flex; gap: 4px; padding: 4px; background: rgba(15, 20, 40, 0.8); border: 1px solid var(--border); border-radius: 10px; backdrop-filter: blur(10px); }
    .lang-btn { border: none; background: transparent; color: var(--muted); font-size: 12px; font-weight: 600; padding: 8px 16px; border-radius: 7px; cursor: pointer; transition: all 0.25s; }
    .lang-btn.active { background: linear-gradient(135deg, rgba(79, 124, 255, 0.4), rgba(167, 139, 250, 0.3)); color: #fff; box-shadow: 0 0 15px rgba(79, 124, 255, 0.3); }

    .hero { margin-bottom: 40px; text-align: center; }
    .hero-tag { display: inline-block; font-size: 12px; font-weight: 600; color: var(--cyan); padding: 8px 18px; border-radius: 20px; background: rgba(34, 211, 238, 0.1); border: 1px solid rgba(34, 211, 238, 0.3); margin-bottom: 20px; box-shadow: 0 0 20px rgba(34, 211, 238, 0.2); letter-spacing: 1px; }
    .hero h2 { font-family: 'Outfit', sans-serif; font-size: clamp(36px, 5vw, 58px); font-weight: 800; line-height: 1.1; background: linear-gradient(135deg, #ffffff 0%, #7ba3ff 50%, #a78bfa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 16px; letter-spacing: -1px; }
    .hero p { font-size: 16px; color: var(--muted); max-width: 680px; margin: 0 auto; line-height: 1.7; }

    .card { background: var(--card); border: 1px solid var(--border); border-radius: 20px; backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px); box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.05); transition: transform 0.3s, box-shadow 0.3s, border-color 0.3s; }
    .card:hover { transform: translateY(-4px); box-shadow: 0 18px 45px rgba(0, 0, 0, 0.6), 0 0 25px rgba(79, 124, 255, 0.2); border-color: rgba(79, 124, 255, 0.4); }

    .input-panel { padding: 24px; margin-bottom: 24px; }
    .input-tabs { display: flex; gap: 8px; border-bottom: 1px solid var(--border); padding-bottom: 14px; margin-bottom: 20px; flex-wrap: wrap; }
    .input-tab { background: transparent; border: 1px solid transparent; color: var(--muted); font-size: 14px; font-weight: 600; padding: 10px 22px; border-radius: 10px; cursor: pointer; transition: all 0.25s; }
    .input-tab:hover { color: #fff; background: rgba(255,255,255,0.05); }
    .input-tab.active { background: linear-gradient(135deg, rgba(79, 124, 255, 0.2), rgba(167, 139, 250, 0.15)); color: #fff; border-color: var(--border-strong); box-shadow: 0 0 20px rgba(79, 124, 255, 0.15); }
    .tab-content { display: none; animation: fadeIn 0.4s ease; }
    .tab-content.active { display: block; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }

    .tab-hint { font-size: 13px; color: var(--muted); margin-bottom: 16px; }
    .info-box { padding: 16px 20px; margin-bottom: 20px; background: rgba(79, 124, 255, 0.06); border: 1px solid rgba(79, 124, 255, 0.15); border-left: 4px solid var(--primary); border-radius: 12px; font-size: 13px; color: var(--muted); line-height: 1.6; }
    .info-box strong { display: block; color: var(--primary-bright); margin-bottom: 6px; font-size: 14px; }

    .text-input { width: 100%; max-width: 340px; background: rgba(10, 15, 30, 0.8); border: 1px solid var(--border); border-radius: 12px; padding: 14px 18px; color: var(--text); font-size: 14px; margin-bottom: 16px; transition: all 0.3s; }
    .text-input:focus { outline: none; border-color: var(--primary); box-shadow: 0 0 15px rgba(79, 124, 255, 0.2); }

    .file-drop { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; padding: 40px 20px; border: 2px dashed var(--border-strong); border-radius: 16px; color: var(--muted); cursor: pointer; margin-bottom: 16px; transition: all 0.3s; background: rgba(10, 15, 30, 0.4); }
    .file-drop:hover { border-color: var(--primary); background: rgba(79, 124, 255, 0.08); color: #fff; box-shadow: 0 0 30px rgba(79, 124, 255, 0.1); }

    .manual-top-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
    .manual-grid { display: flex; flex-direction: column; gap: 16px; margin-bottom: 16px; padding: 20px; background: rgba(10, 15, 30, 0.6); border: 1px solid var(--border); border-radius: 16px; }
    .manual-row { display: flex; flex-direction: column; gap: 12px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }
    .manual-row:last-child { border-bottom: none; padding-bottom: 0; }
    .manual-row-header { display: flex; justify-content: space-between; font-size: 13px; color: var(--muted); }
    .manual-row-header strong { color: var(--primary-bright); }
    .manual-row-cells { display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); gap: 10px; }
    .manual-cell { display: flex; flex-direction: column; gap: 4px; }
    .manual-cell label { font-size: 10px; color: var(--dim); font-family: 'JetBrains Mono', monospace; text-align: center; }
    .manual-cell input { background: rgba(10, 15, 30, 0.9); border: 1px solid var(--border); border-radius: 8px; padding: 10px; color: var(--text); font-family: 'JetBrains Mono', monospace; font-size: 12px; text-align: center; transition: all 0.2s; }
    .manual-cell input:focus { outline: none; border-color: var(--primary); box-shadow: 0 0 10px rgba(79, 124, 255, 0.2); }
    .manual-cell.filled input { border-color: rgba(79, 124, 255, 0.6); background: rgba(79, 124, 255, 0.1); }

    .manual-actions { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }
    .manual-progress { font-size: 13px; color: var(--muted); font-family: 'JetBrains Mono', monospace; margin-top: 10px; }
    .manual-progress.ok { color: var(--primary-bright); }
    .manual-progress.warn { color: var(--gold); }

    .run-btn { display: inline-flex; align-items: center; gap: 10px; padding: 15px 32px; background: linear-gradient(135deg, #4f7cff, #a78bfa); border: none; border-radius: 14px; color: #fff; font-size: 15px; font-weight: 600; cursor: pointer; transition: all 0.3s; box-shadow: 0 8px 25px rgba(79, 124, 255, 0.4); position: relative; overflow: hidden; }
    .run-btn::after { content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: linear-gradient(transparent, rgba(255,255,255,0.3), transparent); transform: rotate(45deg); transition: 0.5s; opacity: 0; }
    .run-btn:hover::after { opacity: 1; left: 100%; }
    .run-btn:hover { transform: translateY(-3px); box-shadow: 0 12px 35px rgba(79, 124, 255, 0.7); }

    .ghost-btn { background: rgba(255,255,255,0.05); border: 1px solid var(--border-strong); color: var(--muted); font-size: 13px; font-weight: 600; padding: 12px 20px; border-radius: 12px; cursor: pointer; transition: all 0.25s; backdrop-filter: blur(10px); }
    .ghost-btn:hover { border-color: var(--primary); color: #fff; background: rgba(79, 124, 255, 0.15); box-shadow: 0 0 15px rgba(79, 124, 255, 0.2); }

    .scenario-btns { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 18px; }
    .scenario-btn { display: flex; align-items: center; gap: 10px; background: rgba(10, 15, 30, 0.6); border: 1px solid var(--border); color: var(--muted); font-size: 14px; font-weight: 500; padding: 12px 20px; border-radius: 12px; cursor: pointer; transition: all 0.25s; }
    .scenario-btn:hover { border-color: var(--border-strong); color: #fff; }
    .scenario-btn.active { background: linear-gradient(135deg, rgba(79, 124, 255, 0.3), rgba(167, 139, 250, 0.2)); border-color: var(--border-strong); color: #fff; box-shadow: 0 0 20px rgba(79, 124, 255, 0.2); }
    .dot { width: 10px; height: 10px; border-radius: 50%; }
    .dot-green { background: var(--primary-bright); box-shadow: 0 0 10px var(--primary-bright); }
    .dot-amber { background: var(--gold); box-shadow: 0 0 10px var(--gold); }
    .dot-red { background: var(--critical); box-shadow: 0 0 10px var(--critical); }

    .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 20px; margin-bottom: 24px; }
    .kpi { padding: 24px; position: relative; overflow: hidden; background: linear-gradient(180deg, rgba(20, 28, 55, 0.7) 0%, rgba(14, 20, 40, 0.5) 100%); }
    .kpi::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, var(--primary), var(--violet), var(--cyan)); opacity: 0.8; }
    .kpi-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; gap: 8px; }
    .kpi-label { font-size: 11px; color: var(--muted); letter-spacing: 1.5px; text-transform: uppercase; font-weight: 600; display: inline-flex; align-items: center; gap: 8px; }
    .info-btn { background: transparent; border: none; color: var(--dim); cursor: pointer; font-size: 16px; transition: color 0.2s; }
    .info-btn:hover { color: var(--cyan); }
    .kpi-main { display: flex; align-items: baseline; gap: 10px; }
    .kpi-value { font-family: 'Outfit', sans-serif; font-size: 42px; font-weight: 800; letter-spacing: -2px; color: #fff; text-shadow: 0 0 20px rgba(255,255,255,0.1); }
    .kpi-value.mono { font-family: 'JetBrains Mono', monospace; font-size: 26px; letter-spacing: 0; color: var(--cyan); text-shadow: 0 0 15px rgba(34, 211, 238, 0.3); }
    .kpi-value.risk { font-size: 26px; letter-spacing: 2px; text-transform: uppercase; }
    .kpi-value.risk.low { color: var(--primary-bright); text-shadow: 0 0 20px rgba(123, 163, 255, 0.4); }
    .kpi-value.risk.medium { color: var(--gold); text-shadow: 0 0 20px rgba(251, 191, 36, 0.4); }
    .kpi-value.risk.high { color: var(--danger); text-shadow: 0 0 20px rgba(244, 63, 94, 0.4); }
    .kpi-value.risk.critical { color: var(--critical); animation: pulse 1.6s infinite; text-shadow: 0 0 20px rgba(239, 68, 68, 0.6); }
    .kpi-unit { font-size: 13px; color: var(--dim); }
    @keyframes pulse { 0%,100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.7; transform: scale(0.98); } }

    .grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 20px; margin-bottom: 24px; }
    .chart-card { grid-column: span 4; padding: 24px; }
    .chart-card.wide { grid-column: span 8; }
    .alerts-card { grid-column: span 12; padding: 24px; }
    .card-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
    .card-head h3 { font-family: 'Outfit', sans-serif; font-size: 16px; font-weight: 600; color: #fff; letter-spacing: 0.5px; }
    .badge { font-family: 'JetBrains Mono', monospace; font-size: 11px; padding: 6px 12px; border-radius: 8px; background: rgba(79, 124, 255, 0.15); color: var(--primary-bright); border: 1px solid rgba(79, 124, 255, 0.4); box-shadow: 0 0 15px rgba(79, 124, 255, 0.2); }
    .badge-gold { background: rgba(251, 191, 36, 0.15); color: var(--gold); border-color: rgba(251, 191, 36, 0.4); box-shadow: 0 0 15px rgba(251, 191, 36, 0.2); }
    .chart-wrap { position: relative; height: 300px; }

    .alert-list { list-style: none; display: flex; flex-direction: column; gap: 10px; }
    .alert-placeholder { color: var(--dim); font-size: 14px; padding: 20px; text-align: center; border: 1px dashed var(--border-strong); border-radius: 12px; }
    .alert-item { display: flex; align-items: center; gap: 16px; padding: 16px; background: rgba(10, 15, 30, 0.6); border: 1px solid var(--border); border-radius: 12px; border-left: 4px solid var(--primary); font-size: 14px; transition: background 0.2s; }
    .alert-item:hover { background: rgba(20, 28, 55, 0.8); }
    .alert-item.critical { border-left-color: var(--critical); }
    .alert-item.high { border-left-color: var(--danger); }
    .alert-item.medium { border-left-color: var(--gold); }
    .alert-sensor { font-family: 'JetBrains Mono', monospace; font-weight: 600; min-width: 100px; color: #fff; }
    .alert-desc { color: var(--muted); flex: 1; }
    .alert-impact { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--primary-bright); font-weight: 600; }

    .assistant-card { padding: 24px; margin-bottom: 24px; }
    .assistant-head-left { display: flex; align-items: center; gap: 12px; }
    .assistant-avatar { width: 40px; height: 40px; border-radius: 12px; background: linear-gradient(135deg, rgba(79, 124, 255, 0.4), rgba(167, 139, 250, 0.3)); border: 1px solid var(--border-strong); display: grid; place-items: center; color: var(--cyan); box-shadow: 0 0 20px rgba(79, 124, 255, 0.3); }
    .assistant-log { display: flex; flex-direction: column; gap: 14px; max-height: 400px; overflow-y: auto; margin-bottom: 16px; padding-right: 8px; }
    .assistant-log::-webkit-scrollbar { width: 6px; }
    .assistant-log::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 10px; }
    .assistant-msg { padding: 14px 18px; border-radius: 14px; font-size: 14px; line-height: 1.7; max-width: 85%; white-space: pre-wrap; animation: fadeIn 0.3s ease; }
    .assistant-msg.bot { background: rgba(79, 124, 255, 0.1); border: 1px solid rgba(79, 124, 255, 0.2); align-self: flex-start; border-bottom-left-radius: 4px; }
    .assistant-msg.user { background: rgba(167, 139, 250, 0.15); border: 1px solid rgba(167, 139, 250, 0.3); align-self: flex-end; border-bottom-right-radius: 4px; }

    .quick-prompts { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 16px; }
    .quick-prompt { font-size: 13px; padding: 8px 16px; background: rgba(255, 255, 255, 0.03); border: 1px solid var(--border); border-radius: 20px; color: var(--muted); cursor: pointer; transition: all 0.2s; }
    .quick-prompt:hover { border-color: var(--cyan); color: var(--cyan); background: rgba(34, 211, 238, 0.05); box-shadow: 0 0 15px rgba(34, 211, 238, 0.15); }
    .assistant-input-row { display: flex; gap: 12px; }
    .assistant-input-row .text-input { margin-bottom: 0; flex: 1; max-width: none; border-radius: 14px; }

    .report-row { display: flex; justify-content: center; margin-top: 12px; margin-bottom: 40px; }
    .report-btn { display: flex; align-items: center; gap: 12px; padding: 16px 36px; background: rgba(15, 20, 40, 0.8); border: 1px solid var(--border-strong); border-radius: 14px; color: #fff; font-size: 15px; font-weight: 600; cursor: pointer; transition: all 0.3s; backdrop-filter: blur(10px); }
    .report-btn:hover { background: rgba(79, 124, 255, 0.15); border-color: var(--primary); box-shadow: 0 0 30px rgba(79, 124, 255, 0.3); transform: translateY(-2px); }

    .footer { text-align: center; margin-top: 60px; padding-top: 30px; border-top: 1px solid var(--border); color: var(--dim); font-size: 13px; letter-spacing: 1px; }

    .toast-container { position: fixed; top: 24px; right: 24px; display: flex; flex-direction: column; gap: 12px; z-index: 9999; }
    .toast { padding: 14px 22px; background: rgba(15, 20, 40, 0.95); border: 1px solid var(--border-strong); border-left: 4px solid var(--primary); border-radius: 12px; font-size: 14px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); backdrop-filter: blur(10px); opacity: 0; transform: translateX(30px); transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275); }
    .toast.show { opacity: 1; transform: translateX(0); }

    .modal-backdrop { position: fixed; inset: 0; background: rgba(0, 0, 0, 0.8); backdrop-filter: blur(8px); display: none; place-items: center; z-index: 10000; padding: 20px; opacity: 0; transition: opacity 0.3s; }
    .modal-backdrop.open { display: grid; opacity: 1; }
    .modal-card { background: linear-gradient(180deg, rgba(25, 32, 60, 0.98) 0%, rgba(15, 20, 40, 0.98) 100%); border: 1px solid var(--border-strong); border-radius: 24px; padding: 36px; max-width: 540px; width: 100%; position: relative; box-shadow: 0 20px 60px rgba(0,0,0,0.6), 0 0 40px rgba(79, 124, 255, 0.2); transform: translateY(20px); transition: transform 0.3s; }
    .modal-backdrop.open .modal-card { transform: translateY(0); }
    .modal-close { position: absolute; top: 18px; right: 20px; background: rgba(255,255,255,0.05); border: none; color: var(--muted); width: 32px; height: 32px; border-radius: 50%; font-size: 20px; display: grid; place-items: center; cursor: pointer; transition: all 0.2s; }
    .modal-close:hover { background: rgba(244, 63, 94, 0.2); color: var(--danger); }

    html[dir="rtl"] body { font-family: 'Tajawal', 'Inter', sans-serif; }
    html[dir="rtl"] .assistant-msg.bot { align-self: flex-end; border-bottom-left-radius: 14px; border-bottom-right-radius: 4px; }
    html[dir="rtl"] .assistant-msg.user { align-self: flex-start; border-bottom-right-radius: 14px; border-bottom-left-radius: 4px; }
    html[dir="rtl"] .alert-item { border-left: none; border-right: 4px solid var(--primary); }

    @media (max-width: 980px) { .chart-card, .chart-card.wide, .alerts-card { grid-column: span 12; } }
    @media (max-width: 640px) { .manual-row-cells { grid-template-columns: repeat(3, 1fr); } .hero h2 { font-size: 32px; } .kpi-value { font-size: 32px; } }
  </style>
</head>
<body>
  <canvas id="starsCanvas"></canvas>
  <div class="aurora"></div>
  <div class="grid-overlay"></div>
  <div class="toast-container" id="toastContainer"></div>

  <div class="modal-backdrop" id="infoModal">
    <div class="modal-card">
      <button class="modal-close" id="modalClose">×</button>
      <h3 id="modalTitle" style="margin-bottom:16px; color:var(--cyan); font-size:20px;"></h3>
      <div id="modalBody" style="font-size:15px; line-height:1.8; color:var(--muted);"></div>
    </div>
  </div>

  <header class="topbar">
    <div class="brand">
      <div class="logo-mark">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2 L20 7 V13 C20 17 16 21 12 22 C8 21 4 17 4 13 V7 Z"/><path d="M9 12 L11 14 L15 10"/></svg>
      </div>
      <div class="brand-text"><h1 data-i18n="brand">AEGIS</h1></div>
    </div>
    <div class="lang-toggle">
      <button class="lang-btn active" data-lang="en">EN</button>
      <button class="lang-btn" data-lang="ar">AR</button>
    </div>
  </header>

  <main>
    <section class="hero">
      <div class="hero-tag" data-i18n="heroTag">Industrial Predictive Analytics</div>
      <h2 data-i18n="heroTitle">Predict Failure.<br/>Extend Equipment Life.</h2>
      <p data-i18n="heroSub">Real-time Remaining Useful Life estimation with explainable AI diagnostics and an embedded engineering assistant.</p>
    </section>

    <section class="card input-panel">
      <div class="input-tabs">
        <button class="input-tab active" data-mode="demo" data-i18n="tabDemo">Demo</button>
        <button class="input-tab" data-mode="upload" data-i18n="tabUpload">Upload Data</button>
        <button class="input-tab" data-mode="manual" data-i18n="tabManual">Manual Entry</button>
      </div>
      <div class="tab-content active" id="tab-demo">
        <p class="tab-hint" data-i18n="demoHint">Try the system with real engine samples at different degradation stages.</p>
        <div class="scenario-btns">
          <button class="scenario-btn active" data-scenario="healthy"><span class="dot dot-green"></span><span data-i18n="scHealthy">Healthy</span></button>
          <button class="scenario-btn" data-scenario="degrading"><span class="dot dot-amber"></span><span data-i18n="scDegrading">Degrading</span></button>
          <button class="scenario-btn" data-scenario="critical"><span class="dot dot-red"></span><span data-i18n="scCritical">Critical</span></button>
        </div>
        <button class="run-btn" id="demoBtn"><span data-i18n="runDemo">Run Demo</span></button>
      </div>
      <div class="tab-content" id="tab-upload">
        <div class="info-box">
          <strong data-i18n="uploadBestTitle">✅ Recommended for real use</strong>
          <p data-i18n="uploadBestText">Upload a CSV or Excel file exported from your monitoring system. Columns like unit_id, cycle, sensor_2, sensor_3, ... are auto-detected.</p>
        </div>
        <input type="text" id="uploadUnitId" class="text-input" placeholder="Unit ID (optional)" />
        <div class="file-drop" id="fileDrop">
          <input type="file" id="fileInput" accept=".csv,.xlsx,.xls" hidden />
          <span data-i18n="dropFile">Click or drop a .csv / .xlsx file</span>
          <small id="fileName" style="color:var(--cyan); font-weight:bold;"></small>
        </div>
        <button class="run-btn" id="uploadBtn"><span data-i18n="runUpload">Analyze File</span></button>
      </div>
      <div class="tab-content" id="tab-manual">
        <div class="info-box">
          <strong data-i18n="manualBestTitle">💡 Quick Entry (Single Reading)</strong>
          <p data-i18n="manualBestText">Enter the latest sensor readings for your machine. You don't need to calculate every second — just the last reading you have. Click "Simulate Data" for an instant test.</p>
        </div>
        <div class="manual-top-row">
          <input type="text" id="manualUnitId" class="text-input" placeholder="Machine ID (e.g. PUMP-07)" />
          <button class="ghost-btn" id="fillExampleBtn" data-i18n="fillExample">🎲 Simulate Data</button>
          <button class="ghost-btn" id="clearManualBtn" data-i18n="clearAll">🗑️ Clear</button>
        </div>
        <div class="manual-grid" id="manualGrid"></div>
        <div class="manual-actions">
          <button class="ghost-btn" id="addRowBtn" data-i18n="addRow">+ Add Second Reading</button>
          <button class="run-btn" id="manualBtn"><span data-i18n="runManual">Analyze Reading</span></button>
        </div>
        <div class="manual-progress" id="manualProgress"></div>
      </div>
      <div class="panel-status" id="panelStatus" style="margin-top:16px; font-size:13px; color:var(--muted);"></div>
    </section>

    <section class="kpis">
      <div class="card kpi">
        <div class="kpi-head">
          <span class="kpi-label"><span data-i18n="kpiRul">Remaining Useful Life</span><button class="info-btn" data-info="rul">ⓘ</button></span>
        </div>
        <div class="kpi-main"><span class="kpi-value" id="rulValue">—</span><span class="kpi-unit" data-i18n="cycles">cycles</span></div>
      </div>
      <div class="card kpi">
        <div class="kpi-head">
          <span class="kpi-label"><span data-i18n="kpiFail">Failure Probability</span><button class="info-btn" data-info="prob">ⓘ</button></span>
        </div>
        <div class="kpi-main"><span class="kpi-value" id="probValue">—</span><span class="kpi-unit">%</span></div>
      </div>
      <div class="card kpi">
        <div class="kpi-head"><span class="kpi-label"><span data-i18n="kpiRisk">Risk Level</span></span></div>
        <div class="kpi-main"><span class="kpi-value risk" id="riskValue">—</span></div>
      </div>
      <div class="card kpi">
        <div class="kpi-head"><span class="kpi-label"><span data-i18n="kpiUnit">Machine</span></span></div>
        <div class="kpi-main"><span class="kpi-value mono" id="unitValue">—</span></div>
      </div>
    </section>

    <section class="grid">
      <div class="card chart-card wide">
        <div class="card-head"><h3 data-i18n="trendTitle">Sensor Trend (Normalized)</h3><span class="badge" data-i18n="live">LIVE</span></div>
        <div class="chart-wrap"><canvas id="trendChart"></canvas></div>
      </div>
      <div class="card chart-card">
        <div class="card-head"><h3 data-i18n="shapTitle">Root-Cause Sensors</h3><span class="badge badge-gold">SHAP</span></div>
        <div class="chart-wrap"><canvas id="shapChart"></canvas></div>
      </div>
      <div class="card alerts-card">
        <div class="card-head"><h3 data-i18n="alertsTitle">Alert Feed</h3></div>
        <ul class="alert-list" id="alertList"><li class="alert-placeholder" data-i18n="noAlerts">No active alerts.</li></ul>
      </div>
    </section>

    <section class="assistant-card card">
      <div class="card-head">
        <div class="assistant-head-left">
          <div class="assistant-avatar"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8V4H8"/><rect x="4" y="8" width="16" height="12" rx="2"/><path d="M2 14h2M20 14h2M15 13v2M9 13v2"/></svg></div>
          <h3 data-i18n="assistantTitle">AI Engineering Assistant</h3>
        </div>
        <span class="badge badge-gold" data-i18n="assistantBadge">CONTEXT-AWARE</span>
      </div>
      <div class="assistant-log" id="assistantLog"><div class="assistant-msg bot" data-i18n="assistantGreeting">Hello. Ask me about risk, RUL, or sensors.</div></div>
      <div class="quick-prompts" id="quickPrompts"></div>
      <div class="assistant-input-row">
        <input type="text" id="assistantInput" class="text-input" placeholder="Ask about the current analysis..." />
        <button class="run-btn small" id="assistantSend"><span data-i18n="send">Send</span></button>
      </div>
    </section>

    <section class="report-row">
      <button class="report-btn" id="reportBtn"><span data-i18n="genReport">Generate PDF Report</span></button>
    </section>
  </main>

  <footer class="footer">Created by SHODO <span style="color: #f43f5e; font-size:16px;">❤️</span></footer>

  <script>
    // كود النجوم المتحركة في الخلفية
    const canvas = document.getElementById('starsCanvas');
    const ctx = canvas.getContext('2d');
    let width, height, stars = [];

    function initCanvas() {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
      stars = [];
      for (let i = 0; i < 150; i++) {
        stars.push({
          x: Math.random() * width,
          y: Math.random() * height,
          radius: Math.random() * 1.5 + 0.5,
          speedY: Math.random() * 0.5 + 0.1,
          alpha: Math.random(),
          pulse: Math.random() * 0.05
        });
      }
    }

    function drawStars() {
      ctx.clearRect(0, 0, width, height);
      stars.forEach(star => {
        star.y -= star.speedY;
        if (star.y < 0) star.y = height;
        star.alpha += star.pulse;
        if (star.alpha > 1 || star.alpha < 0.2) star.pulse *= -1;

        ctx.beginPath();
        ctx.arc(star.x, star.y, star.radius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 255, 255, ${star.alpha})`;
        ctx.shadowBlur = 10;
        ctx.shadowColor = '#7ba3ff';
        ctx.fill();
      });
      requestAnimationFrame(drawStars);
    }

    window.addEventListener('resize', initCanvas);
    initCanvas();
    drawStars();

    // المنطق الأساسي للتطبيق
    const I18N = {
      en: {
        brand: "AEGIS", heroTag: "Industrial Predictive Analytics", heroTitle: "Predict Failure.<br/>Extend Equipment Life.", heroSub: "Real-time Remaining Useful Life estimation with explainable AI diagnostics and an embedded engineering assistant.",
        tabDemo: "Demo", tabUpload: "Upload Data", tabManual: "Manual Entry", demoHint: "Try the system with real engine samples at different degradation stages.",
        uploadBestTitle: "✅ Recommended for real use", uploadBestText: "Upload a CSV or Excel file exported from your monitoring system.",
        manualBestTitle: "💡 Quick Entry (Single Reading)", manualBestText: "Enter the latest sensor readings for your machine. Click 'Simulate Data' for an instant test.",
        dropFile: "Click or drop a .csv / .xlsx file", fillExample: "🎲 Simulate Data", clearAll: "🗑️ Clear",
        scHealthy: "Healthy", scDegrading: "Degrading", scCritical: "Critical", runDemo: "Run Demo", runUpload: "Analyze File", runManual: "Analyze Reading", addRow: "+ Add Second Reading",
        kpiRul: "Remaining Useful Life", kpiFail: "Failure Probability", kpiRisk: "Risk Level", kpiUnit: "Machine", cycles: "cycles",
        trendTitle: "Sensor Trend (Normalized)", live: "LIVE", shapTitle: "Root-Cause Sensors", alertsTitle: "Alert Feed", noAlerts: "No active alerts.",
        assistantTitle: "AI Engineering Assistant", assistantBadge: "CONTEXT-AWARE", assistantGreeting: "Hello. Ask me about risk, RUL, or sensors.", send: "Send", genReport: "Generate PDF Report",
        riskLow: "Low", riskMedium: "Medium", riskHigh: "High", riskCritical: "Critical", alertHigh: "elevated risk contribution", alertRul: "remaining before failure",
        reportGenerating: "Generating report...", errorRun: "Analysis failed.", processing: "Processing...", uploadSuccess: "File analyzed successfully", manualSuccess: "Manual data analyzed", needTwoRows: "Please add at least 1 reading.", demoLoaded: "Demo scenario loaded.", uploadError: "Upload failed.", needSensors: "Fill at least 3 sensors.", filledStatus: (n, t) => `${n}/${t} sensors filled`, cycleLabel: "Reading",
        qRisk: "What is the risk level?", qRul: "How much RUL remains?", qSensor: "Which sensor contributes most?", qMaint: "What maintenance do you recommend?", qUrgent: "Is it urgent?", qSummary: "Give me a full summary.",
        rulInfo: "<strong>RUL</strong><br><br>A statistical estimate of how many operating cycles remain before likely failure.", probInfo: "<strong>Failure Probability</strong><br><br>The model's confidence that this machine will fail within the next 30 operating cycles."
      },
      ar: {
        brand: "إيجيس", heroTag: "منصة تحليل تنبؤي للآلات الصناعية", heroTitle: "تنبّأ بالفشل.<br/>أطِل عمر الآلة.", heroSub: "تقدير فوري للعمر الافتراضي المتبقي للآلة مع تشخيص قابل للتفسير ومساعد هندسي مدمج.",
        tabDemo: "عرض تجريبي", tabUpload: "رفع ملف", tabManual: "إدخال يدوي", demoHint: "جرّب النظام على عيّنات حقيقية في مراحل تدهور مختلفة.",
        uploadBestTitle: "✅ الطريقة الموصى بها للاستخدام الفعلي", uploadBestText: "ارفع ملف CSV أو Excel مُصدَّر من نظام المراقبة لديك.",
        manualBestTitle: "💡 إدخال سريع (قراءة واحدة)", manualBestText: "أدخل قراءات الحساسات الأخيرة للآلة. اضغط 'محاكاة بيانات' لتجربة فورية.",
        dropFile: "انقر أو أفلت ملف .csv / .xlsx هنا", fillExample: "🎲 محاكاة بيانات", clearAll: "🗑️ مسح",
        scHealthy: "سليمة", scDegrading: "متدهورة", scCritical: "حرجة", runDemo: "تشغيل العرض", runUpload: "تحليل الملف", runManual: "تحليل القراءة", addRow: "+ إضافة قراءة ثانية",
        kpiRul: "العمر الافتراضي المتبقي", kpiFail: "احتمالية الفشل", kpiRisk: "مستوى الخطورة", kpiUnit: "الآلة", cycles: "دورة",
        trendTitle: "اتجاه قراءات الحساسات (مُطبّع)", live: "مباشر", shapTitle: "الحساسات المسببة للخطر", alertsTitle: "سجل التنبيهات", noAlerts: "لا توجد تنبيهات نشطة.",
        assistantTitle: "المساعد الهندسي الذكي", assistantBadge: "يفهم سياق التحليل", assistantGreeting: "أهلاً بك. يمكنك سؤالي عن مستوى الخطورة، العمر المتبقي، أو الحساسات المسببة.", send: "إرسال", genReport: "توليد التقرير الفني (PDF)",
        riskLow: "منخفض", riskMedium: "متوسط", riskHigh: "مرتفع", riskCritical: "حرج", alertHigh: "مساهمة مرتفعة في الخطورة", alertRul: "دورة قبل الفشل المتوقع",
        reportGenerating: "جارٍ إعداد التقرير...", errorRun: "تعذّر إتمام التحليل.", processing: "جارٍ المعالجة...", uploadSuccess: "تم تحليل الملف بنجاح", manualSuccess: "تم تحليل البيانات اليدوية", needTwoRows: "يرجى إضافة قراءة واحدة على الأقل.", demoLoaded: "تم تحميل السيناريو التجريبي.", uploadError: "تعذّر رفع الملف.", needSensors: "يُنصح بملء 3 حساسات على الأقل.", filledStatus: (n, t) => `تم ملء ${n} من ${t} حساساً`, cycleLabel: "القراءة",
        qRisk: "ما مستوى الخطورة؟", qRul: "كم العمر المتبقي؟", qSensor: "أي حساس هو الأكثر تأثيراً؟", qMaint: "ما توصيتك للصيانة؟", qUrgent: "هل الوضع عاجل؟", qSummary: "أعطني ملخصاً شاملاً.",
        rulInfo: "<strong>العمر الافتراضي المتبقي (RUL)</strong><br><br>تقدير إحصائي لعدد الدورات التشغيلية المتبقية قبل الفشل المحتمل.", probInfo: "<strong>احتمالية الفشل</strong><br><br>مدى ثقة النموذج في أن هذه الآلة ستفشل خلال الـ 30 دورة تشغيلية القادمة."
      }
    };
    const ESSENTIAL_SENSORS = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_11"];
    const DEMO_VALUES = { sensor_2: 642.15, sensor_3: 1587.5, sensor_4: 1402.3, sensor_7: 553.7, sensor_8: 2388.1, sensor_11: 47.5 };
    let currentLang = "en", currentScenario = "healthy", currentData = null, trendChart = null, shapChart = null, selectedFile = null, manualRowCount = 1;
    const $ = (id) => document.getElementById(id);

    function showToast(msg) {
      const c = $("toastContainer"), t = document.createElement("div"); t.className = "toast"; t.textContent = msg; c.appendChild(t);
      setTimeout(() => t.classList.add("show"), 10); setTimeout(() => { t.classList.remove("show"); setTimeout(() => t.remove(), 300); }, 3200);
    }
    function animateValue(el, end, dec = 1) {
      const start = parseFloat(el.textContent) || 0; if (start === end || isNaN(end)) { el.textContent = end.toFixed(dec); return; }
      const st = performance.now();
      function step(now) { const p = Math.min((now - st) / 900, 1); const e = 1 - Math.pow(1 - p, 3); el.textContent = (start + (end - start) * e).toFixed(dec); if (p < 1) requestAnimationFrame(step); }
      requestAnimationFrame(step);
    }
    function applyLang(lang) {
      currentLang = lang; const d = I18N[lang]; document.documentElement.lang = lang; document.documentElement.dir = lang === "ar" ? "rtl" : "ltr";
      document.querySelectorAll("[data-i18n]").forEach(el => { const k = el.getAttribute("data-i18n"); if (d[k]) el.innerHTML = d[k]; });
      document.querySelectorAll(".lang-btn").forEach(b => b.classList.toggle("active", b.dataset.lang === lang));
      renderQuickPrompts(); buildManualGrid(); if (currentData) renderPrediction(currentData.prediction, currentData.sensor_history);
    }
    function riskLabel(r) { return I18N[currentLang]["risk" + r.charAt(0).toUpperCase() + r.slice(1)] || r; }
    function setStatus(m) { $("panelStatus").textContent = m; }
    function renderQuickPrompts() {
      const d = I18N[currentLang], c = $("quickPrompts"); c.innerHTML = "";
      [d.qRisk, d.qRul, d.qSensor, d.qMaint, d.qUrgent, d.qSummary].forEach(p => {
        const b = document.createElement("button"); b.className = "quick-prompt"; b.textContent = p;
        b.onclick = () => { $("assistantInput").value = p; sendAssistant(); }; c.appendChild(b);
      });
    }
    function showInfoModal(k) { $("modalTitle").textContent = I18N[currentLang][k === "rul" ? "kpiRul" : "kpiFail"]; $("modalBody").innerHTML = I18N[currentLang][k === "rul" ? "rulInfo" : "probInfo"]; $("infoModal").classList.add("open"); }
    function buildManualGrid() {
      const g = $("manualGrid"), d = I18N[currentLang]; g.innerHTML = "";
      for (let r = 0; r < manualRowCount; r++) {
        const row = document.createElement("div"); row.className = "manual-row";
        row.innerHTML = `<div class="manual-row-header"><strong>${d.cycleLabel} ${r+1}</strong><span class="row-fill-count" data-row="${r}">0/${ESSENTIAL_SENSORS.length}</span></div><div class="manual-row-cells" id="cells-${r}"></div>`;
        g.appendChild(row); const cells = row.querySelector(`#cells-${r}`);
        ESSENTIAL_SENSORS.forEach(s => {
          const c = document.createElement("div"); c.className = "manual-cell";
          c.innerHTML = `<label>S${s.replace("sensor_","")}</label><input type="number" step="any" data-row="${r}" data-sensor="${s}" />`;
          cells.appendChild(c);
        });
      }
      g.querySelectorAll("input").forEach(inp => inp.oninput = () => updateRowProgress(parseInt(inp.dataset.row)));
    }
    function updateRowProgress(r) {
      let f = 0; ESSENTIAL_SENSORS.forEach(s => { const inp = document.querySelector(`input[data-row="${r}"][data-sensor="${s}"]`); if (inp) { if (inp.value.trim() !== "") { f++; inp.parentElement.classList.add("filled"); } else inp.parentElement.classList.remove("filled"); } });
      const b = document.querySelector(`.row-fill-count[data-row="${r}"]`); if (b) { b.textContent = I18N[currentLang].filledStatus(f, ESSENTIAL_SENSORS.length); b.style.color = f >= 3 ? "var(--primary-bright)" : "var(--gold)"; }
      updateOverallProgress();
    }
    function updateOverallProgress() {
      let tf = 0, rwe = 0; for (let r = 0; r < manualRowCount; r++) { let rf = 0; ESSENTIAL_SENSORS.forEach(s => { const inp = document.querySelector(`input[data-row="${r}"][data-sensor="${s}"]`); if (inp && inp.value.trim() !== "") rf++; }); tf += rf; if (rf >= 3) rwe++; }
      const el = $("manualProgress"); if (rwe < manualRowCount) { el.textContent = I18N[currentLang].needSensors; el.className = "manual-progress warn"; } else { el.textContent = "✅ " + I18N[currentLang].filledStatus(tf, manualRowCount * ESSENTIAL_SENSORS.length); el.className = "manual-progress ok"; }
    }
    function fillExample() {
      for (let r = 0; r < manualRowCount; r++) { ESSENTIAL_SENSORS.forEach(s => { const inp = document.querySelector(`input[data-row="${r}"][data-sensor="${s}"]`); if (inp) inp.value = (DEMO_VALUES[s] + DEMO_VALUES[s] * (r * 0.008 + (Math.random() - 0.5) * 0.005)).toFixed(4); }); updateRowProgress(r); }
      if (!$("manualUnitId").value) $("manualUnitId").value = "DEMO-UNIT-01"; showToast(currentLang === "ar" ? "تم ملء البيانات" : "Example filled");
    }
    function clearManual() { document.querySelectorAll("#manualGrid input").forEach(i => i.value = ""); for (let r = 0; r < manualRowCount; r++) updateRowProgress(r); }
    function collectManualReadings() { const rows = []; for (let r = 0; r < manualRowCount; r++) { const rec = { cycle: r + 1 }; let h = false; ESSENTIAL_SENSORS.forEach(s => { const inp = document.querySelector(`input[data-row="${r}"][data-sensor="${s}"]`); if (inp && inp.value.trim() !== "") { rec[s] = parseFloat(inp.value); h = true; } }); if (h) rows.push(rec); } return rows; }

    async function runDemo() {
      setStatus(I18N[currentLang].processing); $("demoBtn").disabled = true;
      try {
        const res = await fetch("/api/demo", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ scenario: currentScenario }) });
        if (!res.ok) throw new Error(); const data = await res.json(); currentData = data; renderPrediction(data.prediction, data.sensor_history); setStatus(I18N[currentLang].demoLoaded); showToast(I18N[currentLang].demoLoaded);
      } catch (_) { setStatus(I18N[currentLang].errorRun); showToast(I18N[currentLang].errorRun); } finally { $("demoBtn").disabled = false; }
    }
    async function uploadFile() {
      if (!selectedFile) return; setStatus(I18N[currentLang].processing); $("uploadBtn").disabled = true;
      const fd = new FormData(); fd.append("file", selectedFile); fd.append("unit_id", $("uploadUnitId").value || "UPLOADED-UNIT");
      try {
        const res = await fetch("/api/upload", { method: "POST", body: fd }); const data = await res.json();
        if (!res.ok) throw new Error(data.error); currentData = data; renderPrediction(data.prediction, data.sensor_history); setStatus(I18N[currentLang].uploadSuccess); showToast(I18N[currentLang].uploadSuccess);
      } catch (e) { setStatus(e.message || I18N[currentLang].uploadError); showToast(e.message || I18N[currentLang].uploadError); } finally { $("uploadBtn").disabled = false; }
    }
    async function runManual() {
      const readings = collectManualReadings(); if (readings.length < 1) return; setStatus(I18N[currentLang].processing); $("manualBtn").disabled = true;
      try {
        const res = await fetch("/api/manual", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ unit_id: $("manualUnitId").value || "MANUAL-UNIT", readings }) });
        const data = await res.json(); if (!res.ok) throw new Error(data.error); currentData = data; renderPrediction(data.prediction, data.sensor_history); setStatus(I18N[currentLang].manualSuccess); showToast(I18N[currentLang].manualSuccess);
      } catch (e) { setStatus(e.message || I18N[currentLang].errorRun); showToast(e.message || I18N[currentLang].errorRun); } finally { $("manualBtn").disabled = false; }
    }
    function renderPrediction(pred, history) {
      animateValue($("rulValue"), pred.rul, 1); animateValue($("probValue"), pred.failure_probability * 100, 2);
      const r = $("riskValue"); r.textContent = riskLabel(pred.risk_level); r.className = "kpi-value risk " + pred.risk_level;
      $("unitValue").textContent = pred.unit_id; renderTrendChart(history); renderShapChart(pred.top_sensors); renderAlerts(pred);
    }
    function renderTrendChart(history) {
      if (!history || history.length === 0) return;
      const processed = history.map((h, i) => ({ ...h, cycle: h.cycle !== undefined ? h.cycle : i + 1 }));
      if (processed.length === 1) { const d = { ...processed[0] }; d.cycle = (processed[0].cycle || 1) + 1; processed.push(d); }
      const available = Object.keys(processed[0]).filter(k => k.startsWith("sensor_") && processed.some(h => h[k] != null)).slice(0, 5);
      if (available.length === 0) return;
      const palette = ["#4f7cff", "#a78bfa", "#22d3ee", "#fbbf24", "#f43f5e"];
      const labels = processed.map(h => h.cycle);
      const datasets = available.map((s, i) => {
        const raw = processed.map(h => h[s]); const valid = raw.filter(v => v != null); const min = Math.min(...valid), max = Math.max(...valid), rng = max - min || 1;
        return { label: s.replace("_", " ").toUpperCase(), data: raw.map(v => v != null ? (v - min) / rng : null), borderColor: palette[i % palette.length], backgroundColor: palette[i % palette.length] + "18", borderWidth: 2, tension: 0.35, pointRadius: 0, fill: false, shadowBlur: 10, shadowColor: palette[i % palette.length] };
      });
      if (trendChart) trendChart.destroy();
      trendChart = new Chart($("trendChart").getContext("2d"), { type: "line", data: { labels, datasets }, options: { responsive: true, maintainAspectRatio: false, interaction: { intersect: false, mode: "index" }, plugins: { legend: { labels: { color: "#8896b8", boxWidth: 8, usePointStyle: true, font: {family: 'Inter', size: 12} } } }, scales: { x: { grid: { color: "rgba(99,130,246,0.05)" }, ticks: { color: "#4f5a7a" } }, y: { grid: { color: "rgba(99,130,246,0.05)" }, ticks: { color: "#4f5a7a" } } } } });
    }
    function renderShapChart(top) {
      const labels = top.map(s => s.sensor.replace("_", " ").toUpperCase()), vals = top.map(s => s.impact), p = ["#f43f5e", "#fbbf24", "#a78bfa", "#4f7cff", "#22d3ee"];
      if (shapChart) shapChart.destroy();
      shapChart = new Chart($("shapChart").getContext("2d"), { type: "bar", data: { labels, datasets: [{ data: vals, backgroundColor: p.map(c => c + "cc"), borderColor: p, borderWidth: 1, borderRadius: 6 }] }, options: { indexAxis: "y", responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { grid: { color: "rgba(99,130,246,0.05)" }, ticks: { color: "#4f5a7a" } }, y: { grid: { display: false }, ticks: { color: "#e8ecf8", font: { family: "JetBrains Mono" } } } } } });
    }
    function renderAlerts(pred) {
      const l = $("alertList"); l.innerHTML = "";
      pred.top_sensors.slice(0, 3).forEach(s => { l.innerHTML += `<li class="alert-item ${pred.risk_level}"><span class="alert-sensor">${s.sensor.replace("_", " ").toUpperCase()}</span><span class="alert-desc">${I18N[currentLang].alertHigh}</span><span class="alert-impact">${s.impact.toFixed(3)}</span></li>`; });
      l.innerHTML += `<li class="alert-item ${pred.risk_level}"><span class="alert-sensor">${pred.rul.toFixed(1)}</span><span class="alert-desc">${I18N[currentLang].alertRul}</span><span class="alert-impact">${(pred.failure_probability * 100).toFixed(1)}%</span></li>`;
    }
    function appendMsg(txt, r) { const d = document.createElement("div"); d.className = "assistant-msg " + r; d.innerHTML = txt; $("assistantLog").appendChild(d); $("assistantLog").scrollTop = $("assistantLog").scrollHeight; }
    async function sendAssistant() {
      const q = $("assistantInput").value.trim(); if (!q) return; appendMsg(q, "user"); $("assistantInput").value = "";
      try { const res = await fetch("/api/assistant", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: q, prediction: currentData?.prediction || {}, lang: currentLang }) }); const d = await res.json(); appendMsg(d.answer, "bot"); } catch (_) { appendMsg("—", "bot"); }
    }
    async function generateReport() {
      if (!currentData) return; const b = $("reportBtn"); b.disabled = true;
      try { const res = await fetch("/api/report", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prediction: currentData.prediction }) }); const blob = await res.blob(); const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `AEGIS_Report_${currentData.prediction.unit_id}.pdf`; a.click(); } catch (_) {} finally { b.disabled = false; }
    }
    document.addEventListener("DOMContentLoaded", () => {
      document.querySelectorAll(".lang-btn").forEach(b => b.onclick = () => applyLang(b.dataset.lang));
      document.querySelectorAll(".input-tab").forEach(t => t.onclick = () => { document.querySelectorAll(".input-tab").forEach(x => x.classList.remove("active")); document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active")); t.classList.add("active"); $(`tab-${t.dataset.mode}`).classList.add("active"); });
      document.querySelectorAll(".scenario-btn").forEach(b => b.onclick = () => { document.querySelectorAll(".scenario-btn").forEach(x => x.classList.remove("active")); b.classList.add("active"); currentScenario = b.dataset.scenario; });
      $("demoBtn").onclick = runDemo; $("uploadBtn").onclick = uploadFile; $("manualBtn").onclick = runManual; $("reportBtn").onclick = generateReport; $("assistantSend").onclick = sendAssistant;
      $("assistantInput").onkeypress = e => { if (e.key === "Enter") sendAssistant(); };
      $("fillExampleBtn").onclick = fillExample; $("clearManualBtn").onclick = clearManual; $("addRowBtn").onclick = () => { manualRowCount++; buildManualGrid(); };
      document.querySelectorAll(".info-btn").forEach(b => b.onclick = () => showInfoModal(b.dataset.info));
      $("modalClose").onclick = () => $("infoModal").classList.remove("open");
      const fi = $("fileInput"), fd = $("fileDrop"); fd.onclick = () => fi.click(); fi.onchange = () => { if (fi.files.length) { selectedFile = fi.files[0]; $("fileName").textContent = selectedFile.name; } };
      fd.ondragover = e => e.preventDefault(); fd.ondrop = e => { e.preventDefault(); if (e.dataTransfer.files.length) { selectedFile = e.dataTransfer.files[0]; $("fileName").textContent = selectedFile.name; } };
      applyLang("en"); buildManualGrid();
    });
  </script>
</body>
</html>
"""

# ==========================================
# 5. تشغيل السيرفر (Waitress Production Server)
# ==========================================

if __name__ == "__main__":
    print("[AEGIS] Training engine. Please wait...")
    try:
        metrics = engine.train()
        print("[AEGIS] Metrics:", metrics)
    except Exception as exc:
        print(f"[AEGIS] Training failed: {exc}")

    print("\n[AEGIS] ========================================================")
    print("[AEGIS] Server is running on http://127.0.0.1:5000")
    print("[AEGIS] Using Waitress Production WSGI Server (No Warnings)")
    print("[AEGIS] ========================================================\n")

    serve(app, host='0.0.0.0', port=5000)