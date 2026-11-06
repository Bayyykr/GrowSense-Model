import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.impute import SimpleImputer

# ==============================================================================
# MONKEYPATCH: Kompatibilitas antar-versi scikit-learn
# Memastikan semua instance SimpleImputer selalu memiliki _fill_dtype dan _fit_dtype
# ==============================================================================
def _safe_get_fill_dtype(self):
    if "_fill_dtype" in self.__dict__:
        return self.__dict__["_fill_dtype"]
    if "_fit_dtype" in self.__dict__:
        return self.__dict__["_fit_dtype"]
    if hasattr(self, "statistics_") and hasattr(self.statistics_, "dtype"):
        return self.statistics_.dtype
    return np.dtype("float64")

def _safe_set_fill_dtype(self, val):
    self.__dict__["_fill_dtype"] = val

def _safe_get_fit_dtype(self):
    if "_fit_dtype" in self.__dict__:
        return self.__dict__["_fit_dtype"]
    if "_fill_dtype" in self.__dict__:
        return self.__dict__["_fill_dtype"]
    if hasattr(self, "statistics_") and hasattr(self.statistics_, "dtype"):
        return self.statistics_.dtype
    return np.dtype("float64")

def _safe_set_fit_dtype(self, val):
    self.__dict__["_fit_dtype"] = val

SimpleImputer._fill_dtype = property(_safe_get_fill_dtype, _safe_set_fill_dtype)
SimpleImputer._fit_dtype = property(_safe_get_fit_dtype, _safe_set_fit_dtype)

app = Flask(__name__)
CORS(app)  # Memungkinkan CORS untuk akses dari frontend/mobile app

# Path ke file model joblib
MODEL_PATH = os.path.join(os.path.dirname(__file__), "sigap_model_rf.joblib")

def patch_sklearn_compat(obj):
    """
    Memperbaiki ketidakcocokan versi scikit-learn ketika model di-pickle pada versi lama/berbeda.
    Menyisir seluruh atribut, transformers_, _transformers, dan named_transformers_.
    """
    if obj is None:
        return

    # Jika objek adalah SimpleImputer, lengkapi atribut di __dict__
    if type(obj).__name__ == "SimpleImputer":
        fit_dtype = getattr(obj, "_fit_dtype", None)
        stats = getattr(obj, "statistics_", None)
        stats_dtype = getattr(stats, "dtype", None)
        chosen_dtype = fit_dtype if fit_dtype is not None else (stats_dtype if stats_dtype is not None else np.dtype("float64"))
        
        obj.__dict__["_fill_dtype"] = chosen_dtype
        obj.__dict__["_fit_dtype"] = chosen_dtype

    # Rekursif untuk ColumnTransformer
    if hasattr(obj, "transformers_"):
        for trans_entry in obj.transformers_:
            if len(trans_entry) >= 2:
                patch_sklearn_compat(trans_entry[1])

    if hasattr(obj, "_transformers"):
        for trans_entry in obj._transformers:
            if isinstance(trans_entry, (list, tuple)) and len(trans_entry) >= 2:
                patch_sklearn_compat(trans_entry[1])

    if hasattr(obj, "transformers"):
        for trans_entry in obj.transformers:
            if isinstance(trans_entry, (list, tuple)) and len(trans_entry) >= 2:
                patch_sklearn_compat(trans_entry[1])

    if hasattr(obj, "named_transformers_"):
        for _, trans in obj.named_transformers_.items():
            patch_sklearn_compat(trans)

    # Rekursif untuk Pipeline
    if hasattr(obj, "named_steps"):
        for _, step in obj.named_steps.items():
            patch_sklearn_compat(step)
    elif hasattr(obj, "steps"):
        for _, step in obj.steps:
            patch_sklearn_compat(step)

# Load model saat aplikasi dimulai
try:
    saved_data = joblib.load(MODEL_PATH)
    if isinstance(saved_data, dict):
        model = saved_data.get("model")
        FEATURE_NAMES = saved_data.get("features", [])
        ALERT_THRESHOLD = saved_data.get("alert_threshold", 0.5)
        TARGET_NAME = saved_data.get("target", "stunting")
        MODEL_NAME = saved_data.get("model_name", "Machine Learning Model")
    else:
        model = saved_data
        FEATURE_NAMES = getattr(model, "feature_names_in_", []).tolist() if hasattr(model, "feature_names_in_") else []
        ALERT_THRESHOLD = 0.5
        TARGET_NAME = "stunting"
        MODEL_NAME = type(model).__name__

    # Terapkan patch kompatibilitas ke model/pipeline
    patch_sklearn_compat(model)

    print(f"Model berhasil dimuat: {MODEL_NAME}")
    print(f"Fitur yang diharapkan: {FEATURE_NAMES}")
    print(f"Alert Threshold: {ALERT_THRESHOLD}")
except Exception as e:
    raise RuntimeError(f"Gagal memuat model dari {MODEL_PATH}: {e}")

# Mapping alias untuk mempermudah pemanggilan (misal: snake_case atau bahasa Indonesia)
FEATURE_ALIASES = {
    # Child Age
    "child_age": "Child Age",
    "childage": "Child Age",
    "usia_anak": "Child Age",
    "umur_anak": "Child Age",
    # Birth Weight
    "birth_weight": "Birth Weight",
    "birthweight": "Birth Weight",
    "berat_lahir": "Birth Weight",
    # Birth Length
    "birth_length": "Birth Length",
    "birthlength": "Birth Length",
    "panjang_lahir": "Birth Length",
    # ANC Visits
    "anc_visits": "ANC Visits",
    "ancvisits": "ANC Visits",
    "kunjungan_anc": "ANC Visits",
    # Exclusive Breastfeeding
    "exclusive_breastfeeding": "Exclusive Breastfeeding",
    "exclusivebreastfeeding": "Exclusive Breastfeeding",
    "asi_eksklusif": "Exclusive Breastfeeding",
    # Complementary Feeding Start Age
    "complementary_feeding_start_age": "Complementary Feeding Start Age",
    "complementaryfeedingstartage": "Complementary Feeding Start Age",
    "mpasi_start_age": "Complementary Feeding Start Age",
    "usia_mpasi": "Complementary Feeding Start Age",
    # Immunization
    "immunization": "Immunization",
    "imunisasi": "Immunization",
    # Sanitation
    "sanitation": "Sanitation",
    "sanitasi": "Sanitation",
    # Drinking Water Source
    "drinking_water_source": "Drinking Water Source",
    "drinkingwatersource": "Drinking Water Source",
    "sumber_air_minum": "Drinking Water Source"
}

def normalize_input(raw_dict):
    """
    Menormalkan dictionary input agar sesuai dengan nama fitur yang diharapkan model.
    """
    normalized = {}
    
    # Buat lookup dictionary case-insensitive & tanpa underscore/spasi
    cleaned_input = {}
    for k, v in raw_dict.items():
        clean_key = str(k).strip().lower().replace("_", "").replace(" ", "")
        cleaned_input[clean_key] = v

    for target_feature in FEATURE_NAMES:
        target_clean = target_feature.strip().lower().replace("_", "").replace(" ", "")
        
        # Cek kecocokan langsung atau melalui alias
        val = None
        if target_clean in cleaned_input:
            val = cleaned_input[target_clean]
        else:
            for alias_key, mapped_feature in FEATURE_ALIASES.items():
                if mapped_feature == target_feature:
                    alias_clean = alias_key.strip().lower().replace("_", "").replace(" ", "")
                    if alias_clean in cleaned_input:
                        val = cleaned_input[alias_clean]
                        break
        
        # Konversi ke float jika ditemukan
        if val is not None and val != "":
            try:
                normalized[target_feature] = float(val)
            except (ValueError, TypeError):
                raise ValueError(f"Nilai untuk fitur '{target_feature}' harus berupa angka numerik, diterima: {val}")
        else:
            normalized[target_feature] = None  # Imputer dalam pipeline akan menangani nilai kosong jika ada

    return normalized

def format_prediction_result(df_row, single_prediction, probabilities=None):
    prob_stunting = None
    if probabilities is not None and len(probabilities) > 1:
        prob_stunting = float(probabilities[1])
    elif probabilities is not None and len(probabilities) == 1:
        prob_stunting = float(probabilities[0])

    # Tentukan apakah risiko tinggi berdasarkan alert_threshold
    is_alert = False
    if prob_stunting is not None:
        is_alert = prob_stunting >= ALERT_THRESHOLD
    else:
        is_alert = bool(single_prediction == 1)

    risk_label = "Stunting" if is_alert else "Normal"

    result = {
        "prediction": int(single_prediction),
        "risk_label": risk_label,
        "is_alert": is_alert,
        "alert_threshold": round(float(ALERT_THRESHOLD), 4),
        "target": TARGET_NAME,
        "model_name": MODEL_NAME
    }

    if prob_stunting is not None:
        result["probability_stunting"] = round(prob_stunting, 4)
        result["probability_percentage"] = f"{prob_stunting * 100:.2f}%"

    return result

def safe_predict(df):
    """
    Menjalankan prediksi model dengan fallback aman jika scikit-learn mengalami
    ketidakcocokan atribut serialisasi (seperti SimpleImputer._fill_dtype).
    """
    try:
        preds = model.predict(df)
        probs = model.predict_proba(df) if hasattr(model, "predict_proba") else None
        return preds, probs
    except Exception as e:
        err_str = str(e)
        if "_fill_dtype" in err_str or "SimpleImputer" in err_str or "_fit_dtype" in err_str:
            # Fallback otomatis jika terjadi masalah kompatibilitas scikit-learn
            feature_order = [
                'Child Age', 'Birth Weight', 'Birth Length', 'ANC Visits', 'Complementary Feeding Start Age',
                'Exclusive Breastfeeding', 'Sanitation', 'Drinking Water Source', 'Immunization'
            ]
            defaults = {
                'Child Age': 30.0,
                'Birth Weight': 2.98,
                'Birth Length': 48.0,
                'ANC Visits': 4.0,
                'Complementary Feeding Start Age': 5.0,
                'Exclusive Breastfeeding': 0.0,
                'Sanitation': 1.0,
                'Drinking Water Source': 0.0,
                'Immunization': 2.0
            }
            clean_df = df.copy()
            for col in feature_order:
                if col not in clean_df.columns:
                    clean_df[col] = defaults[col]
                else:
                    clean_df[col] = clean_df[col].fillna(defaults[col])

            clean_df = clean_df[feature_order]
            rf = model.named_steps.get("model") if hasattr(model, "named_steps") else model
            preds = rf.predict(clean_df.values)
            probs = rf.predict_proba(clean_df.values) if hasattr(rf, "predict_proba") else None
            return preds, probs
        raise e

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "success",
        "message": "GrowSense Stunting Prediction API siap digunakan",
        "model_name": MODEL_NAME,
        "endpoints": {
            "GET /health": "Cek status API dan info model",
            "GET /version": "Cek versi deployment aktif",
            "POST /predict": "Prediksi risiko stunting untuk satu atau banyak data anak"
        }
    })

@app.route("/version", methods=["GET"])
def version():
    import sklearn
    return jsonify({
        "status": "success",
        "build_version": "v4-bulletproof",
        "sklearn_version": getattr(sklearn, "__version__", "unknown"),
        "message": "Railway berhasil menjalankan build terbaru dengan perlindungan safe_predict."
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "target": TARGET_NAME,
        "alert_threshold": ALERT_THRESHOLD,
        "required_features": FEATURE_NAMES
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error",
                "message": "Request body harus berupa JSON"
            }), 400

        # Dukung single object maupun list (batch)
        is_batch = isinstance(data, list)
        items = data if is_batch else [data]

        normalized_items = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                return jsonify({
                    "status": "error",
                    "message": f"Data pada indeks {i} harus berupa JSON object"
                }), 400
            normalized = normalize_input(item)
            normalized_items.append(normalized)

        # Ubah ke pandas DataFrame sesuai urutan fitur yang diharapkan
        df = pd.DataFrame(normalized_items, columns=FEATURE_NAMES)

        # Lakukan prediksi dengan fungsi aman
        preds, probs = safe_predict(df)

        results = []
        for i in range(len(preds)):
            row_prob = probs[i] if probs is not None else None
            res = format_prediction_result(df.iloc[i], preds[i], row_prob)
            results.append(res)

        return jsonify({
            "status": "success",
            "data": results if is_batch else results[0]
        }), 200

    except ValueError as ve:
        return jsonify({
            "status": "error",
            "message": str(ve)
        }), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Terjadi kesalahan pada server: {str(e)}"
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Menjalankan Flask API pada port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
