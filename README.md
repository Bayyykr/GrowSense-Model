# GrowSense - Stunting Prediction API (Flask)

API backend berbasis Python Flask untuk melayani inferensi model machine learning (`sigap_model_rf.joblib`).

## 📁 Struktur File
```
growsenseapi/
├── app.py                 # File server Flask utama & konfigurasi endpoint
├── sigap_model_rf.joblib  # Model Random Forest Pipeline & metadata
├── requirements.txt       # Dependensi pustaka Python
├── test_api.py            # Unit test untuk pengujian otomatis endpoint
└── README.md              # Dokumentasi penggunaan API
```

---

## 🚀 Cara Menjalankan

### 1. Pastikan Dependensi Terpasang
Buka terminal dan jalankan:
```bash
pip install -r requirements.txt
```

### 2. Jalankan Server Flask
```bash
python app.py
```
Server akan berjalan di `http://127.0.0.1:5000` (atau port yang ditentukan pada environment variable `PORT`).

### 3. Menjalankan Unit Test
Untuk memastikan semua endpoint berjalan dengan baik:
```bash
python test_api.py
```

---

## 📡 Dokumentasi Endpoint

### 1. Cek Informasi Model & Status API
- **Endpoint**: `GET /health`
- **Deskripsi**: Mengembalikan status API, model yang dimuat, alert threshold, dan daftar nama fitur yang diharapkan.
- **Contoh Request**:
  ```bash
  curl -X GET http://127.0.0.1:5000/health
  ```
- **Contoh Response**:
  ```json
  {
    "alert_threshold": 0.4755201794307802,
    "model_loaded": true,
    "model_name": "Random Forest",
    "required_features": [
      "Child Age",
      "Birth Weight",
      "Birth Length",
      "ANC Visits",
      "Exclusive Breastfeeding",
      "Complementary Feeding Start Age",
      "Immunization",
      "Sanitation",
      "Drinking Water Source"
    ],
    "status": "healthy",
    "target": "stunting (HAZ < -2)"
  }
  ```

---

### 2. Prediksi Risiko Stunting (Single Item)
- **Endpoint**: `POST /predict`
- **Headers**: `Content-Type: application/json`
- **Keterangan Input**:
  API mendukung penamaan atribut baik dalam format asli (`"Child Age"`), snake_case (`"child_age"`), maupun istilah bahasa Indonesia (`"usia_anak"`, `"asi_eksklusif"`, dll).

| Fitur | Alias (snake_case / ID) | Tipe Data | Deskripsi |
| :--- | :--- | :--- | :--- |
| `Child Age` | `child_age`, `usia_anak` | Integer/Float | Usia anak (bulan) |
| `Birth Weight` | `birth_weight`, `berat_lahir` | Float | Berat lahir (kg) |
| `Birth Length` | `birth_length`, `panjang_lahir` | Float | Panjang lahir (cm) |
| `ANC Visits` | `anc_visits`, `kunjungan_anc` | Integer | Jumlah kunjungan antenatal care saat hamil |
| `Exclusive Breastfeeding`| `exclusive_breastfeeding`, `asi_eksklusif` | Integer (0/1) | Pemberian ASI Eksklusif (1 = Ya, 0 = Tidak) |
| `Complementary Feeding Start Age` | `complementary_feeding_start_age`, `mpasi_start_age` | Integer/Float | Usia mulai MPASI (bulan) |
| `Immunization` | `immunization`, `imunisasi` | Integer | Status kelengkapan imunisasi (0, 1, 2) |
| `Sanitation` | `sanitation`, `sanitasi` | Integer (0/1) | Sanitasi layak (1 = Ya, 0 = Tidak) |
| `Drinking Water Source` | `drinking_water_source`, `sumber_air_minum` | Integer (0/1) | Sumber air minum layak (1 = Ya, 0 = Tidak) |

#### Contoh Request Body (JSON):
```json
{
  "child_age": 12,
  "birth_weight": 3.1,
  "birth_length": 49.0,
  "anc_visits": 4,
  "exclusive_breastfeeding": 1,
  "complementary_feeding_start_age": 6,
  "immunization": 1,
  "sanitation": 1,
  "drinking_water_source": 1
}
```

#### Contoh cURL:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "child_age": 12,
    "birth_weight": 3.1,
    "birth_length": 49.0,
    "anc_visits": 4,
    "exclusive_breastfeeding": 1,
    "complementary_feeding_start_age": 6,
    "immunization": 1,
    "sanitation": 1,
    "drinking_water_source": 1
  }'
```

#### Contoh Response:
```json
{
  "status": "success",
  "data": {
    "prediction": 0,
    "risk_label": "Normal",
    "probability_stunting": 0.1726,
    "probability_percentage": "17.26%",
    "alert_threshold": 0.4755,
    "is_alert": false,
    "model_name": "Random Forest",
    "target": "stunting (HAZ < -2)"
  }
}
```

---

### 3. Prediksi Batch (Banyak Data Sekaligus)
Endpoint `/predict` secara otomatis mendukung payload berupa `array/list` objek JSON.

#### Contoh Request Body (Batch):
```json
[
  {
    "child_age": 12,
    "birth_weight": 3.2,
    "birth_length": 49.5,
    "anc_visits": 5,
    "exclusive_breastfeeding": 1,
    "complementary_feeding_start_age": 6,
    "immunization": 1,
    "sanitation": 1,
    "drinking_water_source": 1
  },
  {
    "child_age": 18,
    "birth_weight": 2.0,
    "birth_length": 44.0,
    "anc_visits": 1,
    "exclusive_breastfeeding": 0,
    "complementary_feeding_start_age": 3,
    "immunization": 0,
    "sanitation": 0,
    "drinking_water_source": 0
  }
]
```

#### Contoh Response (Batch):
```json
{
  "status": "success",
  "data": [
    {
      "prediction": 0,
      "risk_label": "Normal",
      "probability_stunting": 0.1238,
      "probability_percentage": "12.38%",
      "alert_threshold": 0.4755,
      "is_alert": false,
      "model_name": "Random Forest",
      "target": "stunting (HAZ < -2)"
    },
    {
      "prediction": 1,
      "risk_label": "Stunting",
      "probability_stunting": 0.9277,
      "probability_percentage": "92.77%",
      "alert_threshold": 0.4755,
      "is_alert": true,
      "model_name": "Random Forest",
      "target": "stunting (HAZ < -2)"
    }
  ]
}
```

---

### 4. Integrasi Langsung Aplikasi GrowSense (Kontrak AI v1)
Endpoint `POST /predict` secara otomatis mendeteksi request dari aplikasi **GrowSense PWA Kader** maupun **Dashboard Web Bidan** (`contract: 1`) sesuai spesifikasi `docs/ai-contract.md`.

Cukup arahkan variabel environment `VITE_AI_URL` pada aplikasi GrowSense ke endpoint `/predict`:
```env
VITE_AI_URL=https://url-hosting-api-anda.up.railway.app/predict
```

#### Contoh Request Body (Kontrak GrowSense):
```json
{
  "contract": 1,
  "instances": [
    {
      "id": "visit-risiko-002",
      "features": {
        "sex": "female",
        "age_months": 14,
        "haz": -1.31,
        "waz": -1.05,
        "whz": -0.42,
        "haz_change_3m": -0.46,
        "waz_change_3m": -0.38,
        "months_between": 3,
        "visits": 7,
        "birth_weight_kg": 2.1,
        "asi_eksklusif": false,
        "mpasi_adekuat": false,
        "imunisasi_lengkap": false,
        "sanitasi_layak": false
      },
      "history": []
    }
  ]
}
```

#### Contoh Response (Kontrak GrowSense):
```json
{
  "model_version": "sigap-rf-v1",
  "predictions": [
    {
      "id": "visit-risiko-002",
      "score": 0.8148,
      "level": "tinggi",
      "factors": [
        "haz_change_3m",
        "waz_change_3m",
        "birth_weight_kg"
      ]
    }
  ]
}
```
