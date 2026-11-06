import os

# Ambil port dari environment variable PORT yang diberikan Railway, default ke 5000
port = os.environ.get("PORT", "5000")

# Wajib bind ke 0.0.0.0 agar dapat diakses dari proxy/router publik Railway
bind = f"0.0.0.0:{port}"

# Jumlah worker dan timeout untuk model ML
workers = 2
timeout = 120
