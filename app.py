from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# === Load Model ===
model = joblib.load("model_risiko_penyakit.pkl")

# === Load encoder meta (opsional) ===
try:
    meta = joblib.load("meta_encoder.json")
    jenis_classes = meta.get('jenis_kelamin_classes', None)
except:
    jenis_classes = None


# === ROUTES ===
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # --- Ambil input dari form ---
        usia = float(request.form.get("usia"))
        jenis = request.form.get("jenis_kelamin")

        # === Encoding jenis kelamin ===
        if jenis_classes is not None:
            try:
                jenis_enc = jenis_classes.index(jenis)
            except ValueError:
                jenis_enc = 1 if jenis.lower().startswith('l') or jenis.lower().startswith('m') else 0
        else:
            jenis_enc = 1 if jenis.lower().startswith('l') else 0

        indeks_massa_tubuh = float(request.form.get("indeks_massa_tubuh"))
        langkah_harian = float(request.form.get("langkah_harian"))
        jam_tidur = float(request.form.get("jam_tidur"))
        air_minum_per_hari = float(request.form.get("air_minum_per_hari"))
        asupan_kalori_harian = float(request.form.get("asupan_kalori_harian"))

        perokok = 1 if request.form.get("perokok") in ["Ya", "ya", "1", "True", "true"] else 0
        alkohol = 1 if request.form.get("alkohol") in ["Ya", "ya", "1", "True", "true"] else 0

        denyut_jantung = float(request.form.get("denyut_jantung"))
        systolic_bp = float(request.form.get("systolic_bp"))
        diastolic_bp = float(request.form.get("diastolic_bp"))
        kolesterol = float(request.form.get("kolesterol"))
        riwayat_keluarga = 1 if request.form.get("riwayat_keluarga") in ["Ya", "ya", "1", "True", "true"] else 0

        # === Fitur tambahan hasil rekayasa ===
        bmi_obesitas = 1 if indeks_massa_tubuh >= 30 else 0
        tidur_singkat = 1 if jam_tidur < 6 else 0

        # === Buat DataFrame input sesuai urutan fitur ===
        feature_names = [
            'usia','jenis_kelamin_enc','indeks_massa_tubuh','langkah_harian','jam_tidur',
            'air_minum_per_hari','asupan_kalori_harian','perokok','alkohol','denyut_jantung',
            'systolic_bp','diastolic_bp','kolesterol','riwayat_keluarga',
            'bmi_obesitas','tidur_singkat'
        ]

        row = [
            usia, jenis_enc, indeks_massa_tubuh, langkah_harian, jam_tidur,
            air_minum_per_hari, asupan_kalori_harian, perokok, alkohol, denyut_jantung,
            systolic_bp, diastolic_bp, kolesterol, riwayat_keluarga,
            bmi_obesitas, tidur_singkat
        ]

        X_input = pd.DataFrame([row], columns=feature_names)

        # === Prediksi ===
        pred = model.predict(X_input)[0]
        hasil = "⚠️ Berisiko Tinggi" if int(pred) == 0 else "✅ Tidak Berisiko"

    except Exception as e:
        hasil = f"❌ Terjadi kesalahan pada input: {e}"

    return render_template("index.html", hasil=hasil)


# === MAIN ===
if __name__ == "__main__":
    app.run(debug=True)
