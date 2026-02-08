from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename
import torch
from model.vitiligosegmenter import VitiligoSegmenter

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Model path is taken from environment variable
MODEL_PATH = os.getenv("VITILIGO_MODEL_PATH")

model = None
if MODEL_PATH:
    try:
        model = VitiligoSegmenter(
            model_path=MODEL_PATH,
            device=device
        )
    except Exception as e:
        print(f"[ERROR] Model yüklenemedi: {e}")
else:
    print("[WARNING] VITILIGO_MODEL_PATH tanımlı değil.")


@app.route('/')
def index():
    return render_template('base.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model yüklenmedi. README dosyasını inceleyin.", 500

    if 'image' not in request.files:
        return "Dosya bulunamadı", 400

    file = request.files['image']
    if file.filename == '':
        return "Dosya seçilmedi", 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)

    output_filename = 'mask_' + filename
    output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)

    result_path, _ = model.segment_image(input_path, output_path)

    return render_template(
        'result.html',
        original_image_url=input_path.replace("\\", "/"),
        mask_image_url=result_path.replace("\\", "/") if result_path else None
    )


@app.route("/analiz", methods=["GET", "POST"])
def analiz():
    if model is None:
        return "Model yüklenmedi. README dosyasını inceleyin.", 500

    if request.method == "POST":
        dosya = request.files.get("resim")
        if dosya:
            dosya_adi = secure_filename(dosya.filename)
            yuklenen_yol = os.path.join(app.config["UPLOAD_FOLDER"], dosya_adi)
            dosya.save(yuklenen_yol)

            output_filename = "mask_" + dosya_adi
            sonuc_yolu = os.path.join(app.config["RESULT_FOLDER"], output_filename)

            overlay_path, oran = model.segment_image(yuklenen_yol, sonuc_yolu)

            if oran < 0.1:
                return render_template(
                    "analiz.html",
                    giris=yuklenen_yol.replace("\\", "/"),
                    cikti=None,
                    oran=0,
                    mesaj="Vitiligo tespit edilmedi."
                )

            return render_template(
                "analiz.html",
                giris=yuklenen_yol.replace("\\", "/"),
                cikti=overlay_path.replace("\\", "/") if overlay_path else None,
                oran=round(oran * 100, 2)
            )

    return render_template("analiz.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)