# batch_send_predict.py
# /predict endpoint'ine klasördeki görselleri POST eder.
# JSON modu: ?api=1 ile çağırıyoruz ki süreler JSON olarak dönsün.

import os, glob, time, requests

URL = "http://127.0.0.1:8080/predict"
PARAMS = {"api": "1"}              # <-- JSON modu
IMG_DIR = r"test_images"           # 20-30 görselini buraya koy (jpg/png)
FIELD_NAME = "image"               # /predict form alanı adı
EXTS = ("*.jpg", "*.jpeg", "*.png")

def iter_images(folder):
    files = []
    for pat in EXTS:
        files.extend(glob.glob(os.path.join(folder, pat)))
    files.sort()
    return files

def main():
    paths = iter_images(IMG_DIR)
    if not paths:
        print(f"Uyarı: {IMG_DIR} içinde jpg/jpeg/png bulunamadı.")
        return

    print(f"Gönderilecek görsel sayısı: {len(paths)}")
    ok, fail = 0, 0

    # (Opsiyonel) warm-up
    try:
        with open(paths[0], "rb") as f:
            r = requests.post(URL, params=PARAMS, files={FIELD_NAME: (os.path.basename(paths[0]), f)})
        print(f"[WARMUP] {paths[0]} -> {r.status_code}")
        time.sleep(0.3)
    except Exception as e:
        print("Warm-up isteği başarısız:", e)

    for p in paths:
        try:
            with open(p, "rb") as f:
                r = requests.post(URL, params=PARAMS, files={FIELD_NAME: (os.path.basename(p), f)})
            if r.status_code == 200:
                ok += 1
                data = r.json()
                timings = data.get("timings_ms", {})
                clf = data.get("classification", {})
                print(f"[OK ] {os.path.basename(p)} -> "
                      f"total={timings.get('total')} ms, "
                      f"clf={timings.get('classification')} ms, "
                      f"seg={timings.get('segmentation')} ms, "
                      f"sev={timings.get('severity')} ms, "
                      f"is_vitiligo={clf.get('is_vitiligo')} "
                      f"(prob={round(clf.get('prob_vitiligo', 0.0), 3) if 'prob_vitiligo' in clf else 'n/a'})")
            else:
                fail += 1
                print(f"[ERR] {os.path.basename(p)} -> {r.status_code}  body={r.text[:200]}")
        except Exception as e:
            fail += 1
            print(f"[EXC] {os.path.basename(p)} -> {e}")
        time.sleep(0.05)

    print(f"\nBitti. Başarılı: {ok}, Hatalı: {fail}.")
    print("Not: latency_log.csv (app.py ile aynı klasörde) sınıflandırma/segmentasyon/severity sürelerini içeriyor.")

if __name__ == "__main__":
    main()
