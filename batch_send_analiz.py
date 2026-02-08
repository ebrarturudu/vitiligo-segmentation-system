# batch_send_analiz.py
# /analiz endpoint'ine klasördeki görselleri POST eder ve CSV'yi doldurur.

import os, glob, time, requests

URL = "http://127.0.0.1:8080/analiz"
IMG_DIR = r"test_images"   # 20-30 görselini buraya koy (jpg/png)
FIELD_NAME = "resim"
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

    # Warm-up (opsiyonel)
    try:
        with open(paths[0], "rb") as f:
            r = requests.post(URL, files={FIELD_NAME: (os.path.basename(paths[0]), f)})
        print(f"[WARMUP] {paths[0]} -> {r.status_code}")
        time.sleep(0.3)
    except Exception as e:
        print("Warm-up isteği başarısız:", e)

    for p in paths:
        try:
            with open(p, "rb") as f:
                r = requests.post(URL, files={FIELD_NAME: (os.path.basename(p), f)})
            if r.status_code == 200:
                ok += 1
                print(f"[OK ] {os.path.basename(p)} -> 200")
            else:
                fail += 1
                print(f"[ERR] {os.path.basename(p)} -> {r.status_code}")
        except Exception as e:
            fail += 1
            print(f"[EXC] {os.path.basename(p)} -> {e}")
        time.sleep(0.05)

    print(f"\nBitti. Başarılı: {ok}, Hatalı: {fail}.")
    print("CSV: latency_log.csv (app.py ile aynı klasörde).")

if __name__ == "__main__":
    main()
