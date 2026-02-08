import requests

url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
output_path = "model/models/sam_vit_b.pth"

r = requests.get(url, stream=True)
with open(output_path, "wb") as f:
    for chunk in r.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)

print("✅ SAM modeli başarıyla indirildi.")
