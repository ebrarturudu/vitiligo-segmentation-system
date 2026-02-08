import sys
import os
sys.path.append(os.path.dirname(__file__))
from segment_anything.sam_model_registry import sam_model_registry
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === MODELLERİ YÜKLE ===
classification_model = EfficientNet.from_name('efficientnet-b0')
classification_model._fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(classification_model._fc.in_features, 1)
)
classification_model.load_state_dict(torch.load("model/models/siniflandirma_model.pth", map_location=device))
classification_model.eval().to(device)

class SAM_UNet(nn.Module):
    def __init__(self, num_classes=1):
       
        super(SAM_UNet, self).__init__() # super().__init__()'i ilk çağırın

        sam_path = "model/models/sam_vit_b.pth"
        assert os.path.exists(sam_path), f"❌ HATA: '{sam_path}' bulunamadı!"

        # self.sam'ı bir kez başlatın ve potansiyel yükleme sorunlarını doğru şekilde ele alın
        try:
            self.sam = sam_model_registry["vit_b"](checkpoint=sam_path)
            # Devam etmeden önce modelin yüklendiğinden emin olun
            assert self.sam is not None, "❌ SAM yüklenemedi. Model yükleme başarısız oldu."
        except Exception as e:
            raise RuntimeError(f"❌ SAM modeli yüklenirken bir hata oluştu: {e}")

        for param in self.sam.parameters():
            param.requires_grad = False
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.sam.get_image_embeddings(x)
        return self.decoder(x)

segmentation_model = SAM_UNet().to(device)
segmentation_model.load_state_dict(torch.load("model/models/best_model.pth", map_location=device))
segmentation_model.eval()

# === TRANSFORMS ===
cls_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

seg_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor()
])

# === GERÇEK segment_image ===
def segment_image(input_path, output_path):
    try:
        img = Image.open(input_path).convert("RGB")

        # Sınıflandır
        img_cls = cls_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = torch.sigmoid(classification_model(img_cls)).item()

        if pred < 0.5:
            print("🟡 Vitiligo tespit edilmedi.")
            return None, 0

        # Segmentasyon uygula
        img_seg = seg_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            mask = torch.sigmoid(segmentation_model(img_seg)).squeeze().cpu().numpy()

        # Binary maske ve alan hesapla
        mask_bin = (mask > 0.5).astype(np.uint8)
        ratio = round((mask_bin.sum() / mask_bin.size) * 100, 2)

        # Saydam bindirme
        img_resized = img.resize(mask.shape[::-1])
        rgba = img_resized.convert("RGBA")
        mask_img = Image.fromarray((mask_bin * 255).astype(np.uint8)).convert("L")
        red_mask = Image.new("RGBA", rgba.size, (255, 0, 0, 0))
        red_mask.putalpha(mask_img)
        overlay = Image.alpha_composite(rgba, red_mask)

        overlay_path = output_path.replace(".png", "_overlay.png")
        overlay.save(overlay_path)

        print(f"✅ Vitiligo tespit edildi. Alan: %{ratio}")
        return overlay_path, ratio

    except Exception as e:
        print("❌ segment_image HATASI:", e)
        return None, 0
