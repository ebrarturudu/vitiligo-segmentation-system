# model/segmenter.py
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model.sam_unet import SAM_UNet
import os
from torchvision.transforms.functional import to_pil_image

class VitiligoSegmenter:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device)
        self.model = SAM_UNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])
        print("Model yüklendi.")

    def segment_image(self, input_path: str, output_path: str):
        try:
            image = Image.open(input_path).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(input_tensor)

            output_mask = torch.sigmoid(output).squeeze().cpu()
            binary_mask = (output_mask > 0.5).float()

            mask_image = to_pil_image(binary_mask)
            mask_image = mask_image.resize(image.size)
            mask_image.save(output_path)

            return output_path, float(binary_mask.mean()) 

        except Exception as e:
            print(f"Hata: {e}")
            return None, 0.0
