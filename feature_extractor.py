import torch
import torchvision.transforms as T
from PIL import Image
from torchreid.utils import FeatureExtractor
import torch.nn.functional as F
import cv2

class ReIDFeatureExtractor:
    def __init__(self, model_path, device='mps'):
        self.extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path=model_path,
            device=device
        )
        self.device = device
        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def extract(self, image, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = image[y1:y2, x1:x2]
        img_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.extractor.model(img_tensor)
        return feat.squeeze(0).cpu()

    @staticmethod
    def compare(f1, f2):
        if f1 is None or f2 is None:
            return 0.0
        return F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()
