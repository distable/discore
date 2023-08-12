import cv2
import numpy as np

from src.classes.Plugin import Plugin
from src.party.maths import norm
from src.plugins import plugfun, plugfun_img

class HFMidas3D(Plugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.midas = None
        self.midas_transforms = None

    def title(self):
        return "hfmidas3d"

    def describe(self):
        return ""

    def init(self):
        pass

    def install(self):
        pass

    def uninstall(self):
        pass

    def load(self):
        pass

    def unload(self):
        pass

    @plugfun(default_return=plugfun_img)
    def get_depth(self, img, model_type="DPT_Hybrid"):
        import torch

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Load midas (lazy & cached)
        if self.midas is None:
            self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
            self.midas.to(device)
            self.midas.eval()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.midas_transforms is None:
            self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        is_large = model_type == "DPT_Large"
        is_hybrid = model_type == "DPT_Hybrid"
        if is_large or is_hybrid:
            transform = self.midas_transforms.dpt_transform
        else:
            transform = self.midas_transforms.small_transform

        input_batch = transform(img).to(device)
        # rv.snap('midas_input', input_batch)

        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()
        grey = cv2.cvtColor((norm(output) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        return grey
