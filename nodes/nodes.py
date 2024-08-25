import cv2
import torch
import numpy as np
from PIL import Image

def pil2tensor(image, device):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0).to(device)

class ImageStitchingNode:
    CATEGORY = "🧩 Custom Nodes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"list": True}),  # Especifica que espera uma lista de imagens
                "device": ("STRING", {"default": "cuda:0"}),  # Permite escolher a GPU
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Panoramic Image",)
    FUNCTION = "stitch_images"
    
    def stitch_images(self, images, device):
        # Verifica se recebeu pelo menos duas imagens
        if len(images) < 2:
            raise ValueError("At least two images are required for stitching.")

        # Verifica se o dispositivo especificado está disponível
        device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")

        # Converter os tensores para arrays de numpy compatíveis com OpenCV
        np_images = [np.array(image.squeeze(0).cpu().numpy() * 255, dtype=np.uint8) for image in images]
        
        # Converte para BGR que é o formato que OpenCV espera
        np_images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in np_images]

        # Cria o objeto Stitcher e realiza o stitching
        stitcher = cv2.Stitcher_create()
        (status, pano) = stitcher.stitch(np_images)

        # Verifica se o stitching foi bem-sucedido
        if status != cv2.Stitcher_OK:
            raise RuntimeError(f"Error when stitching: {status}")

        # Converte a imagem resultante para um tensor que o ComfyUI pode usar
        pano_pil = Image.fromarray(cv2.cvtColor(pano, cv2.COLOR_BGR2RGB))
        pano_tensor = pil2tensor(pano_pil, device)

        return (pano_tensor,)
