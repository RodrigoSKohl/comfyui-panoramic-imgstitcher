import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from comfy.model_management import get_torch_device

def cv2_to_tensor(image, device, return_mask=False):
    # Converte imagem BGR para RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Converte para float e normaliza
    rgb_image = rgb_image.astype(np.float32) / 255.0
    # Cria tensor da imagem: [1, H, W, C]
    image_tensor = torch.from_numpy(rgb_image).unsqueeze(0).to(device)

    if return_mask:
        # Cria máscara em escala de cinza e aplica threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.float32) / 255.0
        binary_mask = np.expand_dims(binary_mask, axis=-1)  # [H, W, 1]
        mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0).to(device)  # [1, H, W, 1]
        return image_tensor, mask_tensor

    return image_tensor


def remove_black_border(image):
    image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))  # Adiciona uma borda preta
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    cnts,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        # Se houver contornos, encontra o maior contorno e desenha ele(isso exclui pixeis pretos que não são parte do objeto)
        c = max(cnts, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)
        thresh = np.zeros_like(thresh)
        cv2.drawContours(thresh, [c], -1, 255, thickness=cv2.FILLED)
    else:
        return image

    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    minRect = mask.copy()
    sub = mask.copy()
    
    while cv2.countNonZero(sub) > 0:
        minRect = cv2.erode(minRect, None)
        sub = cv2.subtract(minRect, thresh)
    
    cnts,_ = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Se não houver contornos, retorna a imagem original
    if not cnts:
        return image
    
    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)
    image = image[y:y + h, x:x + w]
    
    return image


class ImageStitchingNode:
    def __init__(self):
        self.device = get_torch_device()    

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "crop": (["enable", "disable"],),
                "mode": (["panoramic", "scans"],),
                "conf_thresh": ("FLOAT",{
                    "min": 0.0,
                    "max": 1.0,
                    "default": 1.0,
                    "step": 0.01,
                    "round": 0.001,
                    "display": "number",
                }),
                "work_megapix": ("FLOAT", {
                    "min": 0.001,
                    "max": 100.0,
                    "default": 0.6,
                    "step": 0.01,
                    "round": 0.001,
                    "display": "number",
                }), 
                "seam_megapix": ("FLOAT", {
                    "min": 0.001,
                    "max": 100.0,
                    "default": 0.1,
                    "step": 0.01,
                    "round": 0.001,
                    "display": "number",
                }),


            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "stitch_images"
    CATEGORY = "🧩 Custom Nodes"

    
    def stitch_images(self, images, crop, mode, conf_thresh, work_megapix, seam_megapix):
        # Verifica se recebeu pelo menos duas imagens
        if len(images) < 2:
            raise ValueError("At least two images are required for stitching.")

        # Converter os tensores para arrays de numpy compatíveis com OpenCV
        np_images = [np.array(image.squeeze(0).cpu().numpy() * 255, dtype=np.uint8) for image in images]
        
        # Converte para BGR que é o formato que OpenCV espera
        np_images = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in np_images]

        # Cria o objeto Stitcher e realiza o stitching
        if mode == 'panoramic':
            stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        elif mode == 'scans':
            stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
        else:
            raise ValueError("Invalid mode. Use 'panoramic' or 'scans'.")
        
        stitcher.setPanoConfidenceThresh(conf_thresh)
        stitcher.setRegistrationResol(work_megapix)  
        stitcher.setSeamEstimationResol(seam_megapix)
        (status, pano) = stitcher.stitch(np_images)

        # Verifica se o stitching foi bem-sucedido
        if status != cv2.Stitcher_OK:
            raise RuntimeError(f"Error when stitching: {status}")
        
        
        # Aplica crop na imagem se solicitado
        if crop == "enable":
            pano = remove_black_border(pano)
            pano_tensor = cv2_to_tensor(pano, device=self.device)
        # Se crop não for solicitado, converte a imagem e aplica máscara
        else:
            pano_tensor, pano_mask = cv2_to_tensor(pano, device=self.device, return_mask=True)
            pano_mask = pano_mask.clamp(0, 1)
            pano_tensor = torch.cat([pano_tensor, pano_mask], dim=-1)

        return (pano_tensor,)




