import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from comfy.model_management import get_torch_device

def pil2tensor(image, device, rgb=True):
    if rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
        #kernel = np.ones((5, 5), np.uint8)
        #image = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2pill = Image.fromarray(image)
    return torch.from_numpy(np.array(cv2pill).astype(np.float32) / 255.0).unsqueeze(0).to(device)


def remove_black_border(image):
    image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))  # Adiciona uma borda preta
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    cnts,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thresh = np.zeros_like(thresh)
    for c in cnts:
        if cv2.contourArea(c) > 1:  # Ajuste o valor conforme o tamanho da borda externa
            cv2.drawContours(thresh, [c], -1, 255, thickness=cv2.FILLED)

    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    (x, y, w, h) = cv2.boundingRect(c)
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
                "images": ("IMAGE", {"list": True}),  # Especifica que espera uma lista de imagens
                "crop": (["enable", "disable"],),  # Permite escolher se deseja cortar a imagem
                "mode": (["panoramic", "scans"],), # Permite escolher o modo de stitching
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

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "stitch_images"
    CATEGORY = "🧩 Custom Nodes"
    
    def stitch_images(self, images, crop, mode, conf_thresh, work_megapix, seam_megapix):
        # Verifica se recebeu pelo menos duas imagens
        if len(images) < 2:
            raise ValueError("At least two images are required for stitching.")

        # Verifica se o dispositivo especificado está disponível

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
            raise ValueError("Invalid mode. Use 'PANORAMA' or 'SCANS'.")
        
        stitcher.setPanoConfidenceThresh(conf_thresh)
        stitcher.setRegistrationResol(work_megapix)  
        stitcher.setSeamEstimationResol(seam_megapix)
        (status, pano) = stitcher.stitch(np_images)

        # Verifica se o stitching foi bem-sucedido
        if status != cv2.Stitcher_OK:
            raise RuntimeError(f"Error when stitching: {status}")
        
        # Retorna a mascara original da imagem panorâmica mesmo se crop estiver habilitado
        pano_mask = pil2tensor(pano, device=self.device, rgb=False)
        # Corta a imagem para remover as bordas pretas usando a técnica de bounding box
        if crop == "enable":
            pano = remove_black_border(pano)

        

        # Converte a imagem resultante para um tensor que o ComfyUI pode usar
        pano_tensor = pil2tensor(pano, device=self.device, rgb=True)



        return (pano_tensor, pano_mask)

