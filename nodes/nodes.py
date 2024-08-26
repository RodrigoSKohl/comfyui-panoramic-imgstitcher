import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

def pil2tensor(image, device, rgb=True):
    if rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2pill = Image.fromarray(image)
    return torch.from_numpy(np.array(cv2pill).astype(np.float32) / 255.0).unsqueeze(0).to(device)



def image_mask(image, device):
    # Converte a imagem para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Define um limiar baixo para detectar a borda preta
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    
    # Remove pequenos pontos internos que não fazem parte da borda externa
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Opcional: Use uma máscara de área mínima para garantir que apenas a borda externa seja capturada
    # Encontra contornos e filtra os pequenos contornos
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(mask)
    for c in cnts:
        if cv2.contourArea(c) > 100:  # Ajuste o valor conforme o tamanho da borda externa
            cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
    
    # Converte a máscara para tensor
    mask_tensor = pil2tensor(mask, device, rgb=False)
    
    return mask_tensor



def apply_mask(image_tensor, mask_tensor):
    image_tensor = image_tensor.float()

    # Redimensiona a máscara para combinar com a imagem, se necessário
    if image_tensor.shape[2:] != mask_tensor.shape[2:]:
        mask_tensor = F.interpolate(mask_tensor, size=image_tensor.shape[2:], mode='bilinear', align_corners=False)
    
    # Expande a máscara para ter o mesmo número de canais da imagem
    if mask_tensor.shape[1] != image_tensor.shape[1]:
        mask_tensor = mask_tensor.repeat(1, image_tensor.shape[1], 1, 1)

    # Aplica a máscara
    masked_image = image_tensor * mask_tensor
    print("Image tensor shape:", image_tensor.shape)
    print("Mask tensor shape before resize:", mask_tensor.shape)
    print("Mask tensor shape after resize:", mask_tensor.shape)

    return masked_image


def remove_black_border(image):
    image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))  # Adiciona uma borda preta
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thresh = np.zeros_like(thresh)
    for c in cnts:
        if cv2.contourArea(c) > 100:  # Ajuste o valor conforme o tamanho da borda externa
            cv2.drawContours(thresh, [c], -1, 255, thickness=cv2.FILLED)

    #cnts_h = imutils.grab_contours(cnts)
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
    #cnts = imutils.grab_contours(cnts)
    # Se não houver contornos, retorna a imagem original
    if not cnts:
        return image
    
    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)
    image = image[y:y + h, x:x + w]
    
    return image


class ImageStitchingNode:
    

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"list": True}),  # Especifica que espera uma lista de imagens
                "device": (["cuda", "cpu"],),  # Permite escolher a GPU
                "crop": (["enable", "disable"],),  # Permite escolher se deseja cortar a imagem
                "mode": (["panoramic", "scans"],), # Permite escolher o modo de stitching
                "threshold": ("FLOAT",{
                    "min": 0.0,
                    "max": 1.0,
                    "default": 1.0,
                    "step": 0.01,
                    "round": 0.001,
                    "display": "number",
                }),  # Permite escolher o limiar para a má

            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "stitch_images"
    CATEGORY = "🧩 Custom Nodes"
    
    def stitch_images(self, images, device, crop, mode, threshold):
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
        if mode == 'panoramic':
            stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        elif mode == 'scans':
            stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
        else:
            raise ValueError("Invalid mode. Use 'PANORAMA' or 'SCANS'.")
        
        stitcher.setPanoConfidenceThresh(threshold)
        (status, pano) = stitcher.stitch(np_images)

        # Verifica se o stitching foi bem-sucedido
        if status != cv2.Stitcher_OK:
            raise RuntimeError(f"Error when stitching: {status}")

        # Corta a imagem para remover as bordas pretas usando a técnica de bounding box
        if crop == "enable":
            pano = remove_black_border(pano)
        else:
            pano_mask = np.ones(pano.shape[:2], dtype=np.uint8) * 255 # Máscara branca se não for cortar

        pano_mask = image_mask(pano, device)

        # Converte a imagem resultante para um tensor que o ComfyUI pode usar
        pano_tensor = pil2tensor(pano, device)



        return (pano_tensor, pano_mask)

