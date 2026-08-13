"""
Funções de pré-processamento de imagem compartilhadas entre os notebooks
(análise exploratória e treinamento) e os scripts de src/ (train.py,
quantize.py, inference_rpi.py).

Mantidas aqui para evitar duplicação de código entre notebook e script,
como pedido na auditoria do repositório (ver README.md, seção
"Reprodutibilidade").
"""
from __future__ import annotations

import numpy as np

IMG_SIZE = (224, 224)  # (largura, altura) esperado pelo MobileNetV2
SHARPNESS_THRESHOLD = 100.0  # variância do Laplaciano abaixo disso = imagem de baixa qualidade


def resize_image(image: np.ndarray, size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Redimensiona uma imagem HxWxC para o tamanho de entrada do modelo.

    Usa OpenCV se disponível; caso contrário, faz um resize simples via PIL
    para manter a função utilizável em ambientes sem opencv instalado
    (ex.: dispositivo de borda minimalista).
    """
    try:
        import cv2

        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    except ImportError:
        from PIL import Image

        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize(size)
        return np.array(pil_img)


def normalize_mobilenet(image: np.ndarray) -> np.ndarray:
    """Normaliza pixels para o intervalo [-1, 1], como esperado pelo
    ``tf.keras.applications.mobilenet_v2.preprocess_input`` (equivalente a
    ``(pixel / 127.5) - 1.0``). Usado no treinamento (notebook 02) e na
    inferência no Raspberry Pi (src/inference_rpi.py).
    """
    image = image.astype(np.float32)
    return (image / 127.5) - 1.0


def normalize_unit_scale(image: np.ndarray) -> np.ndarray:
    """Normaliza pixels para o intervalo [0, 1] (``pixel / 255.0``), usado
    na análise exploratória (notebook 01) e como pré-processamento genérico
    documentado em dataset/README.md.
    """
    return image.astype(np.float32) / 255.0


def preprocess_for_inference(image: np.ndarray, size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Pipeline completo de pré-processamento para inferência: resize +
    normalização MobileNetV2 + expansão de dimensão de batch.

    Recebe uma imagem RGB ``HxWx3`` (uint8) e devolve um array
    ``1xHxWx3`` (float32) pronto para ``interpreter.set_tensor``.
    """
    resized = resize_image(image, size)
    normalized = normalize_mobilenet(resized)
    return np.expand_dims(normalized, axis=0)


def laplacian_variance(image: np.ndarray) -> float:
    """Calcula a variância do operador Laplaciano de uma imagem em tons de
    cinza — usado como métrica de nitidez/qualidade (notebook 01, seção
    "Análise de Qualidade das Imagens").
    """
    import cv2

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_low_quality(image: np.ndarray, threshold: float = SHARPNESS_THRESHOLD) -> bool:
    """Retorna True se a imagem estiver abaixo do limiar de nitidez."""
    return laplacian_variance(image) < threshold


def parse_class_folder_name(folder_name: str) -> tuple[str, str, bool]:
    """Extrai (espécie, condição, é_saudável) do nome de pasta do
    PlantVillage, no formato ``Especie___Condicao`` (ex.:
    ``Tomato___Late_blight``). Mesma lógica usada no notebook 01.
    """
    parts = folder_name.split("___")
    species = parts[0].replace("_", " ") if len(parts) > 0 else folder_name
    condition = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    is_healthy = "healthy" in condition.lower()
    return species, condition, is_healthy
