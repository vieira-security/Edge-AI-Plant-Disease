# 📦 Dataset — PlantVillage

## Descrição

O **PlantVillage Dataset** é uma coleção de referência amplamente utilizada na literatura para detecção de doenças em plantas via visão computacional.

| Atributo | Valor |
|----------|-------|
| Total de imagens | 54.305 |
| Número de classes | 38 |
| Espécies de plantas | 14 |
| Doenças cobertas | 26 |
| Resolução | 256 × 256 pixels |
| Formato | JPG |
| Licença | Creative Commons Attribution 4.0 (CC BY 4.0) |

## Origem

- **GitHub oficial:** https://github.com/spMohanty/PlantVillage-Dataset  
- **Kaggle:** https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset  
- **Referência:** Hughes, D.; Salathé, M. *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv:1511.08060, 2015.

## Classes Disponíveis

O dataset cobre as seguintes combinações espécie/condição (exemplos):

| Espécie | Condição |
|---------|----------|
| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| Corn | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Esca, Leaf Blight, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Mosaic Virus, Yellow Leaf Curl Virus, Healthy |
| ... | ... (38 classes no total) |

## Como Obter o Dataset

O dataset **não está incluído neste repositório** devido ao tamanho (~1,2 GB). Para reproduzir os experimentos:

```bash
# Opção 1: Kaggle CLI
pip install kaggle
kaggle datasets download -d abdallahalidev/plantvillage-dataset
unzip plantvillage-dataset.zip -d dataset/plantvillage/

# Opção 2: Download manual
# Acesse https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
# e faça o download manualmente para a pasta dataset/plantvillage/
```

## Estrutura Esperada

Após o download, a pasta deve ter a seguinte estrutura:

```
dataset/
├── README.md         ← este arquivo
└── plantvillage/
    ├── Apple___Apple_scab/
    │   ├── image001.jpg
    │   └── ...
    ├── Apple___Black_rot/
    ├── Tomato___Late_blight/
    └── ... (38 pastas no total)
```

## Estrutura para Treinamento (`src/train.py`)

`src/train.py` e o notebook 02 esperam o dataset já dividido em três pastas físicas
separadas (evita vazamento de dados entre treino/validação/teste):

```
dataset/
└── plantvillage_split/
    ├── train/
    │   ├── Apple___Apple_scab/
    │   └── ...
    ├── val/
    │   └── ...
    └── test/
        └── ...
```

Depois de baixar o dataset na estrutura plana acima (`dataset/plantvillage/<classe>/`),
divida-o em 70/15/15 antes de treinar — por exemplo com [`splitfolders`](https://pypi.org/project/split-folders/):

```bash
pip install split-folders
python -c "import splitfolders; splitfolders.ratio('dataset/plantvillage', output='dataset/plantvillage_split', ratio=(.7, .15, .15), seed=42)"
```

## Pré-processamento Aplicado

- Redimensionamento para **224 × 224 pixels** (entrada do MobileNetV2)
- Normalização:
  - Análise exploratória ([`notebooks/01_analise_exploratoria.ipynb`](../notebooks/01_analise_exploratoria.ipynb)): pixels / 255.0 (escala [0, 1])
  - Treinamento ([`notebooks/02_treinamento_quantizacao.ipynb`](../notebooks/02_treinamento_quantizacao.ipynb), [`src/train.py`](../src/train.py)): `tf.keras.applications.mobilenet_v2.preprocess_input` (escala [-1, 1], a normalização esperada pelo MobileNetV2 pré-treinado)
- Divisão: **70% treino / 15% validação / 15% teste** — em pastas físicas separadas (`train/`, `val/`, `test/`), não por `validation_split` reaproveitado (ver [`src/train.py`](../src/train.py))
- Data augmentation (apenas no treino): rotação ±30°, flip horizontal/vertical, zoom ±20%, ajuste de brilho/contraste

As funções de pré-processamento reutilizáveis (resize, normalização, cálculo de nitidez) estão centralizadas em [`src/utils/preprocessing.py`](../src/utils/preprocessing.py), evitando duplicação entre notebooks e scripts.

## Amostra usada nesta auditoria

Para permitir a execução real dos notebooks sem acesso a GPU nem à API do Kaggle, esta auditoria do repositório baixou uma amostra de 48 imagens/classe (1.824 imagens no total) diretamente do [repositório oficial do PlantVillage no GitHub](https://github.com/spMohanty/PlantVillage-Dataset). Essa amostra **não substitui** o dataset completo (54.306 imagens) — ver [README.md](../README.md#-limitações), seção "Limitações".

## Histórico de Atualizações

| Data | Autor | Descrição |
|------|-------|-----------|
| 10/05/2026 | Gabriel Vieira | Criação da descrição do dataset |
| 13/08/2026 | Auditoria do repositório | Corrigida divergência entre normalização documentada e normalização real do treino; documentado split treino/val/teste em pastas separadas; nota sobre amostra usada para execução dos notebooks |
