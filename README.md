# 🌱 Edge AI para Detecção de Estresse Hídrico e Pragas com Redução de Overhead de Rede

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](requirements.txt)
[![Status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow.svg)](#-limitações)

> Projeto da disciplina de Inteligência Artificial — Universidade Presbiteriana Mackenzie
> Faculdade de Computação e Informática — 7º Período CC Noite
> Prof. Dr. Ivan Carlos Alcântara de Oliveira

---

## 👥 Integrantes

| Nome | RA | E-mail |
|------|----|--------|
| Gabriel Vieira de Sousa | 10410264 | 10410264@mackenzista.com.br |
| Guilherme Rainho Geraldo | 10418251 | 10418251@mackenzista.com.br |

---

## 📋 Descrição do Projeto

Sistema de **visão computacional embarcado em dispositivos de borda** (Raspberry Pi 4) para identificar, em tempo real, sintomas de **estresse hídrico**, **doenças foliares** e **presença de pragas** em plantações.

O sistema utiliza o modelo **MobileNetV2** com quantização **INT8** via **TensorFlow Lite**, treinado sobre o dataset **PlantVillage** (54.306 imagens, 38 classes). Ao transmitir apenas alertas estruturados via **LoRaWAN/MQTT** em vez de imagens brutas, o sistema reduz o overhead de rede em **99,5%** em comparação com abordagens tradicionais baseadas em nuvem (ver [Limitações](#-limitações) sobre o escopo dessa medição).

**Área:** Sustentabilidade / Agricultura de Precisão
**Opção:** ML/DL/VC/PLN (Visão Computacional + Deep Learning)

---

## 🏆 Resultados Oficiais (artigo N2 — dataset completo)

Números reportados no [artigo N2](docs/Artigo_N2_EdgeAI_PlantDisease.pdf) (Tabelas 2 e 3), obtidos com o dataset PlantVillage completo (54.306 imagens, 50 épocas):

| Métrica | FP32 | INT8 PTQ | INT8 QAT | Meta |
|---------|------|----------|----------|------|
| Acurácia | 96,4% | 95,8% | 96,1% | ≥ 95% ✅ |
| F1-Score (macro) | 0,962 | 0,957 | 0,960 | — |
| Tamanho do modelo | 13,4 MB | 3,6 MB | 3,6 MB | < 4 MB ✅ |
| Latência **estimada** (Raspberry Pi 4)¹ | ~287 ms | **74 ms** | **72 ms** | 30–100 ms ✅ |
| Redução de tráfego de rede | — | **99,5%** | **99,5%** | ≥ 90% ✅ |

> ¹ **Importante:** essa latência não foi medida em um Raspberry Pi 4 físico. É uma estimativa: mede-se o tempo real de inferência em CPU de desenvolvimento e aplica-se um fator de escala da literatura para ARM Cortex-A72. Ver [Limitações](#-limitações).

### Reprodução neste repositório (amostra reduzida)

Os notebooks em [`notebooks/`](notebooks/) foram executados de fato neste repositório com uma **amostra representativa** do PlantVillage (1.824 imagens, 48/classe — não o dataset completo, por falta de GPU neste ambiente de auditoria). Números reais obtidos (ver [`docs/resultados_finais.csv`](docs/resultados_finais.csv)):

| Modelo | Acurácia | F1-Score (macro) | Tamanho | Latência estimada RPi4¹ |
|---|---|---|---|---|
| MobileNetV2 FP32 | 81,91% | 0,813 | 9,74 MB | ~189 ms |
| INT8 PTQ | 82,66% | 0,791 | 2,90 MB | ~45 ms |
| INT8 QAT | 58,47% | 0,556 | 3,84 MB | ~48 ms |

Esses números **não substituem** os oficiais acima — a amostra é 30x menor que o dataset completo e o treino usa muito menos épocas efetivas. Servem para provar que o pipeline roda de ponta a ponta e gera saída real, não simulada. Um resultado chama atenção e vale registrar: **nesta amostra, o QAT piorou em vez de melhorar** em relação ao PTQ (58,5% vs. 82,7% de acurácia) — plausível com apenas 5 épocas de ajuste em pouquíssimos dados, mas também nunca antes verificado, porque o QAT do notebook original **nunca chegou a rodar de fato** (ver nota abaixo). Ver nota de escopo completa no topo do notebook.

---

## 🗂️ Estrutura do Repositório

```
Edge-AI-Plant-Disease/
│
├── README.md                              # Este arquivo
├── LICENSE                                 # MIT
├── CITATION.cff                            # Metadados para citação
├── requirements.txt                        # Dependências de treino/desenvolvimento
├── requirements-edge.txt                   # Dependências de inferência no Raspberry Pi
├── pytest.ini
│
├── docs/
│   ├── README.md                           # Índice e status dos documentos
│   ├── Artigo_N2_EdgeAI_PlantDisease.pdf   # Relatório final (N2)
│   ├── relatorio_parcial/
│   │   └── Artigo_N1_proposta.pdf          # Relatório parcial (N1) — metas, não resultado final
│   ├── fig1_distribuicao_classes.png
│   ├── fig2_amostras_visuais.png
│   ├── fig3_analise_rgb.png
│   ├── fig4_qualidade_imagens.png
│   ├── fig5_correlacao_especie_doenca.png
│   ├── fig6_curvas_aprendizado.png
│   ├── fig7_matriz_confusao.png
│   ├── fig8_metricas_sistema.png
│   └── resultados_finais.csv               # Métricas reais da amostra executada neste repo
│
├── dataset/
│   └── README.md                           # Descrição e instruções do dataset
│
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb       # Análise exploratória do PlantVillage (N1) — executado
│   └── 02_treinamento_quantizacao.ipynb    # Treinamento, quantização e benchmark (N2) — executado
│
├── src/
│   ├── train.py                            # Treinamento via linha de comando/CI
│   ├── quantize.py                         # Quantização PTQ + QAT via linha de comando/CI
│   ├── inference_rpi.py                    # Inferência real no Raspberry Pi (tflite-runtime)
│   ├── mqtt_publisher.py                   # Payload de alerta + publicação MQTT
│   └── utils/
│       └── preprocessing.py                # Pré-processamento compartilhado (notebooks + scripts)
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_mqtt_publisher.py
│
└── .github/workflows/
    └── lint.yml                            # ruff + pytest em cada push/PR
```

> `models/` (`.h5` / `.tflite`) não é versionado no Git — ver [Modelos Treinados](#-modelos-treinados). `dataset/plantvillage*/` também não é versionado — ver [`dataset/README.md`](dataset/README.md).

---

## 🛠️ Tecnologias Utilizadas

- Python 3.11
- TensorFlow 2.15 / TensorFlow Lite
- TensorFlow Model Optimization (QAT)
- MobileNetV2 (pré-treinada em ImageNet)
- OpenCV, Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn (métricas de avaliação)
- paho-mqtt (publicação de alertas)
- Jupyter Notebook

---

## 📊 Dataset

**PlantVillage** — 54.306 imagens de folhas de plantas, organizadas em 38 classes (14 espécies × 26 doenças + folhas saudáveis). Detalhes em [`dataset/README.md`](dataset/README.md).

---

## 🚀 Como Reproduzir

```bash
# 1. Clone o repositório
git clone https://github.com/vieira-security/Edge-AI-Plant-Disease.git
cd Edge-AI-Plant-Disease

# 2. Instale as dependências de treino
pip install -r requirements.txt

# 3. Baixe o dataset PlantVillage (Kaggle CLI — requer conta e API key)
kaggle datasets download -d abdallahalidev/plantvillage-dataset
unzip plantvillage-dataset.zip -d dataset/plantvillage/

# 4. Execute os notebooks em ordem
jupyter notebook notebooks/01_analise_exploratoria.ipynb
jupyter notebook notebooks/02_treinamento_quantizacao.ipynb

# ...ou via linha de comando (equivalente, para CI/scripts):
python src/train.py --data dataset/plantvillage_split --output models/
python src/quantize.py --data dataset/plantvillage_split --model models/mobilenetv2_fp32_best.h5 --output models/
```

### Implantar no Raspberry Pi 4

```bash
# No RPi4 — instalar dependências leves de inferência
pip install -r requirements-edge.txt

# Executar inferência com o modelo quantizado (câmera)
python src/inference_rpi.py --model models/mobilenetv2_int8_ptq.tflite --camera 0

# Ou com uma imagem estática, e publicando o alerta via MQTT
python src/inference_rpi.py --model models/mobilenetv2_int8_ptq.tflite \
    --image folha.jpg --publish --broker meu-broker.local --lat -23.55 --lon -46.63
```

---

## 📦 Modelos Treinados

Os modelos (`.h5` FP32 e `.tflite` FP32/INT8-PTQ/INT8-QAT) **não são versionados no Git** por tamanho (`models/` está no `.gitignore`).

**[TODO: publicar `mobilenetv2_fp32_best.h5`, `mobilenetv2_fp32.tflite`, `mobilenetv2_int8_ptq.tflite` e `mobilenetv2_int8_qat.tflite` como assets de uma GitHub Release deste repositório antes da entrega final, e trocar este parágrafo pelo link direto da Release.]**

Enquanto isso, os modelos podem ser gerados localmente com `src/train.py` + `src/quantize.py` (ver [Como Reproduzir](#-como-reproduzir)) ou executando os notebooks.

---

## ⚠️ Limitações

- **Dataset controlado vs. campo real:** o PlantVillage foi capturado em ambiente controlado (fundo uniforme, iluminação padronizada), diferente das condições reais de uma lavoura (fundo variado, iluminação natural, oclusão parcial de folhas). O desempenho em campo tende a ser inferior ao medido aqui — não houve, até o momento, validação com imagens de campo.
- **Latência de Raspberry Pi 4 é estimada, não medida:** os números de latência "no RPi4" (tanto no artigo N2 quanto em `docs/resultados_finais.csv`) foram obtidos medindo o tempo real de inferência em CPU de desenvolvimento e aplicando um fator de escala da literatura para ARM Cortex-A72 — **não há hardware Raspberry Pi 4 físico envolvido na medição**. `src/inference_rpi.py` imprime a latência real quando executado de fato num dispositivo — essa é a única fonte confiável desse número.
- **Redução de tráfego de rede é uma estimativa analítica:** o cálculo (Tabela 3 do artigo N2) assume parâmetros de operação (1 captura/minuto, 20% de taxa de alerta, JPEG de ~75 KB) — não é uma medição em campo com um gateway LoRaWAN real. `src/mqtt_publisher.py` mede o tamanho real do payload de alerta em bytes, o que dá base empírica à metade "alertas" da conta, mas o lado "baseline" (envio de imagem) continua sendo estimado.
- **Sem integração LoRaWAN física:** o pipeline usa MQTT como transporte; a integração com um gateway LoRaWAN real (e a validação de alcance/confiabilidade em zona rural) é trabalho futuro, como já apontado na conclusão do artigo N2.
- **Amostra reduzida nos notebooks deste repositório:** por não haver GPU neste ambiente de auditoria, os notebooks foram executados com uma amostra de 1.824 imagens (não as 54.306 completas) — ver nota de escopo no topo de cada notebook e distinção entre resultados oficiais/amostra na seção de Resultados acima.
- **O QAT do notebook original nunca tinha rodado de verdade:** a arquitetura do modelo (MobileNetV2 encapsulado como sub-modelo aninhado) quebrava o `tensorflow-model-optimization` com erro de execução; o notebook original contornava isso usando `acc_ptq + 0,3 p.p.` como valor fixo em vez de medir a acurácia do QAT. Corrigido nesta auditoria (arquitetura achatada + anotação seletiva de camadas — ver comentários em `src/quantize.py` e na célula de QAT do notebook 02). Com o QAT rodando de fato, o resultado da amostra reduzida mostrou o INT8 QAT com desempenho *pior* que o PTQ (58,5% vs. 82,7% de acurácia) — o oposto do que o artigo N2 relata para o dataset completo. Isso não invalida necessariamente o resultado do artigo (mais dados/épocas podem mudar o quadro), mas significa que **a superioridade do QAT sobre o PTQ, citada no artigo N2, nunca foi de fato verificada** até esta auditoria — vale re-executar o QAT no dataset completo antes da entrega final para confirmar se os 96,1%/72ms do artigo se sustentam.
- **Dependência de conectividade:** mesmo o LoRaWAN, por ter maior alcance que Wi-Fi/4G convencional, ainda depende de um gateway próximo; em regiões muito remotas do interior, mesmo a transmissão de alertas compactos pode não ser viável sem infraestrutura adicional (repetidores, satélite).

---

## 📄 Relatórios

| Versão | Arquivo | Descrição |
|--------|---------|-----------|
| N2 (final) | [`docs/Artigo_N2_EdgeAI_PlantDisease.pdf`](docs/Artigo_N2_EdgeAI_PlantDisease.pdf) | Relatório completo: metodologia, resultados medidos, discussão ética, conclusão. Fonte oficial dos números deste README. |
| N1 (parcial) | [`docs/relatorio_parcial/Artigo_N1_proposta.pdf`](docs/relatorio_parcial/Artigo_N1_proposta.pdf) | Proposta e resultados **esperados** (não medidos) — mantido por histórico da disciplina. |

Ver [`docs/README.md`](docs/README.md) para uma pendência conhecida (URL do repositório citada nos PDFs ainda não atualizada na fonte).

---

## 🎥 Vídeo de Demonstração

**[TODO: adicionar link do vídeo antes da entrega final]** — o artigo N2 menciona um vídeo de demonstração no YouTube, mas o link ainda não foi incluído nos documentos nem informado à equipe até o momento desta auditoria.

---

## 📚 Como Citar Este Trabalho

Metadados de citação estruturados estão em [`CITATION.cff`](CITATION.cff) (formato [Citation File Format](https://citation-file-format.github.io/), reconhecido nativamente pelo GitHub — use o botão "Cite this repository" na página do repositório).

Citação manual:

```
VIEIRA DE SOUSA, G.; RAINHO GERALDO, G. Edge AI para
Detecção de Estresse Hídrico e Pragas com Redução de Overhead de Rede.
Universidade Presbiteriana Mackenzie, 2026.
Disponível em: https://github.com/vieira-security/Edge-AI-Plant-Disease
```

---

## 📅 Histórico de Atualizações

| Data | Autor | Descrição |
|------|-------|-----------|
| 22/05/2026 | Equipe | Criação do repositório e estrutura inicial |
| 22/05/2026 | Equipe | Adição do notebook de análise exploratória (N1) |
| 22/05/2026 | Equipe | Adição do relatório N1 e descrição do dataset |
| 28/05/2026 | Guilherme Rainho | Adição do notebook de treinamento e quantização (N2) |
| 28/05/2026 | Gabriel Vieira | Implementação de QAT e benchmark de latência |
| 28/05/2026 | Equipe | Relatório N2 completo e atualização do README |
| 13/08/2026 | Auditoria do repositório | URL canônica unificada, LICENSE corrigido, N1 separado do N2, notebooks executados com outputs reais (2 bugs corrigidos: contagem duplicada de imagens e acurácia QAT fabricada), `src/`, testes, CI, `requirements*.txt` e `CITATION.cff` criados |
