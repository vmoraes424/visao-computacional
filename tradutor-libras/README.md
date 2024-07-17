# Tradutor de Gestos de Libras usando IA

Aluno: Vinícius Cruz de Moraes

Este projeto foi desenvolvido como parte do curso de Inteligência Artificial na faculdade. O objetivo é reconhecer gestos de Libras em tempo real utilizando técnicas de visão computacional e aprendizado profundo.

## Funcionalidades

- **Captura de Vídeo**: Utiliza a câmera para capturar o gesto de Libras.
- **Reconhecimento de Mão**: Usa a biblioteca MediaPipe para detectar e acompanhar a mão em tempo real.
- **Classificação de Gestos**: Um modelo treinado classifica o gesto capturado em uma das letras do alfabeto de Libras.
- **Treinamento do Modelo**: Inclui um script para treinar o modelo usando imagens de gestos de Libras.

## Requisitos

- Python 3.x
- TensorFlow
- OpenCV
- Mediapipe

## Instalação

1. Clone este repositório:

   ```bash
   git clone https://github.com/vmoraes424/visao-computacional.git
   cd tradutor-libras
   ```

2. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

## Uso

### Executar o Reconhecimento de Gestos

Execute o script `prediction_script.py` para iniciar o reconhecimento de gestos em tempo real:

```bash
python prediction_script.py
```

Pressione 'q' para sair do programa.
