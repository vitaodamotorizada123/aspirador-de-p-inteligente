# aspirador-de-p-inteligente

https://colab.research.google.com/drive/1cIvjzUv8bDjBKD4RCne25QaFMxyMlfhU?usp=sharing
https://colab.research.google.com/drive/1cIvjzUv8bDjBKD4RCne25QaFMxyMlfhU?usp=sharing
https://colab.research.google.com/drive/1cIvjzUv8bDjBKD4RCne25QaFMxyMlfhU?usp=sharing


# 🤖 Aspirador Inteligente com Perceptron

Este projeto implementa um perceptron multicamadas para controlar um aspirador de pó inteligente.  
O objetivo é prever **potência de aspiração (1 a 3)** e **velocidade de movimento (1 a 5)**, a partir das seguintes entradas:

- **Piso**: numérico (`1=carpete`, `2=cerâmica`, `3=madeira`)  
- **Poeira**: nível de sujeira (`0 a 9`)  
- **Obstáculos**: proximidade de obstáculos (`0 a 4`, onde 0 = livre e 4 = muito próximo)  

---

## 📌 Estrutura do Problema

### Entradas (normalizadas)
- Piso → dividido por 3  
- Poeira → dividido por 9  
- Obstáculos → dividido por 4  

### Saídas
- **Potência**: codificada em **3 neurônios (one-hot)**  
- **Velocidade**: codificada em **5 neurônios (one-hot)**  

Exemplo de dado de treino:

```json
{"piso": 2, "poeira": 2, "obstaculos": 0, "potencia": 1, "velocidade": 3}
