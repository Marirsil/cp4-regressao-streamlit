# Previsão de Preço de Veículos Usados

Projeto desenvolvido para o **Checkpoint 4 de Data Science & Statistical Computing - FIAP**.

O objetivo é utilizar técnicas de **Regressão Linear** para analisar características de veículos usados e desenvolver um modelo capaz de estimar o preço anunciado de um veículo.

---

# Links

**Repositório:**  
[Github](https://github.com/Marirsil/teste/edit/main/README.md)

**Aplicação Streamlit:**  
[Streamlit](https://cp4-regressao-app222756clonedrepository222756pullingcodechange.streamlit.app)

---

# Tecnologias Utilizadas

<div align="left">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="40px" height="40px" alt="Python" />
  <img width="12" />

  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" width="40px" height="40px" alt="Pandas" />
  <img width="12" />

  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" width="40px" height="40px" alt="NumPy" />
  <img width="12" />

  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/scikitlearn/scikitlearn-original.svg" width="40px" height="40px" alt="Scikit-learn" />
  <img width="12" />

  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/jupyter/jupyter-original.svg" width="40px" height="40px" alt="Google Colab / Jupyter" />
  <img width="12" />

  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/git/git-original.svg" width="40px" height="40px" alt="Git" />
  <img width="12" />

  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="40px" height="40px" alt="GitHub" />
</div>

---

# Base de Dados

Foi utilizada a base **Used Cars Database**, contendo anúncios de veículos usados publicados no Craigslist.

Fonte:

https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data

A variável resposta utilizada no projeto é:

- `price` — preço anunciado do veículo em dólares.

Entre as principais variáveis utilizadas para previsão estão:

- Ano (`year`)
- Quilometragem (`odometer`)
- Fabricante (`manufacturer`)
- Condição (`condition`)
- Combustível (`fuel`)
- Transmissão (`transmission`)
- Tração (`drive`)
- Tipo do veículo (`type`)
- Cor (`paint_color`)

---

# Modelo

Foram comparados diferentes modelos:

- Modelo de referência
- Regressão Linear Simples
- Regressão Linear Múltipla
- Regressão Polinomial

O modelo final selecionado foi a **Regressão Polinomial de Grau 2**.

Principais métricas:

| Métrica | Resultado |
|---|---:|
| MAE | US$ 5.893,57 |
| RMSE | US$ 8.400,24 |
| R² | 0,6034 |

---

# Aplicação Streamlit

Foi desenvolvida uma aplicação em **Streamlit** que permite:

- Visualizar informações da base
- Visualizar gráficos exploratórios
- Consultar as métricas do modelo
- Informar características de um veículo
- Gerar uma previsão de preço

---

# Estrutura do Projeto

```text
cp4-regressao-streamlit/
│
├── app.py
├── notebook.ipynb
├── requirements.txt
├── README.md
│
├── dados/
│   └── base_tratada.csv
│
└── modelo/
    └── modelo.pkl
```

---

# Como Executar

Clone o repositório:

```bash
git clone LINK-DO-REPOSITORIO
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute a aplicação:

```bash
streamlit run app.py
```

O notebook também pode ser aberto e executado pelo **Google Colab** ou Jupyter Notebook.

---

# Limitações

- Os dados representam preços anunciados e não necessariamente valores reais de venda.
- O mercado de veículos pode sofrer alterações ao longo do tempo.
- O modelo pode apresentar erros maiores para veículos com características pouco representadas na base.
- Previsões fora dos intervalos observados nos dados podem ser menos confiáveis.

---

# Autores

- Augusto Valerio - RM:562185  
  GitHub: https://github.com/Augusto-Valerio

- Jonas Esteves França - RM:564143  
  GitHub: https://github.com/Jonas-Franca

- Mariana Silva Oliveira - RM:564241  
  GitHub: https://github.com/Marirsil

- Pedro Marchese - RM:563339  
  GitHub: https://github.com/PedroMarchese01

- Vitor Rodrigues Tigre - RM:561746  
  GitHub: https://github.com/VitorTigre
