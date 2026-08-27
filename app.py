import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# Configuração da página
# =========================================================
st.set_page_config(
    page_title="Previsão de Preço de Veículos Usados",
    # page_icon="🚗",
    layout="wide"
)


# =========================================================
# Caminhos dos arquivos
# =========================================================
CAMINHO_BASE = "dados/base_tratada.csv"
CAMINHO_MODELO = "modelo/modelo.pkl"


# =========================================================
# Carregamento com cache
# =========================================================
@st.cache_data
def carregar_dados():
    return pd.read_csv(CAMINHO_BASE)


@st.cache_resource
def carregar_modelo():
    return joblib.load(CAMINHO_MODELO)


df = carregar_dados()
modelo = carregar_modelo()


# =========================================================
# Variáveis usadas pelo modelo
# =========================================================
colunas_quantitativas = ["year", "odometer"]

colunas_categoricas = [
    "manufacturer",
    "condition",
    "cylinders",
    "fuel",
    "title_status",
    "transmission",
    "drive",
    "type",
    "paint_color",
]

colunas_modelo = colunas_quantitativas + colunas_categoricas


# =========================================================
# Recriação do mesmo conjunto de teste do notebook
# =========================================================
X = df[colunas_modelo]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

pred_test = modelo.predict(X_test)

mae = mean_absolute_error(y_test, pred_test)
rmse = np.sqrt(mean_squared_error(y_test, pred_test))
r2 = r2_score(y_test, pred_test)

residuos = y_test.to_numpy() - pred_test


# =========================================================
# Cabeçalho
# =========================================================
st.title("Previsão de Preço de Veículos Usados")

st.write(
    """
    Este projeto utiliza regressão polinomial de grau 2 para estimar o preço
    anunciado de veículos usados nos Estados Unidos a partir de características
    como ano, quilometragem, fabricante, condição, combustível, transmissão,
    tração, carroceria e cor.
    """
)

st.markdown(
    """
    **Fonte dos dados:** Used Cars Database / Craigslist Cars and Trucks Data,
    organizada por Austin Reese e disponibilizada no Kaggle.

    **Variável resposta (y):** `price` — preço anunciado do veículo em dólares (USD).

    **Variáveis explicativas (X):**
    `year`, `odometer`, `manufacturer`, `condition`, `cylinders`, `fuel`,
    `title_status`, `transmission`, `drive`, `type` e `paint_color`.
    """
)

st.divider()


# =========================================================
# 1. Exploração dos dados
# =========================================================
st.header("1. Exploração dos dados")

st.subheader("Amostra da base tratada")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("Estatísticas descritivas")
st.dataframe(
    df[["price", "year", "odometer"]].describe().round(2),
    use_container_width=True
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribuição dos preços")

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.hist(df["price"].dropna(), bins=50)
    ax1.set_title("Distribuição do preço dos veículos")
    ax1.set_xlabel("Preço (USD)")
    ax1.set_ylabel("Frequência")
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

with col2:
    st.subheader("Preço × quilometragem")

    # Amostra apenas para o gráfico ficar leve no Streamlit
    n_amostra = min(5000, len(df))
    df_grafico = df.sample(n=n_amostra, random_state=42)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.scatter(
        df_grafico["odometer"],
        df_grafico["price"],
        alpha=0.25,
        s=12
    )
    ax2.set_title("Preço vs. quilometragem")
    ax2.set_xlabel("Odômetro (milhas)")
    ax2.set_ylabel("Preço (USD)")
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

st.divider()


# =========================================================
# 2. Avaliação do modelo final
# =========================================================
st.header("2. Avaliação do modelo final")

st.write(
    """
    O modelo final é uma **regressão polinomial de grau 2**.
    As métricas abaixo são calculadas no conjunto de teste,
    separado com `test_size=0.30` e `random_state=42`,
    assim como no notebook.
    """
)

m1, m2, m3 = st.columns(3)

m1.metric("MAE", f"US$ {mae:,.2f}")
m2.metric("RMSE", f"US$ {rmse:,.2f}")
m3.metric("R²", f"{r2:.4f}")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Valores reais × previstos")

    n_plot = min(5000, len(y_test))
    idx = np.random.RandomState(42).choice(
        len(y_test),
        size=n_plot,
        replace=False
    )

    y_real_plot = y_test.to_numpy()[idx]
    y_pred_plot = pred_test[idx]

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.scatter(y_real_plot, y_pred_plot, alpha=0.25, s=12)

    minimo = min(y_real_plot.min(), y_pred_plot.min())
    maximo = max(y_real_plot.max(), y_pred_plot.max())

    ax3.plot([minimo, maximo], [minimo, maximo], linestyle="--")
    ax3.set_title("Valores reais vs. valores previstos")
    ax3.set_xlabel("Preço real (USD)")
    ax3.set_ylabel("Preço previsto (USD)")
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

with col4:
    st.subheader("Resíduos")

    residuos_plot = residuos[idx]
    previsto_plot = pred_test[idx]

    fig4, ax4 = plt.subplots(figsize=(7, 5))
    ax4.scatter(previsto_plot, residuos_plot, alpha=0.25, s=12)
    ax4.axhline(0, linestyle="--")
    ax4.set_title("Resíduos vs. valores previstos")
    ax4.set_xlabel("Preço previsto (USD)")
    ax4.set_ylabel("Resíduo: real - previsto (USD)")
    fig4.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

st.divider()


# =========================================================
# 3. Formulário para nova previsão
# =========================================================
st.header("3. Faça uma nova previsão")

st.write(
    """
    Preencha as características do veículo abaixo.
    A entrada será enviada diretamente ao mesmo pipeline utilizado no treinamento.
    """
)


def opcoes(coluna):
    valores = df[coluna].dropna().astype(str).unique().tolist()
    return sorted(valores)


year_min = int(df["year"].min())
year_max = int(df["year"].max())
odometer_min = float(df["odometer"].min())
odometer_max = float(df["odometer"].max())

year_default = int(df["year"].median())
odometer_default = float(df["odometer"].median())


with st.form("form_previsao"):
    c1, c2 = st.columns(2)

    with c1:
        year = st.number_input(
            "Ano do veículo",
            min_value=1800,
            max_value=2100,
            value=year_default,
            step=1
        )

        odometer = st.number_input(
            "Quilometragem / odômetro (milhas)",
            min_value=0.0,
            max_value=1_000_000.0,
            value=odometer_default,
            step=1000.0
        )

        manufacturer = st.selectbox(
            "Fabricante",
            opcoes("manufacturer")
        )

        condition = st.selectbox(
            "Condição",
            opcoes("condition")
        )

        cylinders = st.selectbox(
            "Cilindros",
            opcoes("cylinders")
        )

        fuel = st.selectbox(
            "Combustível",
            opcoes("fuel")
        )

    with c2:
        title_status = st.selectbox(
            "Situação do título/documento",
            opcoes("title_status")
        )

        transmission = st.selectbox(
            "Transmissão",
            opcoes("transmission")
        )

        drive = st.selectbox(
            "Tração",
            opcoes("drive")
        )

        tipo = st.selectbox(
            "Tipo de carroceria",
            opcoes("type")
        )

        paint_color = st.selectbox(
            "Cor",
            opcoes("paint_color")
        )

    enviar = st.form_submit_button("Calcular preço estimado")


if enviar:
    fora_intervalo = []

    if year < year_min or year > year_max:
        fora_intervalo.append(
            f"`year`: informado {year}, observado entre {year_min} e {year_max}"
        )

    if odometer < odometer_min or odometer > odometer_max:
        fora_intervalo.append(
            f"`odometer`: informado {odometer:,.0f}, observado entre "
            f"{odometer_min:,.0f} e {odometer_max:,.0f} milhas"
        )

    if fora_intervalo:
        st.warning(
            "Atenção: há entrada fora do intervalo observado na base. "
            "A previsão é uma extrapolação e pode ser pouco confiável.\n\n"
            + "\n\n".join(f"- {item}" for item in fora_intervalo)
        )

    novo_veiculo = pd.DataFrame([{
        "year": int(year),
        "odometer": float(odometer),
        "manufacturer": manufacturer,
        "condition": condition,
        "cylinders": cylinders,
        "fuel": fuel,
        "title_status": title_status,
        "transmission": transmission,
        "drive": drive,
        "type": tipo,
        "paint_color": paint_color,
    }])

    previsao = modelo.predict(novo_veiculo)[0]

    st.success(f"Preço estimado: US$ {previsao:,.2f}")

    st.caption(
        "A previsão representa uma estimativa estatística baseada nos dados "
        "observados e não deve ser interpretada como valor de venda garantido."
    )

    with st.expander("Ver dados enviados ao modelo"):
        st.dataframe(novo_veiculo, use_container_width=True)
