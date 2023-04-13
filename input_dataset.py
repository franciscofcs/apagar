def leitura_dataset():
    """# Custom Pipeline - V3 [OFICIAL] - Ajustando aos dados de alunos do IFMA

    This post will use a classic regression example — predicting house prices to demonstrate 4 practical steps in using Pipeline, which are how to:

    <ol>
    <li>Structure a workflow systematically before writing any pipeline code</li>
    <li>Create custom transformers in Pipeline</li>
    <li>Apply modular approach when building pipeline</li>
    <li>Use same pipeline with different models to evaluate models’ performance quickly</li>
    """

    # from google.colab import drive
    # drive.mount('/content/drive')

    """## Imports"""

    # !pip install feature_engine

    import pandas as pd
    # import numpy as np

    import missingno as msno

    # from sklearn.base import BaseEstimator, TransformerMixin
    #
    # from sklearn.base import BaseEstimator, TransformerMixin
    # from sklearn.impute import SimpleImputer
    # from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
    # from sklearn.model_selection import train_test_split
    # from sklearn.pipeline import Pipeline, FeatureUnion
    # from sklearn.linear_model import LinearRegression
    # from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    # from sklearn.model_selection import KFold, cross_val_score
    # from feature_engine import encoding
    #
    # # ampliando as funcionalidades
    # from sklearn.model_selection import cross_val_score
    # import matplotlib.pyplot as plt
    # from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
    #
    # from sklearn.ensemble import RandomForestClassifier
    # from sklearn.metrics import plot_roc_curve

    """## Conhecendo os dados de alunos do IFMA"""

    # dataset url
    # 'https://www.kaggle.com/c/house-prices-advanced-regression-techniques'

    # train = pd.read_csv('./drive/MyDrive/DOUTORADO UFMA/ESTUDO DIRIGIDO I/data/exemplos/house-prices-advanced-regression-techniques/train.csv', index_col ='Id')
    # test = pd.read_csv('./drive/MyDrive/DOUTORADO UFMA/ESTUDO DIRIGIDO I/data/exemplos/house-prices-advanced-regression-techniques/test.csv', index_col ='Id')

    df = pd.read_csv(
        "./drive/MyDrive/DOUTORADO UFMA/ESTUDO DIRIGIDO I/data/suap/23062022/dados_caracterizacaosocial.csv")

    # train = pd.read_csv("house-prices-advanced-regression-techniques/train.csv", index_col ='Id')
    # test = pd.read_csv("house-prices-advanced-regression-techniques/test.csv", index_col ='Id')

    # X = train.drop('SalePrice', axis = 1)
    # y = train['SalePrice']

    print(df.info())
    df.shape

    # Transformando a variável-alvo
    query = ['Concluído', 'Formado']
    formados = df[df['situacao'].isin(query)]
    formados.replace(['Concluído', 'Formado'], 1, inplace=True)  # 1 - formado

    query = ['Evasão', 'Cancelado', 'Cancelamento Compulsório', 'Jubilado', 'Transferido Externo']
    evadidos = df[df['situacao'].isin(query)]

    evadidos.replace(['Evasão', 'Cancelado', 'Cancelamento Compulsório', 'Jubilado', 'Transferido Externo'], 0,
                     inplace=True)  # 0 - evadido

    df = pd.concat([formados, evadidos])
    df.situacao.value_counts()

    print(df.info())
    df.shape

    filtro1 = df['modalidade'] == 'Concomitante'  # 1,5 anos para conclusão
    filtro2 = df['modalidade'] == 'Integrado'  # 3 anos para conclusão
    filtro3 = df['modalidade'] == 'Integrado EJA'  # 3 anos para conclusão
    filtro4 = df['modalidade'] == 'Subsequente'  # 1,5 anos para conclusão

    df = df[filtro1 | filtro2 | filtro3 | filtro4]
    print(df.info())
    df.shape

    ano_conclusao = pd.to_datetime(df['dataconclusao'], errors='coerce')
    df['ano_conclusao'] = ano_conclusao.dt.year

    # Preenchendo dados ausentes com 0
    # df['ano_conclusao'].fillna(df['ano_conclusao'].median(), inplace=True)
    # df['ano_conclusao'] = df['ano_conclusao'].apply(int)

    # Removendo linhas com ano de conclusão superior a 2022. (Foram removidas 3 linhas)
    # df = dataset[df['ano_conclusao']<2023]

    # Removendo coluna (s)
    df.drop('dataconclusao', axis=1, inplace=True)

    """### Criação da coluna 'pais_escolarizados'

    Criação da coluna **'pais_escolarizados'**, a partir de 3 colunas do dataset: ***'companhia_domiciliar', 'mae_nivel_escolaridade', 'pai_nivel_escolaridade'***. O objetivo é verificar a relevância desta variável na predição da variável alvo.
    """

    ### Criação da coluna 'pais_escolarizados'

    # Criação da coluna **'pais_escolarizados'**, a partir de 3 colunas do df: ***'companhia_domiciliar', 'mae_nivel_escolaridade', 'pai_nivel_escolaridade'***. O objetivo é verificar a relevância desta variável na predição da variável alvo.

    df["pais_escolarizados"] = 1  # escolarizado

    mae = df.companhia_domiciliar == 'Mãe'
    pai = df.companhia_domiciliar == 'Pai'
    pais = df.companhia_domiciliar == 'Pais'
    mae_escolaridade = df.mae_nivel_escolaridade == "Não Estudou"
    pai_escolaridade = df.pai_nivel_escolaridade == "Não Estudou"
    escola = df[(mae | pai | pais) & (mae_escolaridade & pai_escolaridade)]
    escola.pais_escolarizados = 0  # não escolarizado

    nao_informado_mae = df.mae_nivel_escolaridade == "Não sei informar"
    nao_informado_pai = df.pai_nivel_escolaridade == "Não sei informar"
    nao_conhece_pai = df.pai_nivel_escolaridade == "Não conhece"
    nao_informado = df[nao_informado_mae | nao_informado_pai | nao_conhece_pai]
    nao_informado.pais_escolarizados = 2  # não soube informar

    df.pais_escolarizados.update(pd.Series(escola.pais_escolarizados, index=escola.pais_escolarizados.index.tolist()))
    df.pais_escolarizados.update(
        pd.Series(nao_informado.pais_escolarizados, index=nao_informado.pais_escolarizados.index.tolist()))

    print(df.info())
    df.shape

    """## Separaçãodo dataframe"""

    print(df.info())
    df.shape

    df['pais_escolarizados'].value_counts()

    df.isnull().sum().sort_values(ascending=False) * 100 / len(df)

    # create train & validation set
    # As colunas removidas por não se aplicarem ao contexto do dataset de treinamento
    # 'situacao' é a classe-alvo
    # as demais colunas ou são quase 100% nulas ou irrelevantes para predição ('ira' é um caso à parte, mas, excluída, no momento).
    # X = df.drop(['alunoid','ira','pontuacao_seletivo','razao_ausencia_educacional','exclusivo_rede_publica','forma_acesso_seletivo','dataconclusao','situacao'], axis = 1).copy()
    X = df.drop(['alunoid', 'ira', 'pontuacao_seletivo', 'razao_ausencia_educacional', 'exclusivo_rede_publica',
                 'pontuacao_seletivo', 'situacao'], axis=1).copy()
    # X = df.drop(['alunoid','ira','pontuacao_seletivo','razao_ausencia_educacional','exclusivo_rede_publica','situacao'], axis = 1).copy()
    # ['alunoid','ira','pontuacao_seletivo','razao_ausencia_educacional','exclusivo_rede_publica','situacao']
    y = df['situacao']

    X.shape

    y.value_counts()

    # check pattern of missingness
    msno.matrix(X.iloc[:, :10])
    msno.matrix(X.iloc[:, 10:])

    msno.bar(X)

    msno.heatmap(X)

    """## Step 1: Structure a workflow systematically before writing any pipeline code

    In this experiments, the preprocessing steps are categorised into 3 following sections:-

    <ol>
      <li> Filtering
        <ul>
          <li>Drop columns with more than 50% of missingness</li>
          <li>Select numerical / category features depending on the pipeline flow</li>
        </ul>
      </li>
      <li> Missingness Treatment
        <ul>
          <li>Median imputation for numerical features</li>
          <li>Most frequent imputation for categorical features</li>
        </ul>
      </li>
      <li> Feature Engineering
        <ul>
          <li>Create practical and predictive features using domain knowledge (e.g. adding all bathrooms). Full details check on this link.</li>
          <li>Encode all categorical features’ items with less than 20% frequency as 'rare'.</li>
          <li>One hot encoding all categories features at last step of preprocessing in categorical pipeline.</li>
        </ul>
      </li>
    </ol>
    """