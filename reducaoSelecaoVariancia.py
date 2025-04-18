import pandas as pd
import os

def reduzir_features(input_csv, output_csv, n_features=100, coluna_alvo='class'):
    print(f"\n🔍 Processando: {input_csv}")
    df = pd.read_csv(input_csv)

    # Limpar nomes de colunas
    df.columns = [col.strip() for col in df.columns if not col.startswith('Unnamed')]

    # Verifica se a coluna class existe
    if coluna_alvo not in df.columns:
        raise ValueError(f"A coluna '{coluna_alvo}' não foi encontrada.")

    # Remove linhas sem valor na coluna class
    df = df.dropna(subset=[coluna_alvo])

    # Remove colunas com apenas um valor (ex: tudo 0)
    nunique = df.nunique()
    colunas_validas = nunique[nunique > 1].index.tolist()

    df = df[colunas_validas]

    # Separar X e y
    y = df[coluna_alvo]
    X = df.drop(columns=[coluna_alvo])

    # Calcular variância de cada coluna
    variancias = X.var().sort_values(ascending=False)

    # Selecionar top N features mais variáveis
    top_features = variancias.head(n_features).index.tolist()

    # Combinar X reduzido + y
    df_reduzido = pd.concat([X[top_features], y.reset_index(drop=True)], axis=1)

    # Salvar resultado
    df_reduzido.to_csv(output_csv, index=False)
    print(f"✅ Salvou: {output_csv} ({df_reduzido.shape[1]} colunas totais, incluindo class)")


base_path = r"C:\Users\lukas\MalSinGen"
datasets = ["androcrawl.csv", "defensedroid_apicalls_closeness.csv"]

for nome in datasets:
    entrada = os.path.join(base_path, nome)
    saida = os.path.join(base_path, nome.replace(".csv", "_reduzido.csv"))
    reduzir_features(entrada, saida)
