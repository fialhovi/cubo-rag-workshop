# Databricks notebook source
# MAGIC %md
# MAGIC # Demo: Processo de Embedding no Databricks
# MAGIC
# MAGIC Pipeline básico de processamento de texto para gerar embeddings, comparando **três textos** em cada etapa:
# MAGIC
# MAGIC - **Texto 1**: frase de referência
# MAGIC - **Texto 2**: mesmo significado do Texto 1, palavras diferentes (sinônimos)
# MAGIC - **Texto 3**: assunto totalmente diferente
# MAGIC
# MAGIC Etapas:
# MAGIC
# MAGIC 1. **Tokenização** - quebra do texto em tokens
# MAGIC 2. **Reconstrução** - volta dos tokens para texto
# MAGIC 3. **Chunking** - divisão do texto em pedaços
# MAGIC 4. **Embedding** - vetorização do texto + comparação de similaridade
# MAGIC
# MAGIC Modelo utilizado: `databricks-qwen3-embedding-0-6b` (Foundation Model API — multilíngue, baseado em Qwen3 0.6B).

# COMMAND ----------

import subprocess, sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet",
     "tokenizers", "langchain-text-splitters", "mlflow"],
    capture_output=True, check=True,
)

print("Dependências instaladas.")
dbutils.library.restartPython()

# COMMAND ----------

# Configurações para suprimir avisos cosméticos (HF Hub, progress bars, etc.)
import os, warnings, logging

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Textos de entrada
# MAGIC
# MAGIC Definimos três frases:
# MAGIC - **Texto 1** e **Texto 2** têm o **mesmo significado** com palavras diferentes.
# MAGIC - **Texto 3** trata de um assunto **completamente diferente** (finanças).

# COMMAND ----------

texto_1 = "O cachorro atravessou a rua"
texto_2 = "O cão cruzou a via"
texto_3 = "A bolsa de valores fechou em alta hoje"

print(f"Texto 1: {texto_1}")
print(f"Texto 2: {texto_2}")
print(f"Texto 3: {texto_3}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Tokenização
# MAGIC
# MAGIC Quebra o texto em unidades menores (tokens). Usamos a biblioteca `tokenizers` diretamente
# MAGIC (mais leve que `transformers`, sem dependência de PyTorch) carregando o tokenizer do
# MAGIC modelo **Qwen3 Embedding 0.6B** — o mesmo do endpoint `databricks-qwen3-embedding-0-6b`.

# COMMAND ----------

from tokenizers import Tokenizer

tok = Tokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")

def tokenize(texto):
    encoded = tok.encode(texto)
    return encoded.tokens, encoded.ids

tokens_1, ids_1 = tokenize(texto_1)
tokens_2, ids_2 = tokenize(texto_2)
tokens_3, ids_3 = tokenize(texto_3)

print(f"Texto 1: {texto_1}")
print(f"  Tokens ({len(tokens_1)}): {tokens_1}")
print(f"  Token IDs: {ids_1}")
print()
print(f"Texto 2: {texto_2}")
print(f"  Tokens ({len(tokens_2)}): {tokens_2}")
print(f"  Token IDs: {ids_2}")
print()
print(f"Texto 3: {texto_3}")
print(f"  Tokens ({len(tokens_3)}): {tokens_3}")
print(f"  Token IDs: {ids_3}")

# Sobreposição de tokens (Jaccard) sempre comparando contra o Texto 1
def jaccard(a, b):
    sa, sb = set(a), set(b)
    return (len(sa & sb) / len(sa | sb)) if (sa | sb) else 0, sorted(sa & sb)

j_12, comum_12 = jaccard(tokens_1, tokens_2)
j_13, comum_13 = jaccard(tokens_1, tokens_3)

print()
print("=== Comparação de tokens (Jaccard) ===")
print(f"Texto 1 vs Texto 2 -> {j_12:.2%}  | tokens em comum: {comum_12}")
print(f"Texto 1 vs Texto 3 -> {j_13:.2%}  | tokens em comum: {comum_13}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Reconstrução do texto a partir dos tokens
# MAGIC
# MAGIC Tokenização é uma operação **reversível**: a partir dos `token_ids` remontamos a frase original.

# COMMAND ----------

reconstruido_1 = tok.decode(ids_1)
reconstruido_2 = tok.decode(ids_2)
reconstruido_3 = tok.decode(ids_3)

print(f"Original 1     : {texto_1}")
print(f"Reconstruído 1 : {reconstruido_1}")
print()
print(f"Original 2     : {texto_2}")
print(f"Reconstruído 2 : {reconstruido_2}")
print()
print(f"Original 3     : {texto_3}")
print(f"Reconstruído 3 : {reconstruido_3}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Chunking
# MAGIC
# MAGIC Em pipelines de RAG dividimos textos longos em **chunks** para que cada pedaço caiba no
# MAGIC contexto do modelo de embedding. Usamos o `RecursiveCharacterTextSplitter` do LangChain —
# MAGIC o splitter mais usado na prática.

# COMMAND ----------

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=15,
    chunk_overlap=3,
    separators=[" ", ""],
)

chunks_1 = splitter.split_text(texto_1)
chunks_2 = splitter.split_text(texto_2)
chunks_3 = splitter.split_text(texto_3)

print(f"Texto 1: {texto_1}")
print(f"  {len(chunks_1)} chunks: {chunks_1}")
print()
print(f"Texto 2: {texto_2}")
print(f"  {len(chunks_2)} chunks: {chunks_2}")
print()
print(f"Texto 3: {texto_3}")
print(f"  {len(chunks_3)} chunks: {chunks_3}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Embedding + Similaridade
# MAGIC
# MAGIC Chamamos o endpoint **`databricks-qwen3-embedding-0-6b`** (multilíngue) para gerar os
# MAGIC vetores das três frases e calculamos a **similaridade de cosseno** comparando o
# MAGIC **Texto 1** contra os outros dois.
# MAGIC
# MAGIC > **Expectativa:**
# MAGIC > - Texto 1 vs Texto 2 (mesmo significado) → similaridade **alta**
# MAGIC > - Texto 1 vs Texto 3 (assunto diferente) → similaridade **baixa**

# COMMAND ----------

from mlflow.deployments import get_deploy_client
import numpy as np

client = get_deploy_client("databricks")

response = client.predict(
    endpoint="databricks-qwen3-embedding-0-6b",
    inputs={"input": [texto_1, texto_2, texto_3]},
)

emb_1 = np.array(response["data"][0]["embedding"])
emb_2 = np.array(response["data"][1]["embedding"])
emb_3 = np.array(response["data"][2]["embedding"])

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

sim_12 = cosine_similarity(emb_1, emb_2)
sim_13 = cosine_similarity(emb_1, emb_3)

print(f"Texto 1: {texto_1}")
print(f"Texto 2: {texto_2}")
print(f"Texto 3: {texto_3}")
print()
print(f"Dimensão dos embeddings: {len(emb_1)}")
print()
print("=== Similaridade de cosseno (referência: Texto 1) ===")
print(f"Texto 1 vs Texto 2 (mesmo significado)  -> {sim_12:.4f}")
print(f"Texto 1 vs Texto 3 (assunto diferente)  -> {sim_13:.4f}")
print()
print("(1.0 = idênticos, 0.0 = não relacionados, -1.0 = opostos)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Resumo comparativo
# MAGIC
# MAGIC Quanto mais similar é o **significado**, maior a similaridade do embedding —
# MAGIC mesmo quando os tokens não se parecem.

# COMMAND ----------

import pandas as pd

resumo = pd.DataFrame([
    {
        "comparacao": "Texto 1 vs Texto 2 (sinônimos)",
        "jaccard_tokens": f"{j_12:.2%}",
        "cosseno_embedding": round(sim_12, 4),
    },
    {
        "comparacao": "Texto 1 vs Texto 3 (assunto diferente)",
        "jaccard_tokens": f"{j_13:.2%}",
        "cosseno_embedding": round(sim_13, 4),
    },
])

display(resumo)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Comparação visual dos embeddings (vetores lado a lado)

# COMMAND ----------

df_comparacao = pd.DataFrame({
    "posicao": range(len(emb_1)),
    "emb_texto_1": emb_1,
    "emb_texto_2": emb_2,
    "emb_texto_3": emb_3,
})

display(df_comparacao)