# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 1 — Chunking & Embeddings
# MAGIC
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | 🎯 **Objetivo** | Comparar 3 estratégias de chunking, gerar embeddings via Foundation Model API, observar trade-offs |
# MAGIC | ⏱ **Tempo** | 25 min |
# MAGIC
# MAGIC **Conceitos abordados:**
# MAGIC - Chunking **fixed-size** (mais simples, ignora estrutura)
# MAGIC - Chunking **recursive** (respeita boundaries: parágrafo → frase → palavra) ← *default em produção*
# MAGIC - Chunking **semantic** (split por similaridade de embedding) ← *caro mas poderoso*
# MAGIC - Embeddings via **`databricks-gte-large-en`**
# MAGIC
# MAGIC **Padrão Spark:** chunking aplicado via **UDF + `posexplode`** (1 doc → N chunks) e tudo persistido em Delta.

# COMMAND ----------

# MAGIC %pip install -q langchain-text-splitters mlflow>=2.18
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../config/00_config

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

source_df = spark.table(SOURCE_TABLE)
print(f"Carregados {source_df.count()} artigos.")
display(source_df.select("article_id", "category", "title"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Estratégia A — Fixed-size
# MAGIC
# MAGIC Quebra a cada N caracteres, sem olhar o conteúdo. Simples, rápido, **ruim** pra texto com estrutura — quebra frases no meio.
# MAGIC
# MAGIC **Pattern Spark:** UDF que recebe 1 string e devolve um `Array[String]`, depois `posexplode` pra 1 linha por chunk.

# COMMAND ----------

CHUNK_SIZE = 400
OVERLAP = 50

@F.udf(returnType=ArrayType(StringType()))
def fixed_chunk_udf(text: str):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - OVERLAP
    return chunks

def explode_chunks(df, chunker_udf, strategy_name):
    """1 doc → N chunks via UDF + posexplode. Retorna Spark DF tipado."""
    return (df
        .withColumn("chunks", chunker_udf("content"))
        .select(
            "article_id", "category", "title",
            F.posexplode("chunks").alias("chunk_idx", "chunk"),
        )
        .withColumn("chunk_len", F.length("chunk"))
        .withColumn("strategy", F.lit(strategy_name))
    )

fixed_df = explode_chunks(source_df, fixed_chunk_udf, "fixed")
print(f"Fixed: {fixed_df.count()} chunks")
display(fixed_df.select("article_id", "chunk_idx", "chunk", "chunk_len").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Estratégia B — Recursive
# MAGIC
# MAGIC `RecursiveCharacterTextSplitter` tenta separadores em ordem: `\n\n`, `\n`, `. `, ` `. **Respeita boundaries** sem custo extra de embeddings.

# COMMAND ----------

@F.udf(returnType=ArrayType(StringType()))
def recursive_chunk_udf(text: str):
    # Import dentro do UDF — código roda no worker
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    )
    return splitter.split_text(text)

recursive_df = explode_chunks(source_df, recursive_chunk_udf, "recursive")
print(f"Recursive: {recursive_df.count()} chunks")
display(recursive_df.select("article_id", "chunk_idx", "chunk", "chunk_len").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Estratégia C — Semantic
# MAGIC
# MAGIC Embedda cada frase, junta frases consecutivas enquanto a similaridade for alta. **Quebra onde o tópico muda.**
# MAGIC
# MAGIC > **Por que essa NÃO é UDF?** O semantic chama o endpoint de embeddings (1 call por frase). Em UDF, cada call serializa o client → lento + flaky. Como temos só 14 docs, colhemos pro driver e processamos sequencialmente. Em produção (milhares de docs), o caminho seria `pandas_udf` vetorizado.

# COMMAND ----------

from mlflow.deployments import get_deploy_client
import numpy as np
import re

deploy_client = get_deploy_client("databricks")

def embed_batch(texts: list[str]) -> list[list[float]]:
    resp = deploy_client.predict(endpoint=EMBEDDING_MODEL, inputs={"input": texts})
    return [d["embedding"] for d in resp["data"]]

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

def semantic_split(text: str, threshold: float = 0.75) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) <= 1:
        return sentences
    embs = embed_batch(sentences)
    chunks, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        if cosine(embs[i-1], embs[i]) >= threshold:
            current.append(sentences[i])
        else:
            chunks.append(" ".join(current))
            current = [sentences[i]]
    chunks.append(" ".join(current))
    return chunks

# Coleta os docs no driver, splitta, reconstrói como Spark DF
docs = source_df.collect()
semantic_rows = [
    (row.article_id, row.category, row.title, i, ch, len(ch), "semantic")
    for row in docs
    for i, ch in enumerate(semantic_split(row.content))
]
semantic_df = spark.createDataFrame(
    semantic_rows,
    schema="article_id string, category string, title string, chunk_idx int, chunk string, chunk_len int, strategy string",
)
print(f"Semantic: {semantic_df.count()} chunks")
display(semantic_df.select("article_id", "chunk_idx", "chunk", "chunk_len").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Comparação — agregação em Spark

# COMMAND ----------

all_chunks = (
    fixed_df.select("strategy", "chunk_len")
    .unionByName(recursive_df.select("strategy", "chunk_len"))
    .unionByName(semantic_df.select("strategy", "chunk_len"))
)

summary = all_chunks.groupBy("strategy").agg(
    F.count("*").alias("n_chunks"),
    F.avg("chunk_len").cast("int").alias("avg_len"),
    F.min("chunk_len").alias("min_len"),
    F.max("chunk_len").alias("max_len"),
)
display(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Trade-off observado
# MAGIC
# MAGIC | Estratégia | Pró | Contra | Use quando |
# MAGIC |---|---|---|---|
# MAGIC | **Fixed** | Trivial, determinístico | Quebra frases no meio → embedding pior | Texto homogêneo (logs, OCR) |
# MAGIC | **Recursive** | Respeita estrutura, custo zero | Pode gerar chunks muito desiguais | **Default produção** |
# MAGIC | **Semantic** | Quebra onde o tópico muda | Custa O(N) embeddings; threshold é hyperparam | Docs bem estruturados, base estável |
# MAGIC
# MAGIC ## 5. Teste de retrieval rápido
# MAGIC
# MAGIC Pergunta *"política de devolução"* embedada contra chunks recursive vs fixed → top-1 ranking.

# COMMAND ----------

query = "Quantos dias eu tenho pra devolver?"
[q_emb] = embed_batch([query])

def topk(df, k: int = 3):
    """Coleta chunks, embedda cada um, retorna top-k por similaridade — como Spark DF."""
    rows = df.select("article_id", "chunk").collect()
    chunk_embs = embed_batch([r.chunk for r in rows])
    scored = [(r.article_id, float(cosine(q_emb, e)), r.chunk) for r, e in zip(rows, chunk_embs)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return spark.createDataFrame(
        scored[:k],
        schema="article_id string, sim double, chunk string",
    )

print("🔍 TOP-3 com FIXED:")
display(topk(fixed_df))

print("🔍 TOP-3 com RECURSIVE:")
display(topk(recursive_df))

# COMMAND ----------

# MAGIC %md
# MAGIC **O que observar:** com recursive, o top-1 quase sempre é `faq-002` (devolução) e o chunk começa numa frase completa. Com fixed, mesmo quando bate o artigo certo, o chunk pode começar no meio de uma palavra — pior contexto pro LLM.
# MAGIC
# MAGIC ## 6. Persiste os chunks **recursive** (default escolhido)
# MAGIC
# MAGIC Direto da DataFrame Spark → Delta com CDF habilitado (alimenta o Vector Search Sync).

# COMMAND ----------

chunks_to_save = (recursive_df
    .withColumn("id", F.concat_ws("::", "article_id", F.col("chunk_idx").cast("string")))
    .select("id", "article_id", "category", "title", "chunk_idx", "chunk")
)

(chunks_to_save.write
    .mode("overwrite")
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable(CHUNKS_TABLE)
)

# Garante CDF caso a tabela existisse sem ele
spark.sql(f"ALTER TABLE {CHUNKS_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

print(f"✅ {chunks_to_save.count()} chunks salvos em {CHUNKS_TABLE} (CDF habilitado)")
display(spark.table(CHUNKS_TABLE).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Concluído
# MAGIC
# MAGIC ➡️ Próximo: [`02_vector_search`](./02_vector_search) — criar o índice e fazer retrieval real.
