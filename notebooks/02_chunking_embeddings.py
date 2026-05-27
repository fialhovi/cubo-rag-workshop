# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 1 — Chunking & Embeddings
# MAGIC
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | 🎯 **Objetivo** | Comparar 2 estratégias de chunking (fixed vs recursive) e ver na prática por que recursive ganha |
# MAGIC | ⏱ **Tempo** | 25 min |
# MAGIC
# MAGIC **Conceitos abordados:**
# MAGIC - Chunking **fixed-size** (corta a cada N caracteres, ignora estrutura)
# MAGIC - Chunking **recursive** (respeita boundaries: parágrafo → frase → palavra) ← *default em produção*
# MAGIC - Comparação lado-a-lado: olhe o **conteúdo** dos chunks, não só os números
# MAGIC
# MAGIC **Padrão Spark:** chunking aplicado via **UDF + `posexplode`** (1 doc → N chunks) e tudo persistido em Delta.

# COMMAND ----------

# MAGIC %pip install -q langchain-text-splitters
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
# MAGIC Corta a cada **N caracteres**, sem olhar o conteúdo. Pra deixar a diferença bem visível, vamos usar `CHUNK_SIZE=180` **sem overlap** — assim os cortes caem no meio de palavras e frases.
# MAGIC
# MAGIC **Pattern Spark:** UDF que recebe 1 string e devolve um `Array[String]`, depois `posexplode` pra 1 linha por chunk.

# COMMAND ----------

FIXED_CHUNK_SIZE = 180   # propositalmente pequeno pra mostrar quebras feias

@F.udf(returnType=ArrayType(StringType()))
def fixed_chunk_udf(text: str):
    return [text[i:i + FIXED_CHUNK_SIZE] for i in range(0, len(text), FIXED_CHUNK_SIZE)]

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
display(fixed_df.select("article_id", "chunk_idx", "chunk", "chunk_len").limit(8))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Estratégia B — Recursive
# MAGIC
# MAGIC `RecursiveCharacterTextSplitter` tenta separadores em ordem: `\n\n`, `\n`, `. `, `, `, ` `. **Respeita boundaries** — só corta no meio da palavra como último recurso.
# MAGIC
# MAGIC Usamos `chunk_size=400` (mais que o dobro do fixed) com `overlap=80` pra preservar contexto entre chunks.

# COMMAND ----------

RECURSIVE_CHUNK_SIZE = 400
RECURSIVE_OVERLAP = 80

@F.udf(returnType=ArrayType(StringType()))
def recursive_chunk_udf(text: str):
    # Import dentro do UDF — código roda no worker
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RECURSIVE_CHUNK_SIZE,
        chunk_overlap=RECURSIVE_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    )
    return splitter.split_text(text)

recursive_df = explode_chunks(source_df, recursive_chunk_udf, "recursive")
print(f"Recursive: {recursive_df.count()} chunks")
display(recursive_df.select("article_id", "chunk_idx", "chunk", "chunk_len").limit(8))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Comparação numérica

# COMMAND ----------

all_chunks = (
    fixed_df.select("strategy", "chunk_len")
    .unionByName(recursive_df.select("strategy", "chunk_len"))
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
# MAGIC ## 4. 🔍 Comparação visual — mesmo artigo, dois cortes
# MAGIC
# MAGIC Pegamos o artigo `faq-002` (Política de devolução) e mostramos como cada estratégia partiu o mesmo texto.
# MAGIC
# MAGIC **O que observar:**
# MAGIC - **Fixed** corta no meio de palavras (ex: `...documen`, `to seguro...`) e no meio de frases — chunk começa/termina sem contexto.
# MAGIC - **Recursive** termina em ponto final, parágrafo ou vírgula — cada chunk é uma unidade legível.

# COMMAND ----------

TARGET = "faq-002"

print("=" * 80)
print(f"📄 ESTRATÉGIA A (FIXED) — chunk_size={FIXED_CHUNK_SIZE}, sem overlap")
print("=" * 80)
for row in fixed_df.filter(F.col("article_id") == TARGET).orderBy("chunk_idx").collect():
    print(f"\n--- Chunk {row.chunk_idx} ({row.chunk_len} chars) ---")
    print(row.chunk)

print("\n" + "=" * 80)
print(f"📄 ESTRATÉGIA B (RECURSIVE) — chunk_size={RECURSIVE_CHUNK_SIZE}, overlap={RECURSIVE_OVERLAP}")
print("=" * 80)
for row in recursive_df.filter(F.col("article_id") == TARGET).orderBy("chunk_idx").collect():
    print(f"\n--- Chunk {row.chunk_idx} ({row.chunk_len} chars) ---")
    print(row.chunk)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Trade-off observado
# MAGIC
# MAGIC | Aspecto | Fixed | Recursive ⭐ |
# MAGIC |---|---|---|
# MAGIC | Quebra palavras | ❌ Sim | ✅ Não (último recurso) |
# MAGIC | Respeita parágrafo / frase | ❌ Não | ✅ Sim |
# MAGIC | Tamanho dos chunks | Uniforme | Variável (até `chunk_size`) |
# MAGIC | Custo computacional | Trivial | Trivial |
# MAGIC | Qualidade do embedding | Pior (frase quebrada) | Melhor (unidade semântica) |
# MAGIC | Quando usar | OCR | **Default produção** (docs, FAQ, KB) |
# MAGIC
# MAGIC **Conclusão:** Recursive ganha quase sempre. É o default que recomendamos pra workshop e pra produção.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Persiste os chunks **recursive** (default escolhido)
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
# MAGIC ➡️ Próximo: **03_vector_search** — criar o índice e fazer retrieval real.