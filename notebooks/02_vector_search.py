# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 2 — Vector Search
# MAGIC
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | 🎯 **Objetivo** | Criar um Vector Search Index (Delta Sync, managed embeddings) e fazer retrieval |
# MAGIC | ⏱ **Tempo** | 20 min |
# MAGIC
# MAGIC **Conceitos abordados:**
# MAGIC - **Delta Sync Index** — UC table com CDF habilitado → Vector Search atualiza sozinho
# MAGIC - **Managed embeddings** — Databricks calcula embeddings (não precisamos passar vetores)
# MAGIC - **Hybrid search** — vector + keyword num único índice

# COMMAND ----------

# MAGIC %pip install -q databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../config/00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Aguarda o endpoint ficar ONLINE
# MAGIC
# MAGIC O `00_setup` disparou a criação do seu endpoint pessoal. Aqui esperamos ele virar `ONLINE` antes de criar o index.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vs_client = VectorSearchClient(disable_notice=True)

# Se o endpoint não existir (alguém pulou o Lab 0), cria agora
try:
    vs_client.get_endpoint(VS_ENDPOINT_NAME)
except Exception:
    print(f"⏳ Endpoint não encontrado — criando '{VS_ENDPOINT_NAME}' agora...")
    vs_client.create_endpoint(name=VS_ENDPOINT_NAME, endpoint_type="STANDARD")

# Aguarda ficar PROVISIONED/ONLINE — até 15 min
print(f"⏳ Aguardando '{VS_ENDPOINT_NAME}' ficar pronto (pode levar até 10min se for cold start)...")
vs_client.wait_for_endpoint(name=VS_ENDPOINT_NAME, timeout=900)
print(f"✅ Endpoint pronto")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Cria o Index (Delta Sync + managed embeddings)
# MAGIC
# MAGIC Em **managed embeddings**, você diz qual coluna textual usar + qual endpoint de embedding — Databricks faz o resto. Sem essa abstração, você embeddaria manualmente toda inserção/update.

# COMMAND ----------

try:
    existing = vs_client.get_index(endpoint_name=VS_ENDPOINT_NAME, index_name=VS_INDEX_NAME)
    print(f"⚠️  Index {VS_INDEX_NAME} já existe. Disparando sync pra pegar mudanças na source...")
    existing.sync()
except Exception:
    print(f"⏳ Criando index {VS_INDEX_NAME}...")
    vs_client.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT_NAME,
        source_table_name=CHUNKS_TABLE,
        index_name=VS_INDEX_NAME,
        primary_key="id",
        pipeline_type="TRIGGERED",   # snapshot batch — mais barato. Use CONTINUOUS pra streaming.
        embedding_source_column="chunk",
        embedding_model_endpoint_name=EMBEDDING_MODEL,
    )
    print("✅ Index criado. Vai levar 2-5min pra ficar READY.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Espera o index ficar pronto

# COMMAND ----------

import time

idx = vs_client.get_index(endpoint_name=VS_ENDPOINT_NAME, index_name=VS_INDEX_NAME)

for i in range(60):
    status = idx.describe().get("status", {})
    state = status.get("detailed_state", "")
    ready = status.get("ready", False)
    print(f"[{i*5}s] state={state} ready={ready}")
    if ready and "ONLINE" in state:
        print("✅ Index online")
        break
    time.sleep(5)
else:
    raise TimeoutError("Index não ficou pronto em 5min. Verifique no UI.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Faz queries
# MAGIC
# MAGIC `similarity_search` aceita `query_text` (busca semântica), `query_vector` (você passa o embedding) ou ambos (hybrid).

# COMMAND ----------

def search(query: str, k: int = 5, query_type: str = "ANN", filters: dict | None = None):
    """
    query_type: ANN = puramente semântico; HYBRID = semântico + keyword (BM25-like).
    filters: ex. {"category": "faq"} ou {"category": ["faq", "produto"]}.
    """
    return idx.similarity_search(
        query_text=query,
        columns=["id", "article_id", "category", "title", "chunk"],
        num_results=k,
        query_type=query_type,
        filters=filters or {},
    )

def show_results(results: dict):
    col_names = [c["name"] for c in results["manifest"]["columns"]]
    rows = results["result"]["data_array"]
    import pandas as pd
    df = pd.DataFrame(rows, columns=col_names)
    display(df)

# COMMAND ----------

# Query 1: pergunta natural
print("🔍 'política de devolução' (semântico)")
show_results(search("política de devolução"))

# COMMAND ----------

# Query 2: jargão / código (onde semântico pode falhar)
print("🔍 'SKU AUR-CASE-IP15-001' (semântico)")
show_results(search("SKU AUR-CASE-IP15-001"))

print("🔍 'SKU AUR-CASE-IP15-001' (HYBRID)")
show_results(search("SKU AUR-CASE-IP15-001", query_type="HYBRID"))

# COMMAND ----------

# MAGIC %md
# MAGIC **Observe:** com query de SKU/código, o **HYBRID** quase sempre ranqueia o `prod-001` em #1 (keyword match exato), enquanto o semântico puro pode ranquear chunks parecidos por contexto mas sem o SKU.
# MAGIC
# MAGIC ## 4. Filtro de metadata

# COMMAND ----------

print("🔍 'garantia' filtrado por category=politica")
show_results(search("garantia", filters={"category": "politica"}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Concluído
# MAGIC
# MAGIC O index está vivo e disponível em `{VS_INDEX_NAME}`. Daqui em diante, qualquer notebook (ou agent deployado) pode consultar.
# MAGIC
# MAGIC ➡️ Próximo: [`03_rag_agent`](./03_rag_agent) — montar o agent que consome esse retrieval.
