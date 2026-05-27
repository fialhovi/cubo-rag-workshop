# Databricks notebook source
# MAGIC %md
# MAGIC # Configuração centralizada
# MAGIC
# MAGIC Não precisa editar nada — o `SCHEMA_NAME` é derivado automaticamente do seu usuário corrente
# MAGIC e o catálogo `workshop_databricks` já tá pronto. Todos os notebooks importam esse arquivo via `%run`.

# COMMAND ----------

# ============================================================================
# Schema derivado do seu e-mail: "fulano.silva@empresa.com" → "fulano_silva_rag"
# ============================================================================
_CURRENT_USER = spark.sql("SELECT current_user()").collect()[0][0]
_USER_HANDLE = _CURRENT_USER.split("@")[0].replace(".", "_").replace("-", "_").lower()
SCHEMA_NAME = f"{_USER_HANDLE}_rag"

# ============================================================================
# Catálogo compartilhado do workshop — criado pelo facilitador com GRANTS pros participantes
# ============================================================================
CATALOG_NAME = "workshop_databricks"

# Endpoints Foundation Model (pré-provisionados no workspace)
EMBEDDING_MODEL = "databricks-gte-large-en"
LLM_MODEL = "databricks-meta-llama-3-3-70b-instruct"

# Vector Search — 1 endpoint por participante (cold start ~5-10min, criado no Lab 0)
VS_ENDPOINT_NAME = f"{_USER_HANDLE}_vs"
VS_INDEX_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.kb_chunks_index"

# Tabelas
SOURCE_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.kb_articles"
CHUNKS_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.kb_chunks"
EVAL_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.eval_dataset"

# MLflow Experiment — usa o usuário já descoberto acima
MLFLOW_EXPERIMENT_PATH = f"/Users/{_CURRENT_USER}/cubo_rag_workshop"

# COMMAND ----------

print(f"✅ Config carregado:")
print(f"   Usuário:  {_CURRENT_USER}")
print(f"   Catalog:  {CATALOG_NAME}")
print(f"   Schema:   {SCHEMA_NAME}")
print(f"   VS Index: {VS_INDEX_NAME}")
print(f"   LLM:      {LLM_MODEL}")
print(f"   Embedder: {EMBEDDING_MODEL}")
