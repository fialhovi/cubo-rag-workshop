# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 3 — RAG Agent
# MAGIC
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | 🎯 **Objetivo** | Montar o agent que consome o Vector Search + LLM + variante com reranking |
# MAGIC | ⏱ **Tempo** | 25 min |
# MAGIC
# MAGIC **Conceitos abordados:**
# MAGIC - Função RAG com `@mlflow.trace` (cada span aparece na UI de tracing)
# MAGIC - **Reranking via LLM judge** — re-ordena top-K do retrieval por relevância
# MAGIC - Comparação rápida: baseline vs reranked

# COMMAND ----------

# MAGIC %pip install -q databricks-vectorsearch mlflow>=2.18 databricks-langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../config/00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Conecta no index + LLM

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from databricks_langchain import ChatDatabricks
import mlflow
import json

mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)
mlflow.langchain.autolog()

EXPERIMENT_ID = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_PATH).experiment_id
WORKSPACE_URL = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
EXP_URL = f"{WORKSPACE_URL}/ml/experiments/{EXPERIMENT_ID}"

displayHTML(f"""
<div style='font-family: sans-serif; padding: 12px; background: #eef6ff; border-left: 4px solid #1976d2; border-radius: 4px;'>
  <b>🧵 Traces vão pra cá:</b> <a href="{EXP_URL}/traces" target="_blank">aba Traces do experimento</a><br>
  <span style='color: #555; font-size: 13px;'>Cada chamada com <code>@mlflow.trace</code> abaixo gera 1 linha clicável com a árvore de spans.</span>
</div>
""")

vs_client = VectorSearchClient(disable_notice=True)
idx = vs_client.get_index(endpoint_name=VS_ENDPOINT_NAME, index_name=VS_INDEX_NAME)

llm = ChatDatabricks(endpoint=LLM_MODEL, temperature=0.0, max_tokens=500)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Retrieval tool

# COMMAND ----------

@mlflow.trace(span_type="RETRIEVER")
def retrieve(query: str, k: int = 5, query_type: str = "ANN") -> list[dict]:
    """Busca os top-k chunks mais relevantes no Vector Search Index."""
    results = idx.similarity_search(
        query_text=query,
        columns=["id", "article_id", "category", "title", "chunk"],
        num_results=k,
        query_type=query_type,
    )
    cols = [c["name"] for c in results["manifest"]["columns"]]
    return [
        {**dict(zip(cols, row[:-1])), "score": row[-1]}
        for row in results["result"]["data_array"]
    ]

# Teste rápido
docs = retrieve("Qual o prazo de devolução?")
for d in docs:
    print(f"[{d['score']:.3f}] {d['article_id']} — {d['title']}")
    print(f"   {d['chunk'][:100]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Prompt template + generation

# COMMAND ----------

SYSTEM_PROMPT = """Você é o assistente da Aurorinha, e-commerce brasileiro. Responda **apenas** com base no CONTEXTO abaixo. Se a informação não estiver no contexto, diga claramente "Não tenho essa informação". Seja objetivo, em português do Brasil.

CONTEXTO:
{context}
"""

@mlflow.trace(span_type="LLM")
def generate(query: str, docs: list[dict]) -> str:
    context = "\n\n".join(
        f"[{d['article_id']} — {d['title']}]\n{d['chunk']}"
        for d in docs
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": query},
    ]
    response = llm.invoke(messages)
    return response.content

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Agent baseline (retrieve → generate)

# COMMAND ----------

@mlflow.trace(span_type="CHAIN", name="rag_baseline")
def rag_baseline(query: str, k: int = 5) -> dict:
    docs = retrieve(query, k=k)
    answer = generate(query, docs)
    return {"response": answer, "retrieved": docs}

# COMMAND ----------

# Teste
out = rag_baseline("Como rastreio meu pedido pelo WhatsApp?")
print("📨 Resposta:")
print(out["response"])
print("\n📚 Documentos usados:")
for d in out["retrieved"]:
    print(f"  - {d['article_id']} (score {d['score']:.3f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Variante com **reranking** (LLM-judge)
# MAGIC
# MAGIC Buscamos top-K maior (k=10), pedimos pro LLM pontuar cada doc por relevância à query, e ficamos com os top-N (n=3).
# MAGIC
# MAGIC > Em produção, prefira um cross-encoder dedicado (ex: `databricks-bge-reranker-large`). LLM-judge funciona aqui pra evitar custo de outro endpoint.

# COMMAND ----------

RERANK_PROMPT = """Você é um juiz que pontua relevância de documentos a uma pergunta.

Pergunta: {query}

Documento:
{doc}

Pontue de 0 a 10 quão relevante o documento é pra responder a pergunta. Responda APENAS um JSON: {{"score": <numero>}}.
"""

@mlflow.trace(span_type="RERANKER")
def rerank(query: str, docs: list[dict], n: int = 3) -> list[dict]:
    rated = []
    for d in docs:
        prompt = RERANK_PROMPT.format(query=query, doc=d["chunk"])
        resp = llm.invoke([{"role": "user", "content": prompt}]).content
        try:
            score = float(json.loads(resp.strip().strip("`").replace("json\n", ""))["score"])
        except Exception:
            score = 0.0
        rated.append({**d, "rerank_score": score})
    return sorted(rated, key=lambda x: x["rerank_score"], reverse=True)[:n]

@mlflow.trace(span_type="CHAIN", name="rag_reranked")
def rag_reranked(query: str, k_retrieve: int = 10, k_final: int = 3) -> dict:
    docs = retrieve(query, k=k_retrieve)
    top = rerank(query, docs, n=k_final)
    answer = generate(query, top)
    return {"response": answer, "retrieved": top}

# COMMAND ----------

# Teste lado a lado
query = "Posso devolver um produto personalizado?"

print("🔵 BASELINE:")
b = rag_baseline(query)
print(b["response"])
print("Docs:", [d["article_id"] for d in b["retrieved"]])

print("\n🟢 RERANKED:")
r = rag_reranked(query)
print(r["response"])
print("Docs:", [(d["article_id"], round(d["rerank_score"], 1)) for d in r["retrieved"]])

# COMMAND ----------

# MAGIC %md
# MAGIC **O que observar:**
# MAGIC - Baseline pode trazer chunks de `faq-002` (devolução) + ruído de outras políticas
# MAGIC - Reranked tipicamente isola o chunk *específico* sobre produtos personalizados (que está dentro de `faq-002`) e ignora ruído
# MAGIC - A resposta com rerank fica mais precisa, mas custou +K chamadas extras de LLM (latência sobe)
# MAGIC
# MAGIC ## 6. Veja os traces na UI

# COMMAND ----------

displayHTML(f"""
<div style='font-family: sans-serif; padding: 16px; background: #f5f5f5; border-radius: 8px;'>
  <h3 style='margin: 0 0 8px 0;'>🧵 Suas traces estão prontas</h3>
  <a href="{EXP_URL}/traces" target="_blank" style='font-size: 16px;'>👉 Abrir aba Traces</a>
  <p style='margin: 8px 0 0 0; color: #555;'>
    Cada chamada acima de <code>rag_baseline</code> e <code>rag_reranked</code> apareceu lá.
    Clica numa trace pra ver:
    <ul style='margin: 4px 0;'>
      <li>Árvore de spans com latência por etapa (<code>retrieve</code>, <code>rerank</code>, <code>generate</code>)</li>
      <li>Inputs/outputs preservados de cada span</li>
      <li>Tempo total e onde o gargalo tá</li>
    </ul>
  </p>
</div>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Concluído
# MAGIC
# MAGIC Temos duas variantes do agent. No próximo lab vamos comparar **objetivamente** qual é melhor.
# MAGIC
# MAGIC ➡️ Próximo: [`04_evaluation`](./04_evaluation)
