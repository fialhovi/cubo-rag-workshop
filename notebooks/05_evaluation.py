# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 4 — Avaliação
# MAGIC
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | 🎯 **Objetivo** | Avaliar duas variantes do agent com `mlflow.evaluate` e LLM-as-judge |
# MAGIC | ⏱ **Tempo** | 25 min |
# MAGIC
# MAGIC **Conceitos abordados:**
# MAGIC - `mlflow.evaluate(model_type="databricks-agent")` — dispara judges built-in (relevance, faithfulness, correctness, context sufficiency)
# MAGIC - Comparação A/B na UI de Experiments
# MAGIC - Trade-off custo × qualidade

# COMMAND ----------

# MAGIC %pip install -q databricks-agents>=0.16 mlflow>=2.18 databricks-vectorsearch databricks-langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../config/00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Re-define as funções RAG
# MAGIC
# MAGIC Em um projeto real, isso viraria um módulo Python. Pro workshop, redefinimos pra notebook ser auto-contido.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from databricks_langchain import ChatDatabricks
import mlflow, json

mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)
EXPERIMENT_ID = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_PATH).experiment_id
WORKSPACE_URL = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
EXP_URL = f"{WORKSPACE_URL}/ml/experiments/{EXPERIMENT_ID}"

displayHTML(f"""
<div style='font-family: sans-serif; padding: 12px; background: #eef6ff; border-left: 4px solid #1976d2; border-radius: 4px;'>
  <b>📊 Experimento MLflow:</b> <a href="{EXP_URL}" target="_blank">{MLFLOW_EXPERIMENT_PATH}</a><br>
  <span style='color: #555;'>Deixa essa aba aberta — vamos voltar várias vezes pra ver runs, traces e a aba <i>Evaluation</i>.</span>
</div>
""")

vs_client = VectorSearchClient(disable_notice=True)
idx = vs_client.get_index(endpoint_name=VS_ENDPOINT_NAME, index_name=VS_INDEX_NAME)
llm = ChatDatabricks(endpoint=LLM_MODEL, temperature=0.0, max_tokens=500)

SYSTEM_PROMPT = """Você é o assistente da Aurorinha. Responda **apenas** com base no CONTEXTO. Se não souber, diga "Não tenho essa informação". PT-BR.

CONTEXTO:
{context}
"""

def retrieve(query, k=5):
    r = idx.similarity_search(
        query_text=query,
        columns=["id", "article_id", "category", "title", "chunk"],
        num_results=k,
    )
    cols = [c["name"] for c in r["manifest"]["columns"]]
    return [{**dict(zip(cols, row[:-1])), "score": row[-1]} for row in r["result"]["data_array"]]

def generate(query, docs):
    ctx = "\n\n".join(f"[{d['article_id']}] {d['chunk']}" for d in docs)
    return llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT.format(context=ctx)},
        {"role": "user", "content": query},
    ]).content

def rerank(query, docs, n=3):
    rated = []
    for d in docs:
        p = f"Pergunta: {query}\nDoc: {d['chunk']}\nResponda apenas JSON {{\"score\": <0-10>}}."
        try:
            s = float(json.loads(llm.invoke([{"role": "user", "content": p}]).content.strip().strip("`").replace("json\n", ""))["score"])
        except Exception:
            s = 0.0
        rated.append({**d, "rerank_score": s})
    return sorted(rated, key=lambda x: x["rerank_score"], reverse=True)[:n]

def rag_baseline(query, k=5):
    docs = retrieve(query, k=k)
    return {"response": generate(query, docs), "retrieved_context": [{"doc_uri": d["article_id"], "content": d["chunk"]} for d in docs]}

def rag_reranked(query, k_retrieve=5, k_final=3):
    docs = retrieve(query, k=k_retrieve)
    top = rerank(query, docs, n=k_final)
    return {"response": generate(query, top), "retrieved_context": [{"doc_uri": d["article_id"], "content": d["chunk"]} for d in top]}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Carrega o eval set

# COMMAND ----------

import pandas as pd

# Em produção: roda em todo o eval set. No workshop, limitamos a 6 pra caber
# no QPM (queries-per-minute) do endpoint Foundation Model. Cada linha dispara
# ~1 generate + ~4 judges = 5 chamadas LLM. 6 linhas × 2 variantes = 60 calls + rerank.
eval_pdf_full = spark.table(EVAL_TABLE).toPandas()
eval_pdf = eval_pdf_full.head(6).reset_index(drop=True)
print(f"{len(eval_pdf)} exemplos pro lab (de {len(eval_pdf_full)} disponíveis no dataset)")
display(eval_pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Roda **baseline** + `mlflow.evaluate`
# MAGIC
# MAGIC O `model_type="databricks-agent"` dispara judges como `relevance_to_query`, `groundedness`, `correctness`, `chunk_relevance`, `context_sufficiency`. Cada um faz 1 chamada de LLM por linha do eval — por isso mantivemos o set em 15 exemplos.

# COMMAND ----------

import time

def eval_model(name: str, fn, eval_pdf: pd.DataFrame, params: dict):
    """Roda o agent em todas as perguntas e chama mlflow.evaluate.

    Tags e params aparecem na UI do Experiments como colunas filtráveis.
    """
    t0 = time.time()
    preds = [fn(q) for q in eval_pdf["request"]]
    latency = (time.time() - t0) / len(preds)

    data = eval_pdf.copy()
    data["response"] = [p["response"] for p in preds]
    data["retrieved_context"] = [p["retrieved_context"] for p in preds]

    with mlflow.start_run(run_name=name) as run:
        # Tags → aparecem como facetas filtráveis na lista de runs
        mlflow.set_tags({
            "variant": name,
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": LLM_MODEL,
            "lab": "05_evaluation",
        })
        # Params → coluna na UI; permite ordenar por "k", "rerank", etc.
        mlflow.log_params(params)
        mlflow.log_metric("avg_latency_sec", latency)

        results = mlflow.evaluate(
            data=data,
            model_type="databricks-agent",
        )
        return run.info.run_id, results, data

# COMMAND ----------

print("🔵 Avaliando BASELINE (k=5)...")
baseline_run_id, baseline_results, baseline_data = eval_model(
    "baseline_k5", rag_baseline, eval_pdf,
    params={"k_retrieve": 5, "rerank": False, "k_final": 5},
)
print(f"   Run ID: {baseline_run_id}")

displayHTML(f"""
<div style='font-family: sans-serif; padding: 8px 12px; background: #f9f9f9; border-left: 3px solid #1976d2;'>
  ✅ Baseline avaliado.
  <a href="{EXP_URL}/runs/{baseline_run_id}" target="_blank">Abrir run no UI</a>
  · <a href="{EXP_URL}/runs/{baseline_run_id}/evaluations" target="_blank">Aba Evaluation</a> (resposta + judges por linha)
</div>
""")

# COMMAND ----------

print("🟢 Avaliando RERANKED (retrieve k=5 → top 3)...")
reranked_run_id, reranked_results, reranked_data = eval_model(
    "reranked_k5_n3", rag_reranked, eval_pdf,
    params={"k_retrieve": 5, "rerank": True, "k_final": 3},
)
print(f"   Run ID: {reranked_run_id}")

displayHTML(f"""
<div style='font-family: sans-serif; padding: 8px 12px; background: #f9f9f9; border-left: 3px solid #2e7d32;'>
  ✅ Reranked avaliado.
  <a href="{EXP_URL}/runs/{reranked_run_id}" target="_blank">Abrir run no UI</a>
  · <a href="{EXP_URL}/runs/{reranked_run_id}/evaluations" target="_blank">Aba Evaluation</a>
</div>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Comparação lado a lado

# COMMAND ----------

import pandas as pd

def flatten(m: dict):
    return {k: round(v, 3) if isinstance(v, (int, float)) else v for k, v in m.items()}

comparison = pd.DataFrame([
    {"variant": "baseline_k5",      **flatten(baseline_results.metrics)},
    {"variant": "reranked_k5_n3",   **flatten(reranked_results.metrics)},
])
display(comparison)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Como ler
# MAGIC
# MAGIC Métricas chave que aparecem nos resultados (depende da versão do `databricks-agents`):
# MAGIC
# MAGIC | Métrica | O que mede | Bom > |
# MAGIC |---|---|---|
# MAGIC | `response/llm_judged/correctness/rating` | Resposta bate com `expected_response`? | 0.8 |
# MAGIC | `response/llm_judged/relevance_to_query/rating` | Resposta é relevante à pergunta? | 0.9 |
# MAGIC | `response/llm_judged/groundedness/rating` | Resposta tem base no contexto (anti-alucinação)? | 0.9 |
# MAGIC | `retrieval/llm_judged/chunk_relevance/precision` | % dos chunks recuperados que são relevantes | 0.7 |
# MAGIC | `retrieval/ground_truth/document_recall` | % dos docs esperados que apareceram no retrieval | 0.8 |
# MAGIC | `agent/total_input_token_count` | Tokens consumidos | (menor melhor) |
# MAGIC | `avg_latency_sec` | Latência média por query | (menor melhor) |
# MAGIC
# MAGIC **Trade-off típico:** reranked ganha em `correctness` e `chunk_relevance/precision`, mas perde em `latency` (e custo de tokens). Decisão depende do SLA do produto.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. 📊 Comparação visual na UI do Experiments
# MAGIC
# MAGIC O `model_type="databricks-agent"` desbloqueia uma aba **Evaluation** dedicada que mostra:
# MAGIC - Resposta de cada variante lado a lado por linha do eval set
# MAGIC - Nota de cada judge com **rationale** (LLM explica por que deu 0/1)
# MAGIC - Contextos recuperados com highlight dos relevantes
# MAGIC
# MAGIC É o caminho mais didático pra ensinar "como saber se mudança é melhoria".

# COMMAND ----------

import urllib.parse
compare_param = urllib.parse.quote(f'["{baseline_run_id}","{reranked_run_id}"]')
compare_url = f"{EXP_URL}/compare-runs?runs={compare_param}"

displayHTML(f"""
<div style='font-family: sans-serif; padding: 16px; background: linear-gradient(90deg, #eef6ff, #f0fdf4); border-radius: 8px; border: 1px solid #ddd;'>
  <h3 style='margin: 0 0 12px 0;'>🔗 Atalhos pra UI do Experiments</h3>
  <table style='border-collapse: collapse; width: 100%;'>
    <tr><td style='padding: 6px;'>📂 Experimento</td>
        <td><a href="{EXP_URL}" target="_blank">{MLFLOW_EXPERIMENT_PATH}</a></td></tr>
    <tr><td style='padding: 6px;'>🔵 Run baseline</td>
        <td><a href="{EXP_URL}/runs/{baseline_run_id}" target="_blank">{baseline_run_id}</a>
            · <a href="{EXP_URL}/runs/{baseline_run_id}/evaluations" target="_blank">aba Evaluation</a></td></tr>
    <tr><td style='padding: 6px;'>🟢 Run reranked</td>
        <td><a href="{EXP_URL}/runs/{reranked_run_id}" target="_blank">{reranked_run_id}</a>
            · <a href="{EXP_URL}/runs/{reranked_run_id}/evaluations" target="_blank">aba Evaluation</a></td></tr>
    <tr><td style='padding: 6px;'><b>📊 COMPARE</b></td>
        <td><a href="{compare_url}" target="_blank"><b>Side-by-side das duas runs</b></a></td></tr>
  </table>
  <p style='margin: 12px 0 0 0; color: #555; font-size: 13px;'>
    Dica de demo: depois de abrir o Compare, vai em <b>Chart</b> e jogue
    <code>response/llm_judged/correctness/rating</code> no eixo Y pra visualizar a diferença.
  </p>
</div>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Respostas individuais (visualização rápida no notebook)

# COMMAND ----------

# Tabela inline pra inspeção rápida — a UI de Evaluation é mais rica, mas
# essa tabela é útil pra um sanity check sem trocar de aba.
side_by_side = pd.DataFrame({
    "request": baseline_data["request"].values,
    "expected_response": baseline_data["expected_response"].values,
    "baseline_response": baseline_data["response"].values,
    "reranked_response": reranked_data["response"].values,
})
display(side_by_side)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Discussão (5 min)
# MAGIC
# MAGIC Olhando pros números — qual variante vocês escolheriam pra **produção**? Considere:
# MAGIC - **Caso de uso**: chat de suporte (latência importa) ou análise interna (qualidade > tudo)?
# MAGIC - **Custo**: reranked usa 5-10x mais tokens. Quanto é "ok" gastar pra ganhar X% de correctness?
# MAGIC - **Riscos do `Não tenho essa informação`**: o groundedness alto previne alucinação, mas usuário sente "robô burro" se vir muito esse fallback
# MAGIC
# MAGIC ➡️ Próximo: **06_observability_tradeoffs** — fechamento + lições de prod.