# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 5 — Observabilidade & Trade-offs de Produção
# MAGIC
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | 🎯 **Objetivo** | Ver traces, entender inference tables, ancorar decisões pra produção |
# MAGIC | ⏱ **Tempo** | 15 min (5 código + 10 discussão) |
# MAGIC
# MAGIC **Conceitos abordados:**
# MAGIC - **MLflow Tracing** — debug visual do pipeline
# MAGIC - **Inference Tables** — capturar request/response em Delta automaticamente (no deploy)
# MAGIC - Checklist de prod

# COMMAND ----------

# MAGIC %pip install -q mlflow>=2.18
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../config/00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Conecta no experimento + links pra UI

# COMMAND ----------

import mlflow

mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)
EXPERIMENT_ID = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_PATH).experiment_id
WORKSPACE_URL = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
EXP_URL = f"{WORKSPACE_URL}/ml/experiments/{EXPERIMENT_ID}"
TRACES_URL = f"{EXP_URL}/traces"

displayHTML(f"""
<div style='font-family: sans-serif; padding: 16px; background: #f8f4ff; border-left: 4px solid #7e57c2; border-radius: 4px;'>
  <h3 style='margin: 0 0 8px 0;'>🔍 Onde os traces vivem</h3>
  Todo notebook que rodou com <code>@mlflow.trace</code> nos labs 03 e 04 deixou rastro aqui:
  <ul style='margin: 8px 0;'>
    <li><a href="{EXP_URL}" target="_blank">📂 Experimento (Runs)</a></li>
    <li><a href="{TRACES_URL}" target="_blank"><b>🧵 Aba Traces</b></a> ← cada chamada = 1 linha clicável</li>
  </ul>
  <p style='margin: 8px 0 0 0; font-size: 13px; color: #555;'>Clica numa trace → árvore de spans (retrieve → rerank → generate) com latência por span e inputs/outputs preservados.</p>
</div>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Busca traces via API
# MAGIC
# MAGIC A UI mostra a árvore; a API permite **filtrar e agregar** — base pra eval contínua e drift detection.

# COMMAND ----------

# Todas as traces das últimas execuções
traces = mlflow.search_traces(
    experiment_ids=[EXPERIMENT_ID],
    max_results=50,
    order_by=["timestamp_ms DESC"],
)

print(f"📊 {len(traces)} traces nesse experimento")
print(f"Colunas disponíveis: {list(traces.columns)}")

# Mostra um subset com as colunas que existirem (schema muda entre versões de mlflow)
preferred = ["trace_id", "request", "response", "execution_time_ms", "status", "timestamp_ms"]
available = [c for c in preferred if c in traces.columns]
display(traces[available].head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Stats agregadas — latência das traces

# COMMAND ----------

import pandas as pd

# Procura a coluna de latência (nome varia entre versões: execution_time_ms ou execution_duration)
lat_col = next((c for c in ["execution_time_ms", "execution_duration", "duration_ms"] if c in traces.columns), None)

if len(traces) > 0 and lat_col:
    lat = pd.to_numeric(traces[lat_col], errors="coerce").dropna()
    summary = pd.DataFrame([{
        "total_traces": len(lat),
        "p50_ms": int(lat.quantile(0.50)),
        "p95_ms": int(lat.quantile(0.95)),
        "max_ms": int(lat.max()),
        "mean_ms": int(lat.mean()),
    }])
    print(f"⏱ Latência das traces (coluna usada: {lat_col}):")
    display(summary)
    print("\n💡 Na UI, use o filtro por nome (ex: 'rag_baseline' vs 'rag_reranked') pra comparar variantes.")
else:
    print(f"ℹ️ Sem coluna de latência detectada em {list(traces.columns)}. Veja na UI Traces.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Inspeciona uma trace específica
# MAGIC
# MAGIC Programaticamente vs UI: a árvore de spans é a mesma — UI ganha em UX, API ganha em scriptabilidade.

# COMMAND ----------

if len(traces) > 0:
    sample = traces.iloc[0]
    trace_id = sample["trace_id"]
    trace_url = f"{EXP_URL}/traces/{trace_id}"
    lat_text = f"{sample[lat_col]} ms" if lat_col and lat_col in sample.index else "n/a"

    displayHTML(f"""
    <div style='font-family: sans-serif; padding: 12px; background: #fff8e1; border-left: 3px solid #f57c00;'>
      <b>📍 Trace amostral:</b> <code>{trace_id}</code><br>
      <b>Latência total:</b> {lat_text}<br>
      <a href="{trace_url}" target="_blank"><b>🔗 Abrir no UI</b></a>
    </div>
    """)

    full_trace = mlflow.get_trace(trace_id)
    print(f"\n🌲 Árvore de spans ({len(full_trace.data.spans)} spans):")
    for span in full_trace.data.spans:
        latency = (span.end_time_ns - span.start_time_ns) / 1e6
        indent = "  " * (1 if span.parent_id else 0)
        print(f"{indent}[{span.span_type}] {span.name}: {latency:.0f} ms")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Inference Tables (conceitual — não vamos deployar)
# MAGIC
# MAGIC Quando você deploya o agent como Model Serving endpoint com `databricks-agents`, ele **automaticamente** loga toda inferência numa Delta table:
# MAGIC
# MAGIC ```python
# MAGIC from databricks import agents
# MAGIC
# MAGIC # Registra no UC
# MAGIC mlflow.register_model(
# MAGIC     model_uri="runs:/<run_id>/model",
# MAGIC     name=f"{CATALOG_NAME}.{SCHEMA_NAME}.aurorinha_agent",
# MAGIC )
# MAGIC
# MAGIC # Deploy — gera endpoint + inference table
# MAGIC agents.deploy(
# MAGIC     model_name=f"{CATALOG_NAME}.{SCHEMA_NAME}.aurorinha_agent",
# MAGIC     model_version=1,
# MAGIC )
# MAGIC
# MAGIC # Em ~5min você tem:
# MAGIC #   - Endpoint REST público (autenticado via OAuth)
# MAGIC #   - Review App pra coletar feedback humano
# MAGIC #   - Inference table em <catalog>.<schema>.<model_name>_payload
# MAGIC ```
# MAGIC
# MAGIC A inference table tem schema:
# MAGIC
# MAGIC | Coluna | Conteúdo |
# MAGIC |---|---|
# MAGIC | `request_id` | UUID da chamada |
# MAGIC | `request` | JSON da entrada (incluindo `messages`) |
# MAGIC | `response` | JSON da saída (incluindo `retrieved_context` se você setou) |
# MAGIC | `trace` | MLflow trace serializado |
# MAGIC | `databricks_request_id` | Pra cruzar com logs de billing |
# MAGIC | `client_request_id` | UUID que o cliente pode mandar pra correlação |
# MAGIC | `execution_time_ms` | Latência total |
# MAGIC | `status_code` | Sucesso/erro |
# MAGIC | `timestamp` | Quando |
# MAGIC
# MAGIC ### O que dá pra fazer com isso
# MAGIC
# MAGIC - **Eval contínua**: `mlflow.evaluate` direto na inference table com judges → dashboard de qualidade ao longo do tempo
# MAGIC - **Drift detection**: tópico de perguntas mudou? Lakehouse Monitoring sobre o text drift do `request`
# MAGIC - **Replay**: pegou bug em prod? Re-roda o request offline com a nova versão do agent

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Checklist pra levar pro mundo real
# MAGIC
# MAGIC O que fizemos aqui é um **POC funcional**. Pra ir pra produção falta:
# MAGIC
# MAGIC ### Dados
# MAGIC - [ ] Ingestão automática (Auto Loader / DLT / Lakeflow Connect) → source table
# MAGIC - [ ] CDF habilitado na source table (`delta.enableChangeDataFeed=true`)
# MAGIC - [ ] Pipeline TRIGGERED do VS Index agendado (Workflow / Lakeflow) — ou CONTINUOUS se streaming
# MAGIC - [ ] Alerta no `last_sync_time` do índice (Databricks SQL alert)
# MAGIC
# MAGIC ### Qualidade
# MAGIC - [ ] Eval set vivo com pipeline de promoção (tickets/Slack → curadoria → eval set)
# MAGIC - [ ] CI: `mlflow.evaluate` em PR que mexe no agent — bloqueia merge se métrica cair >X%
# MAGIC - [ ] LLM-as-judge calibrado contra anotação humana (Spearman > 0.6)
# MAGIC
# MAGIC ### Retrieval
# MAGIC - [ ] Hybrid search habilitado se há jargão/códigos
# MAGIC - [ ] Reranker (cross-encoder, não LLM judge) em endpoint dedicado pra latência
# MAGIC - [ ] Query rewriting (HyDE / decomposition) antes do retrieval
# MAGIC - [ ] Filtros de metadata pra row-level access control (`category`, `tenant_id`)
# MAGIC
# MAGIC ### Operação
# MAGIC - [ ] Deploy via `agents.deploy()` → endpoint + inference table
# MAGIC - [ ] Review App pra coletar 👍/👎 dos usuários
# MAGIC - [ ] Dashboard Lakeview com: latência p50/p95/p99, error rate, judge scores ao longo do tempo, custo (tokens/dia)
# MAGIC - [ ] Alerta no p95 latency > SLA e no judge score < threshold
# MAGIC
# MAGIC ### Custos
# MAGIC - [ ] Budget cap no endpoint (rate limiting por consumer)
# MAGIC - [ ] Prompt caching habilitado (system prompt fixo → reuso)
# MAGIC - [ ] Logging do `total_input_token_count` por chamada — base pra cost attribution
# MAGIC
# MAGIC ### Compliance (cliente BR)
# MAGIC - [ ] LGPD: PII redaction no log do request (ex. CPF, e-mail)
# MAGIC - [ ] Retention policy na inference table (TTL na Delta)
# MAGIC - [ ] DPA assinado se usar dados de cliente
# MAGIC
# MAGIC ## 5. Discussão (10 min)
# MAGIC
# MAGIC Sugestões de pergunta pra abrir:
# MAGIC
# MAGIC 1. **Onde RAG não basta?** — perguntas agregadas ("quantos pedidos…?") precisam Genie/text-to-SQL. Multi-hop precisa de planner. Veja [`docs/02_APRENDIZADOS_PRODUCAO`](../docs/02_APRENDIZADOS_PRODUCAO).
# MAGIC 2. **Como vocês versionariam o agent?** — `mlflow.register_model` no UC com aliases (`@champion`, `@candidate`). Promotion via API.
# MAGIC 3. **Quem dá ground truth na sua empresa?** — sem isso, qualquer eval é teatro. Tipicamente: SME revisa amostra semanal.
# MAGIC 4. **Custo vs qualidade no seu caso?** — chatbot mass-market = otimizar custo agressivo. Analista jurídico = paga 10x mais por correctness.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🧹 Limpeza (importante — custo de endpoint!)
# MAGIC
# MAGIC O Vector Search endpoint que você criou no Lab 0 fica cobrando enquanto ativo. **Delete quando terminar**:
# MAGIC
# MAGIC ```python
# MAGIC from databricks.vector_search.client import VectorSearchClient
# MAGIC vs_client = VectorSearchClient(disable_notice=True)
# MAGIC vs_client.delete_endpoint(VS_ENDPOINT_NAME)  # ~10s, deleta endpoint e todos os indexes dentro
# MAGIC ```
# MAGIC
# MAGIC Você pode também dropar o schema pra limpar tabelas:
# MAGIC ```python
# MAGIC spark.sql(f"DROP SCHEMA IF EXISTS {CATALOG_NAME}.{SCHEMA_NAME} CASCADE")
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Fim do workshop
# MAGIC
# MAGIC ### O que você construiu
# MAGIC - Pipeline de ingest com chunking comparado
# MAGIC - Vector Search Index com managed embeddings (Delta Sync)
# MAGIC - Agent RAG com retrieval + reranking variável
# MAGIC - Avaliação reproduzível com LLM judges
# MAGIC - Pronto pra adicionar deploy + monitoring
# MAGIC
# MAGIC ### Próximos passos sugeridos
# MAGIC 1. Trocar o dataset pelo seu (qualquer corpus em UC funciona)
# MAGIC 2. Plug-in query rewriting (LLM antes do retrieve)
# MAGIC 3. `agents.deploy()` → endpoint + Review App
# MAGIC 4. Lakeview dashboard sobre a inference table
# MAGIC
# MAGIC **Dúvidas?** vinicius.fialho@databricks.com
