# Databricks notebook source
# MAGIC %md
# MAGIC # Arquitetura do Pipeline
# MAGIC
# MAGIC ## Visão geral
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────────┐    ┌────────────┐    ┌──────────────────┐    ┌────────────────┐
# MAGIC │  KB Articles     │───▶│  Chunking  │───▶│   Delta Table    │───▶│ Vector Search  │
# MAGIC │  (raw text)      │    │  + Splits  │    │  (CDF habilitado)│    │  Index (sync)  │
# MAGIC └──────────────────┘    └────────────┘    └──────────────────┘    └────────┬───────┘
# MAGIC                                                                              │
# MAGIC                                                                              ▼
# MAGIC                              ┌─────────────────────────────────────────────────────────┐
# MAGIC                              │                    RAG Agent                            │
# MAGIC                              │                                                         │
# MAGIC                              │   ┌──────────┐   ┌──────────┐   ┌─────────────────┐   │
# MAGIC                              │   │ Retrieve │──▶│  Rerank  │──▶│  Generate (LLM) │   │
# MAGIC                              │   │  (top-K) │   │ (top-N)  │   │  Foundation API │   │
# MAGIC                              │   └──────────┘   └──────────┘   └─────────────────┘   │
# MAGIC                              └─────────────────────────────────────────────────────────┘
# MAGIC                                                       │
# MAGIC                                                       ▼
# MAGIC                              ┌────────────────┐    ┌─────────────────┐
# MAGIC                              │ MLflow Tracing │    │ Inference Table │
# MAGIC                              │   (spans)      │    │   (Delta)       │
# MAGIC                              └────────────────┘    └─────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ## Componentes
# MAGIC
# MAGIC ### 1. Camada de dados (Unity Catalog)
# MAGIC - **`kb_articles`** — texto bruto (artigo, FAQ, política). Tabela source.
# MAGIC - **`kb_chunks`** — pedaços + metadata após chunking. CDF (`delta.enableChangeDataFeed=true`) pra alimentar Vector Search Sync.
# MAGIC - **`eval_dataset`** — pergunta + ground-truth (resposta + context_ids). Curado manualmente; pequeno (20-50 linhas).
# MAGIC
# MAGIC ### 2. Camada de retrieval (Vector Search)
# MAGIC - **Endpoint**: **1 endpoint serverless por participante** — nome derivado do usuário (ex: `vinicius_fialho_vs`). Criado no Lab 0 em background pra absorver os 5-10min de cold start enquanto o Lab 1 roda. Limpeza no final do evento via `vs_client.delete_endpoint()` (custo $).
# MAGIC - **Index**: `Delta Sync` com **managed embeddings** — Databricks chama `databricks-gte-large-en` no ingest e no query.
# MAGIC - **Modo**: semântico puro pra simplicidade; hybrid (semantic+keyword) é 1-flag de mudança.
# MAGIC
# MAGIC ### 3. Camada de geração (Agent)
# MAGIC - **Função Python** com `@mlflow.trace`. Não usamos `ChatAgent` aqui pra deixar transparente — em produção, embrulhar em `ChatAgent` ou `LangGraph` é o passo natural.
# MAGIC - **Retrieval tool**: chama o VS Index e devolve top-K chunks.
# MAGIC - **Rerank**: LLM judge (zero-cost adicional além das chamadas de LLM) reordena top-K → top-N.
# MAGIC - **Generation**: prompt template + chamada ao `databricks-meta-llama-3-3-70b-instruct`.
# MAGIC
# MAGIC ### 4. Camada de avaliação (MLflow)
# MAGIC - `mlflow.evaluate(model_type="databricks-agent")` → judges: `relevance_to_query`, `faithfulness`, `context_sufficiency`, `correctness`.
# MAGIC - Cada execução vira uma run; runs são comparáveis lado a lado na UI.
# MAGIC
# MAGIC ### 5. Camada de observabilidade
# MAGIC - **MLflow Tracing** automático nas funções decoradas com `@mlflow.trace`.
# MAGIC - **Inference Tables** quando deployar como endpoint (não cobrimos deploy no workshop, mas o config tá pronto).
# MAGIC
# MAGIC ## Decisões de design
# MAGIC
# MAGIC | Decisão | Por que |
# MAGIC |---|---|
# MAGIC | **Managed embeddings (não self-managed)** | Evita drift entre embedder de ingest e de query. Menos código pro participante. |
# MAGIC | **Endpoint compartilhado de VS** | Cold start de endpoint dedicado leva 5-10min. Workshop de 2h não tem esse luxo. |
# MAGIC | **Recursive chunking como default** | Melhor relação qualidade/custo. Semantic chunking é demo do Lab 1, não default. |
# MAGIC | **LLM-judge rerank (não cross-encoder)** | Cross-encoder exige endpoint dedicado. LLM-judge reusa o LLM principal — viável no workshop. Em prod, recomenda-se [Databricks rerank](https://docs.databricks.com/en/generative-ai/vector-search.html#rerank). |
# MAGIC | **Função `@mlflow.trace` em vez de chain** | Transparência > açúcar sintático em workshop. Em prod, LangGraph/LangChain são padrão. |
# MAGIC | **Eval set pequeno (20 linhas)** | Cada linha = 4 chamadas de LLM (1 per judge). 20 linhas × 3 variantes = 240 calls. Cabe em <5min. |
# MAGIC
# MAGIC ## O que está fora do escopo (a propósito)
# MAGIC
# MAGIC - **Deploy do agent** como Model Serving endpoint — código existe, mas não rodamos pra economizar tempo.
# MAGIC - **Inference table review** — depende de deploy. Discutimos conceitualmente no Lab 5.
# MAGIC - **Streaming responses** — UX melhor mas não muda fundamentos.
# MAGIC - **Multi-tenant / row-level security** no retrieval — tópico avançado.
# MAGIC - **Fine-tuning do embedder** — raramente vale o esforço; melhor investir em chunking + reranking.
