# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop: RAG em Produção com Databricks
# MAGIC
# MAGIC > Hands-on de 2h para construir, avaliar e operar um pipeline RAG end-to-end usando **Mosaic AI Vector Search**, **Agent Framework** e **MLflow Evaluation**.
# MAGIC
# MAGIC ## Objetivos
# MAGIC
# MAGIC Ao final você vai saber:
# MAGIC - Comparar estratégias de **chunking** e medir impacto no retrieval
# MAGIC - Criar e consultar um **Vector Search Index** com embeddings gerenciados
# MAGIC - Montar um **Agent** com retrieval tool + reranking
# MAGIC - Avaliar pipelines RAG com **`mlflow.evaluate`** e LLM-as-judge
# MAGIC - Instrumentar **MLflow Tracing** e ler trade-offs de produção
# MAGIC
# MAGIC ## Pré-requisitos
# MAGIC
# MAGIC | Item | O que checar |
# MAGIC |---|---|
# MAGIC | **Workspace** | Acesso ao Foundation Model API (`databricks-gte-large-en`, `databricks-meta-llama-3-3-70b-instruct`) |
# MAGIC | **Compute** | Serverless OU cluster DBR 15.4 LTS ML (ou superior) |
# MAGIC | **Unity Catalog** | Permissão `CREATE SCHEMA` em algum catálogo (default: `users`) |
# MAGIC | **Vector Search** | Endpoint compartilhado ou permissão pra criar um |
# MAGIC
# MAGIC ## Roteiro (2h)
# MAGIC
# MAGIC | # | Notebook | Tempo | O que faz |
# MAGIC |---|---|---|---|
# MAGIC | 0 | [`notebooks/00_setup`](./notebooks/00_setup) | 10 min | Schema + dataset sintético "Aurorinha" (e-commerce BR) |
# MAGIC | — | [`notebooks/01_extra`](./notebooks/01_extra) | _opcional_ | Demo: tokenização + chunking + embedding com 3 textos |
# MAGIC | 1 | [`notebooks/02_chunking_embeddings`](./notebooks/02_chunking_embeddings) | 25 min | 3 estratégias de chunking + embeddings via FM API |
# MAGIC | 2 | [`notebooks/03_vector_search`](./notebooks/03_vector_search) | 20 min | Delta + Vector Search Index + retrieval semântico |
# MAGIC | 3 | [`notebooks/04_rag_agent`](./notebooks/04_rag_agent) | 25 min | Agent com retrieval tool + variante com reranking |
# MAGIC | 4 | [`notebooks/05_evaluation`](./notebooks/05_evaluation) | 25 min | `mlflow.evaluate` + LLM judges + comparação |
# MAGIC | 5 | [`notebooks/06_observability_tradeoffs`](./notebooks/06_observability_tradeoffs) | 15 min | Tracing, inference tables, trade-offs de prod |
# MAGIC
# MAGIC ## Antes de começar
# MAGIC
# MAGIC 1. **Edite `config/00_config`** colocando seu nome no `SCHEMA_NAME`
# MAGIC 2. Execute os notebooks **na ordem** — eles compartilham tabelas via UC
# MAGIC 3. Leia os docs (`docs/`) entre os labs pra ancorar a teoria
# MAGIC
# MAGIC ## Documentação extra
# MAGIC
# MAGIC - [`docs/00_GUIA_WORKSHOP`](./docs/00_GUIA_WORKSHOP) — Roteiro detalhado pra facilitador
# MAGIC - [`docs/01_ARQUITETURA`](./docs/01_ARQUITETURA) — Diagrama + decisões de design
# MAGIC - [`docs/02_APRENDIZADOS_PRODUCAO`](./docs/02_APRENDIZADOS_PRODUCAO) — Lições de RAG em escala
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Dataset:** "Aurorinha" — rede de e-commerce fictícia (FAQ de cliente + políticas + KB de produtos).
# MAGIC **Stack:** Unity Catalog · Mosaic AI Vector Search · Foundation Model API · MLflow 2.x · Agent Framework
# MAGIC **Idioma:** PT-BR (código + docs + dataset)
