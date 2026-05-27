# Databricks notebook source
# MAGIC %md
# MAGIC # Guia do Facilitador
# MAGIC
# MAGIC Workshop **RAG em Produção com Databricks** · 2h · iniciante→intermediário.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Cronograma sugerido
# MAGIC
# MAGIC | Tempo | Bloco | Atividade |
# MAGIC |---|---|---|
# MAGIC | 0:00 – 0:05 | Abertura | Apresentar objetivos, dataset, stack |
# MAGIC | 0:05 – 0:15 | Lab 0 | Setup + geração de dados **+ dispara criação do VS endpoint (5-10min em background)** |
# MAGIC | _opcional_ | `01_extra` | Demo curta de tokenização + embedding pra audiências iniciantes (pular se a turma já tem repertório) |
# MAGIC | 0:15 – 0:40 | Lab 1 | Chunking + embeddings (endpoint provisiona em paralelo) |
# MAGIC | 0:40 – 1:00 | Lab 2 | Aguarda endpoint pronto + cria índice + retrieval |
# MAGIC | 1:00 – 1:05 | ☕ | Pausa curta |
# MAGIC | 1:05 – 1:30 | Lab 3 | RAG Agent com retrieval tool + reranking |
# MAGIC | 1:30 – 1:55 | Lab 4 | Avaliação com `mlflow.evaluate` + LLM judges |
# MAGIC | 1:55 – 2:10 | Lab 5 | Observabilidade + trade-offs de produção |
# MAGIC | 2:10 – 2:20 | Discussão | Q&A + lições reais (driver) |
# MAGIC
# MAGIC ## Pontos pra enfatizar como facilitador
# MAGIC
# MAGIC ### Lab 1 — Chunking
# MAGIC - **Mostre o trade-off de chunk size**: chunks grandes = mais contexto mas embedding "borra"; chunks pequenos = retrieval preciso mas perde contexto narrativo
# MAGIC - **Recursive splitter** é o default produção: respeita boundaries semânticos (parágrafo → frase → palavra) sem custo extra
# MAGIC - **Semantic chunking** (split por similaridade de embeddings) ganha em texto bem estruturado, mas custa O(N) embeddings só pra splittar
# MAGIC
# MAGIC ### Lab 2 — Vector Search
# MAGIC - **Delta Sync Index** vs **Direct Vector Access**: Sync é o caminho "feliz" — UC table com CDF habilitado → o índice atualiza sozinho. Mostre como funciona via UI.
# MAGIC - **Managed embeddings** (Databricks calcula embeddings) vs **self-managed** (você passa o vetor): para 99% dos casos, managed é a escolha — menos código, menos drift entre embedder de ingest e de query
# MAGIC - **Hybrid search**: vector + keyword (BM25-like) — útil quando há jargão/SKU/código que embedding não captura bem
# MAGIC
# MAGIC ### Lab 3 — Agent
# MAGIC - **Por que Agent e não chain pura?** Tool-calling permite o LLM decidir *quando* buscar (vs sempre buscar). Em produção isso reduz latência e custo em perguntas que não precisam de retrieval
# MAGIC - **Reranking**: model cross-encoder que re-ordena os top-K do vector search. Ganho de quality vs latência ~50-200ms extras
# MAGIC - Em workshop, vamos usar reranker simples (LLM judge sobre cada doc) pra não exigir endpoint dedicado
# MAGIC
# MAGIC ### Lab 4 — Avaliação
# MAGIC - **Sem eval set você não sabe se mudança é melhoria**. Mostre o eval_dataset gerado no Lab 0 — perguntas + ground truth + contextos esperados
# MAGIC - **`mlflow.evaluate(model_type="databricks-agent")`** dispara judges built-in: relevance to query, faithfulness, context sufficiency, correctness
# MAGIC - Compare A/B/C: baseline (k=5), com reranking, com chunk maior — qual vence em qual métrica?
# MAGIC
# MAGIC ### Lab 5 — Produção
# MAGIC - **MLflow Tracing**: cada chamada do agent vira um trace com spans (retrieve, generate, rerank). Custo: ~0, ganho: debugar latência e erros é trivial
# MAGIC - **Inference tables** capturam requests/responses em Delta automaticamente — base pra eval contínua e drift detection
# MAGIC - Discutir trade-offs do doc [`02_APRENDIZADOS_PRODUCAO`](./02_APRENDIZADOS_PRODUCAO)
# MAGIC
# MAGIC ## Bloqueios comuns
# MAGIC
# MAGIC | Sintoma | Causa | Fix |
# MAGIC |---|---|---|
# MAGIC | "Endpoint not found" no FM API | Endpoint não habilitado no workspace | Trocar pra `databricks-mixtral-8x7b-instruct` ou pedir ao admin |
# MAGIC | VS endpoint demora >10min no Lab 2 | Cold start excepcionalmente lento ou workspace lotado | Se passar de 15min, talvez já tenha estourado limite de endpoints — pedir admin pra liberar |
# MAGIC | Participante "esqueceu" o Lab 0 | Endpoint não foi criado | Lab 2 tem fallback que cria, mas o cold start vai bloquear 5-10min. Repassar no início. |
# MAGIC | `mlflow.evaluate` lento | Judges fazem 1 call de LLM por linha | Limitar eval set a 10-20 exemplos no workshop |
# MAGIC | `databricks_langchain` import fail | Lib não instalada | `%pip install -U databricks-langchain mlflow langchain` no topo |
# MAGIC
# MAGIC ## Materiais de apoio
# MAGIC
# MAGIC - [Mosaic AI Vector Search docs](https://docs.databricks.com/en/generative-ai/vector-search.html)
# MAGIC - [Agent Framework](https://docs.databricks.com/en/generative-ai/agent-framework/build-genai-apps.html)
# MAGIC - [MLflow LLM Evaluate](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html)
# MAGIC - [Foundation Model APIs](https://docs.databricks.com/en/machine-learning/foundation-models/index.html)
