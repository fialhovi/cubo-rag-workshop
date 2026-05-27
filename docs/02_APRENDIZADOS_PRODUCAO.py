# Databricks notebook source
# MAGIC %md
# MAGIC # Aprendizados de RAG em produção
# MAGIC
# MAGIC Compilação prática de coisas que aparecem quando o sistema passa do POC pra escala. Use como checklist quando alguém te perguntar "RAG funciona em prod?".
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 1. Qualidade de retrieval domina qualidade da resposta
# MAGIC
# MAGIC **Regra de bolso:** se o LLM não vê o chunk certo, ele não vai responder certo — não importa quão bom seja o prompt.
# MAGIC
# MAGIC - Antes de tunar prompt, **meça `context_recall`** (o contexto certo está nos top-K?). Se < 70%, qualquer tuning de prompt é otimização local.
# MAGIC - Em sistemas reais, **chunking errado é a fonte #1 de regressão**. Re-chunkar é caro (precisa reindexar) — vale gastar tempo testando estratégias antes de produção.
# MAGIC
# MAGIC ## 2. Hybrid > pure semantic em domínios com jargão
# MAGIC
# MAGIC Vector search puro falha quando a query contém:
# MAGIC - **Códigos / SKUs / IDs**: "ERR-4429", "X9-15B"
# MAGIC - **Nomes próprios raros**: "Aurorinha Black Edition"
# MAGIC - **Acrônimos novos**: produto lançado depois do treinamento do embedder
# MAGIC
# MAGIC Em todos esses casos, BM25 / keyword pega o que o embedding "borra". Mosaic Vector Search expõe `query_type="HYBRID"` — custo: zero código extra, latência: ~10-20ms a mais.
# MAGIC
# MAGIC ## 3. Reranking quase sempre vale
# MAGIC
# MAGIC - Retrieval inicial otimiza recall (k=20-50)
# MAGIC - Reranker re-ordena pra precisão (top-N=3-5)
# MAGIC - Ganho típico: +10-25% em context precision, custo: 50-300ms
# MAGIC - **Quando NÃO usar:** chatbot real-time com <500ms SLA, ou quando top-3 já é "óbvio"
# MAGIC
# MAGIC ## 4. Query rewriting > retrieval mais sofisticado
# MAGIC
# MAGIC Usuários escrevem mal. Pra cada dólar gasto em embedder melhor, 5 dólares em query rewriting rendem mais:
# MAGIC
# MAGIC - **Decomposition** — "Quanto custa o plano X e tem desconto pra estudante?" → 2 queries
# MAGIC - **HyDE** — gerar resposta hipotética e usar como query (paradoxalmente funciona)
# MAGIC - **Conversational rewriting** — em chat, resolver pronomes ("e o anterior?") antes de buscar
# MAGIC
# MAGIC LLM-as-rewriter custa ~50-150ms e melhora retrieval em 15-30% em queries reais.
# MAGIC
# MAGIC ## 5. Eval set vivo, não estático
# MAGIC
# MAGIC O eval set inicial sempre vira shelf-ware em 3 meses. Coisas que dão certo:
# MAGIC
# MAGIC - **Pipeline de promoção**: tickets de suporte / Slack / feedback negativo → eval set candidato → revisão humana → eval set oficial
# MAGIC - **Sampling de produção**: 1% de requests com feedback ruim → eval automático com judges → revisão semanal
# MAGIC - **Eval por persona**: separe eval set por tipo de pergunta (factual / procedural / opinião). Cada um tem judge diferente.
# MAGIC
# MAGIC ## 6. Judges precisam de calibração
# MAGIC
# MAGIC LLM-as-judge é poderoso mas tem viéses:
# MAGIC - Prefere respostas **longas e estruturadas** (mesmo erradas)
# MAGIC - Prefere o **estilo do próprio modelo** (Llama judge gosta de respostas Llama-style)
# MAGIC - **Position bias** quando compara A vs B
# MAGIC
# MAGIC Fixos: amostra 50 pares (judge_score, human_score), calcula correlação. Se Spearman < 0.6, troque o judge ou ajuste o prompt.
# MAGIC
# MAGIC ## 7. Custo: tokenize antes de tudo
# MAGIC
# MAGIC RAG estoura budget quando: contexto cresce sem freio. Patterns:
# MAGIC
# MAGIC | Sintoma | Causa típica | Fix |
# MAGIC |---|---|---|
# MAGIC | $$$ subindo linear no número de usuários | k=20 chunks de 1000 tokens cada → 20k tokens/req | k=5 + reranker; chunks de 400 tok |
# MAGIC | Latência alta variável | Contexto cresce com histórico da conversa | Summarize history a cada N turnos |
# MAGIC | Cache hit ratio baixo | Prompt template muda entre chamadas | Padroniza prompt; usa prompt caching do provider |
# MAGIC
# MAGIC ## 8. Drift de fonte → drift de resposta
# MAGIC
# MAGIC Documentos mudam. Se não tem CDF + reindex automático:
# MAGIC - Política nova fica fora do índice por semanas
# MAGIC - Usuário recebe info **errada**, não desatualizada — é pior porque parece certa
# MAGIC
# MAGIC **Setup mínimo de prod:** Delta CDF + Vector Search Sync + alerta no `last_sync_time` se > X horas.
# MAGIC
# MAGIC ## 9. Observabilidade: traces > logs
# MAGIC
# MAGIC `print()` não funciona quando o agent tem 4 tools e 3 níveis de recursão. MLflow Tracing dá:
# MAGIC - **Span por chamada** (retrieve, rerank, generate)
# MAGIC - **Latência por span** — quem é o gargalo?
# MAGIC - **Inputs/outputs preservados** — debug de produção sem repro local
# MAGIC
# MAGIC Custo: ~1 linha de código (`@mlflow.trace`). Inflar trace storage só vira problema com 1M+ req/dia.
# MAGIC
# MAGIC ## 10. Onde RAG ainda quebra
# MAGIC
# MAGIC Honestidade: cenários onde RAG sozinho não basta:
# MAGIC
# MAGIC - **Agregação**: "Quantos clientes têm plano X?" → precisa SQL, não retrieval. Use Genie/text-to-SQL.
# MAGIC - **Multi-hop**: "Qual cliente do plano X que abriu ticket sobre Y semana passada?" → precisa joins. Idem.
# MAGIC - **Raciocínio temporal**: "O que mudou na política desde abril?" → precisa diff de versões.
# MAGIC - **Confiança calibrada**: o LLM responde com confiança mesmo sem contexto suficiente. Mitigação: `context_sufficiency` judge + "não sei" como output válido.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## TL;DR pra levar pra reunião
# MAGIC
# MAGIC 1. **Meça retrieval antes de mexer em prompt**
# MAGIC 2. **Hybrid + rerank** é o setup padrão em prod
# MAGIC 3. **Query rewriting** rende mais que tunar embedder
# MAGIC 4. **Eval set é processo, não artefato**
# MAGIC 5. **Custo escala com K × chunk_size** — controle isso
# MAGIC 6. **Traces > logs** desde o primeiro dia
