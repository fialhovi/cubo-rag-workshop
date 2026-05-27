# Workshop Cubo — RAG em Produção com Databricks

Hands-on de 2h para construir, avaliar e operar um pipeline RAG end-to-end usando **Mosaic AI Vector Search**, **Agent Framework** e **MLflow Evaluation**.

## Objetivos

Ao final você vai saber:

- Comparar estratégias de **chunking** e medir impacto no retrieval
- Criar e consultar um **Vector Search Index** com embeddings gerenciados
- Montar um **Agent** com retrieval tool + reranking
- Avaliar pipelines RAG com `mlflow.evaluate` e LLM-as-judge
- Instrumentar **MLflow Tracing** e ler trade-offs de produção

## Pré-requisitos

| Item | O que checar |
|---|---|
| **Workspace Databricks** | Acesso ao Foundation Model API (`databricks-gte-large-en`, `databricks-meta-llama-3-3-70b-instruct`) |
| **Compute** | Serverless OU cluster DBR 15.4 LTS ML (ou superior) |
| **Unity Catalog** | Catálogo `workshop_databricks` com permissão `CREATE SCHEMA` |
| **Vector Search** | Permissão pra criar endpoint serverless |

## Roteiro (2h)

| # | Notebook | Tempo | O que faz |
|---|---|---|---|
| 0 | [`notebooks/00_setup`](notebooks/00_setup.py) | 10 min | Schema + dataset "Aurorinha" + dispara criação do VS endpoint |
| 1 | [`notebooks/01_chunking_embeddings`](notebooks/01_chunking_embeddings.py) | 25 min | 3 estratégias de chunking (PySpark) + embeddings |
| 2 | [`notebooks/02_vector_search`](notebooks/02_vector_search.py) | 20 min | Aguarda endpoint + cria Delta Sync Index + retrieval |
| 3 | [`notebooks/03_rag_agent`](notebooks/03_rag_agent.py) | 25 min | Agent com retrieval tool + variante com LLM-judge rerank |
| 4 | [`notebooks/04_evaluation`](notebooks/04_evaluation.py) | 25 min | `mlflow.evaluate(model_type="databricks-agent")` A/B |
| 5 | [`notebooks/05_observability_tradeoffs`](notebooks/05_observability_tradeoffs.py) | 15 min | MLflow Tracing + checklist de prod + cleanup |

## Como usar

### Opção A — Import no seu workspace Databricks (recomendado)

1. No seu workspace, clique com botão direito numa pasta → **Import** → **URL**
2. Cole: `https://github.com/<user>/cubo-rag-workshop/archive/refs/heads/main.zip`
3. Execute os notebooks na ordem (`_LEIA_PRIMEIRO` → `notebooks/00` → ...)

### Opção B — Databricks CLI

```bash
git clone https://github.com/<user>/cubo-rag-workshop.git
cd cubo-rag-workshop
databricks workspace import-dir . /Users/seu.email@empresa.com/cubo_rag --overwrite
```

### Configuração

Não precisa editar nada — o `SCHEMA_NAME` é derivado automaticamente do seu usuário (`fulano.silva@empresa.com` → `fulano_silva_rag`) e o catálogo `workshop_databricks` precisa existir no workspace (o facilitador cria antes do evento).

## Estrutura

```
.
├── _LEIA_PRIMEIRO.py           # Entrypoint com visão geral
├── config/
│   └── 00_config.py            # Schema/catalog/endpoint derivados do usuário
├── docs/
│   ├── 00_GUIA_WORKSHOP.py     # Roteiro do facilitador
│   ├── 01_ARQUITETURA.py       # Diagrama + decisões de design
│   └── 02_APRENDIZADOS_PRODUCAO.py  # 10 lições de RAG em escala
└── notebooks/
    ├── 00_setup.py
    ├── 01_chunking_embeddings.py
    ├── 02_vector_search.py
    ├── 03_rag_agent.py
    ├── 04_evaluation.py
    └── 05_observability_tradeoffs.py
```

## Stack

- **Unity Catalog** — `workshop_databricks.<user>_rag`
- **Mosaic AI Vector Search** — Delta Sync Index com managed embeddings
- **Foundation Model API** — `databricks-gte-large-en` (embeddings) + `databricks-meta-llama-3-3-70b-instruct` (LLM)
- **MLflow 2.18+** — Tracing + `mlflow.evaluate(model_type="databricks-agent")`
- **PySpark** — UDFs + `posexplode` pra chunking

## Dataset

**Aurorinha** — rede de e-commerce fictícia brasileira. 14 artigos (FAQ, produtos, políticas) + 15 perguntas com ground truth pra avaliação. Tudo em PT-BR.

## Cleanup

No final do workshop, **delete o seu Vector Search Endpoint** pra não gerar custo:

```python
from databricks.vector_search.client import VectorSearchClient
VectorSearchClient(disable_notice=True).delete_endpoint(VS_ENDPOINT_NAME)
spark.sql(f"DROP SCHEMA IF EXISTS {CATALOG_NAME}.{SCHEMA_NAME} CASCADE")
```

## Inspirado em

[CaduBettanim/AgentBricks-Lab](https://github.com/CaduBettanim/AgentBricks-Lab) — padrão de workshop hands-on da Databricks BR.

## Autor

Vinicius Fialho · Field Engineering Brasil · Databricks
