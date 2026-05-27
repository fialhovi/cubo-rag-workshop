# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 0 — Setup do ambiente
# MAGIC
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | 🎯 **Objetivo** | Criar schema e gerar o dataset "Aurorinha" (e-commerce BR fictício) |
# MAGIC | ⏱ **Tempo** | 10 min |
# MAGIC | 🧰 **Compute** | Serverless ou DBR 15.4 LTS ML |
# MAGIC
# MAGIC **Antes de rodar:** edite `../config/00_config` colocando seu `SCHEMA_NAME`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Carrega config

# COMMAND ----------

# MAGIC %run ../config/00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Cria schema + volume

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG_NAME}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG_NAME}.{SCHEMA_NAME}")
spark.sql(f"USE CATALOG {CATALOG_NAME}")
spark.sql(f"USE SCHEMA {SCHEMA_NAME}")

print(f"✅ Catalog {CATALOG_NAME} + schema {SCHEMA_NAME} prontos")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Gera o dataset "Aurorinha"
# MAGIC
# MAGIC Rede de e-commerce fictícia. Três tipos de documento:
# MAGIC - **FAQ** — perguntas frequentes (devolução, frete, pagamento)
# MAGIC - **Produtos** — fichas técnicas
# MAGIC - **Políticas** — termos longos (privacidade, troca, garantia)

# COMMAND ----------

from pyspark.sql import Row

kb_articles = [
    # ─── FAQ ───────────────────────────────────────────────────────────────
    Row(article_id="faq-001", category="faq", title="Como rastrear meu pedido",
        content="""Você pode rastrear seu pedido Aurorinha de duas formas. Pelo site: faça login em aurorinha.com.br, vá em "Meus Pedidos" e clique no número do pedido — o status atual e o histórico de movimentações aparecem na hora. Pelo WhatsApp: envie a palavra "rastrear" para o canal oficial Aurorinha (link wa.aurorinha.com.br) e digite o número do pedido quando solicitado. O código de rastreio dos Correios fica disponível assim que o pedido sai do nosso CD em Cajamar (SP), tipicamente em até 24h úteis após a confirmação do pagamento. Caso o status não atualize por mais de 48h após o envio, abra um chamado em "Ajuda > Pedido não atualiza"."""),

    Row(article_id="faq-002", category="faq", title="Política de devolução",
        content="""A Aurorinha aceita devolução de qualquer produto em até 7 dias corridos a partir do recebimento, conforme o Código de Defesa do Consumidor. O produto precisa estar sem uso, com etiqueta original e na embalagem. Para iniciar, acesse "Meus Pedidos > Solicitar devolução" — emitimos um código de postagem gratuito que você usa em qualquer agência dos Correios. O estorno é feito em até 5 dias úteis após o produto chegar no nosso CD: cartão de crédito vira estorno na fatura, Pix volta na conta de origem e boleto vira crédito Aurorinha (válido por 1 ano). Produtos personalizados (gravados a laser) não podem ser devolvidos, exceto em caso de defeito."""),

    Row(article_id="faq-003", category="faq", title="Formas de pagamento aceitas",
        content="""Aceitamos: cartão de crédito (Visa, Master, Elo, Amex, Hipercard) em até 12x sem juros para compras acima de R$ 300; cartão de débito; Pix com 5% de desconto à vista; boleto bancário com 3% de desconto (pago em até 2 dias úteis); e Aurorinha Pay (carteira digital própria, com cashback de 2% em todas as compras). Não aceitamos cheque, transferência TED nem criptomoedas. Compras acima de R$ 5.000 são analisadas pelo time antifraude e podem demorar até 24h pra aprovar."""),

    Row(article_id="faq-004", category="faq", title="Prazo e custo de frete",
        content="""O frete é calculado pelo CEP no carrinho. Capitais Sudeste/Sul: 2-4 dias úteis, R$ 12,90 a R$ 29,90. Capitais NE/CO/N: 4-7 dias úteis, R$ 19,90 a R$ 49,90. Interior: +1 a 3 dias. Frete grátis em compras acima de R$ 199 para Sudeste/Sul e acima de R$ 299 para outras regiões. Frete expresso (Sedex 10) disponível pra capitais Sudeste, +R$ 25 sobre o valor normal. Não entregamos em alguns CEPs de área de risco — o site avisa no checkout se for o seu caso."""),

    Row(article_id="faq-005", category="faq", title="Aurorinha Prime — assinatura",
        content="""Aurorinha Prime é nossa assinatura mensal de R$ 19,90/mês ou R$ 199/ano. Benefícios: frete grátis em qualquer valor pra todo o Brasil, acesso antecipado de 24h em promoções (Black Friday inclusive), 5% de desconto extra em todos os produtos da marca própria Aurora, e atendimento prioritário no chat (fila separada). Cancelamento a qualquer momento sem multa via "Minha Conta > Assinaturas". O primeiro mês é grátis pra contas novas."""),

    # ─── PRODUTOS ──────────────────────────────────────────────────────────
    Row(article_id="prod-001", category="produto", title="Capa de Celular Aurora X — iPhone 15",
        content="""Capa em silicone líquido premium para iPhone 15. Espessura 1.2mm, peso 18g. Bordas elevadas (1.5mm na frente, 1mm na câmera) protegem tela e lentes em quedas de até 1.5m. Compatível com MagSafe — ímãs N52 garantem alinhamento perfeito com carregadores e acessórios MagSafe oficiais. Forro interno em microfibra antiestática evita arranhões. Disponível em 6 cores: preto, azul-meia-noite, rosa, lavanda, verde-musgo e branco. Garantia de 6 meses contra defeito. SKU AUR-CASE-IP15-001. Preço R$ 89,90."""),

    Row(article_id="prod-002", category="produto", title="Fone Aurora Sound Pro — Bluetooth",
        content="""Fone in-ear Bluetooth 5.3 com cancelamento ativo de ruído (ANC) de até -32dB. Driver dinâmico 11mm, resposta 20Hz-20kHz. Bateria: 8h por carga + 28h no estojo (36h total). Carga rápida — 15min de cabo = 3h de uso. Resistência IPX5 (suor e chuva leve, não submergir). Codecs AAC, SBC, LDAC. App Aurora Sound permite EQ customizado e firmware update. Cor: preto fosco ou branco perolado. SKU AUR-EAR-PRO-002. Preço R$ 449,90. Frete grátis em todo Brasil."""),

    Row(article_id="prod-003", category="produto", title="Cuia de Chimarrão Aurora Pampa",
        content="""Cuia tradicional gaúcha em porongo natural curado por 60 dias, revestida em couro bovino tingido a vegetal. Capacidade 350ml. Bocal em alpaca prata 90% com encaixe sob medida para bomba 7mm padrão Aurorinha (incluída). Base em couro pra firmeza. Cada peça é única — pequenas variações de cor e textura são esperadas. Modo de uso: hidratar por 24h com água morna antes do primeiro chimarrão. Não lavar com detergente. SKU AUR-CUIA-PMP-003. Preço R$ 159,90. Produto artesanal de Caçapava do Sul/RS."""),

    Row(article_id="prod-004", category="produto", title="Cafeteira Aurora Brew — italiana 6 xícaras",
        content="""Cafeteira italiana (moka) em alumínio fundido, 6 xícaras (300ml). Indução, gás, elétrico e vitrocerâmico — compatível com qualquer fogão. Cabo em baquelite antitérmica, válvula de segurança em pressão >2 bar. Não vai na máquina de lavar — lavar à mão com água morna sem detergente preserva o "tempero" do alumínio que melhora o sabor com o uso. SKU AUR-CAF-IT6-004. Garantia 1 ano. Preço R$ 129,90."""),

    Row(article_id="prod-005", category="produto", title="Mochila Aurora Trail 30L",
        content="""Mochila técnica 30L em ripstop 600D impermeável (coluna d'água 3000mm). Compartimento principal com acesso top e lateral, divisória pra notebook até 16'', bolso interno com chave-coleira. 2 bolsos externos elásticos pra garrafa. Alças ergonômicas com peitoral e cinto abdominal removíveis. Peso vazio 980g. Capacidade de carga recomendada até 12kg. Cores: cinza-grafite, verde-oliva, preto. SKU AUR-MOC-TR30-005. Preço R$ 349,90."""),

    # ─── POLÍTICAS ─────────────────────────────────────────────────────────
    Row(article_id="pol-001", category="politica", title="Política de privacidade",
        content="""A Aurorinha trata dados pessoais conforme a LGPD. Coletamos: dados cadastrais (nome, documento, e-mail, telefone), endereço de entrega e cobrança, histórico de compras e navegação no site (cookies first-party). Não coletamos dados sensíveis. Compartilhamos dados com: transportadoras (apenas o necessário pra entrega), gateways de pagamento, e ferramentas de analytics anônimas. Você pode solicitar acesso, correção ou exclusão dos seus dados pelo formulário em aurorinha.com.br/privacidade — respondemos em até 15 dias. Mantemos dados de compras por 5 anos por obrigação fiscal. Cookies podem ser desabilitados nas configurações do navegador, mas isso pode afetar o funcionamento do site. O Encarregado de Proteção de Dados (DPO) pode ser contatado pelo mesmo formulário."""),

    Row(article_id="pol-002", category="politica", title="Política de troca e garantia",
        content="""Troca em até 30 dias: produtos sem uso, com etiqueta e embalagem originais, podem ser trocados por outro tamanho/cor ou crédito. Diferença de preço é cobrada/estornada. Garantia legal de 90 dias contra defeitos de fabricação, conforme CDC. Garantia estendida Aurorinha de 6 meses (eletrônicos) ou 12 meses (cafeteiras e utensílios de cozinha) é incluída sem custo adicional — não cobre danos por mau uso, queda, exposição a líquidos ou tentativa de reparo por terceiros. Para acionar garantia: "Meus Pedidos > Defeito > Solicitar assistência". Enviamos código de postagem; após análise técnica (até 15 dias úteis), substituímos, consertamos ou estornamos. Produtos da Aurora marca própria têm garantia estendida de 12 meses por padrão."""),

    Row(article_id="pol-003", category="politica", title="Termos de uso Aurorinha Pay",
        content="""Aurorinha Pay é a carteira digital da Aurorinha, mantida em parceria com uma instituição de pagamento autorizada pelo Banco Central. Você pode adicionar saldo via Pix, cartão de débito ou cashback de compras. Saldo Aurorinha Pay tem validade de 1 ano a partir do último crédito. Cashback de 2% é creditado em D+30 após confirmação da entrega. Saldo NÃO é resgatável em dinheiro — só pode ser usado em compras na Aurorinha. Em caso de cancelamento de pedido pago com Aurorinha Pay, o saldo volta na hora. Limite máximo de saldo: R$ 5.000. Aurorinha Pay não rende juros nem é coberto pelo FGC. Os termos completos estão em aurorinha.com.br/pay/termos."""),

    Row(article_id="pol-004", category="politica", title="Política antifraude",
        content="""Compras acima de R$ 5.000, ou com perfil atípico (endereço novo, CEP de risco, mudança de cartão recente), são analisadas pelo nosso time antifraude em até 24h úteis. Durante a análise, o pedido fica em status "Em verificação" — não cobramos o cartão até aprovar. Podemos solicitar documentos adicionais via link seguro enviado pela área logada. Se houver suspeita confirmada de fraude, cancelamos o pedido e bloqueamos a conta — o titular pode contestar pelo formulário em aurorinha.com.br/contestacao. Compras com cartão clonado: a Aurorinha colabora com investigações policiais e cobre o estorno conforme regulamento do cartão (chargeback)."""),
]

df = spark.createDataFrame(kb_articles)
df.write.mode("overwrite").option("delta.enableChangeDataFeed", "true").saveAsTable(SOURCE_TABLE)

print(f"✅ {df.count()} artigos salvos em {SOURCE_TABLE}")
display(df.select("article_id", "category", "title"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Cria eval dataset
# MAGIC
# MAGIC Perguntas com **ground truth** — gabarito + qual artigo deveria ser usado.
# MAGIC Em produção, esse set viria de tickets de suporte / feedback / curadoria humana.

# COMMAND ----------

eval_examples = [
    {"request": "Quantos dias eu tenho pra devolver um produto?",
     "expected_response": "7 dias corridos a partir do recebimento.",
     "expected_retrieved_context": [{"doc_uri": "faq-002"}]},

    {"request": "Como faço pra rastrear meu pedido pelo WhatsApp?",
     "expected_response": "Envie 'rastrear' para o canal oficial Aurorinha (wa.aurorinha.com.br) e informe o número do pedido.",
     "expected_retrieved_context": [{"doc_uri": "faq-001"}]},

    {"request": "Tem desconto pra pagar no Pix?",
     "expected_response": "Sim, 5% de desconto à vista no Pix.",
     "expected_retrieved_context": [{"doc_uri": "faq-003"}]},

    {"request": "Quanto custa frete pra capital do Nordeste?",
     "expected_response": "R$ 19,90 a R$ 49,90, prazo 4-7 dias úteis.",
     "expected_retrieved_context": [{"doc_uri": "faq-004"}]},

    {"request": "O que é a Aurorinha Prime?",
     "expected_response": "Assinatura mensal de R$ 19,90 com frete grátis ilimitado, acesso antecipado a promoções e desconto extra.",
     "expected_retrieved_context": [{"doc_uri": "faq-005"}]},

    {"request": "Posso usar a capa do iPhone 15 com MagSafe?",
     "expected_response": "Sim, ímãs N52 garantem alinhamento com carregadores e acessórios MagSafe oficiais.",
     "expected_retrieved_context": [{"doc_uri": "prod-001"}]},

    {"request": "Qual a autonomia do fone Aurora Sound Pro?",
     "expected_response": "8h por carga, 36h com o estojo. Carga rápida: 15min = 3h de uso.",
     "expected_retrieved_context": [{"doc_uri": "prod-002"}]},

    {"request": "Como devo cuidar da cuia de chimarrão antes de usar?",
     "expected_response": "Hidratar por 24h com água morna antes do primeiro chimarrão; não lavar com detergente.",
     "expected_retrieved_context": [{"doc_uri": "prod-003"}]},

    {"request": "A cafeteira italiana funciona em fogão de indução?",
     "expected_response": "Sim, compatível com indução, gás, elétrico e vitrocerâmico.",
     "expected_retrieved_context": [{"doc_uri": "prod-004"}]},

    {"request": "Quantos litros tem a mochila Trail?",
     "expected_response": "30 litros.",
     "expected_retrieved_context": [{"doc_uri": "prod-005"}]},

    {"request": "Por quanto tempo a Aurorinha guarda meus dados?",
     "expected_response": "Dados de compras por 5 anos por obrigação fiscal.",
     "expected_retrieved_context": [{"doc_uri": "pol-001"}]},

    {"request": "Qual a garantia de uma cafeteira da Aurora?",
     "expected_response": "12 meses por ser marca própria (Aurora), além dos 90 dias de garantia legal.",
     "expected_retrieved_context": [{"doc_uri": "pol-002"}]},

    {"request": "Posso resgatar o saldo do Aurorinha Pay em dinheiro?",
     "expected_response": "Não — só pode ser usado em compras na Aurorinha.",
     "expected_retrieved_context": [{"doc_uri": "pol-003"}]},

    {"request": "Por que meu pedido tá 'em verificação'?",
     "expected_response": "Pedidos acima de R$ 5.000 ou com perfil atípico passam por análise antifraude em até 24h úteis.",
     "expected_retrieved_context": [{"doc_uri": "pol-004"}]},

    {"request": "Tem cashback na Aurorinha?",
     "expected_response": "Sim, 2% de cashback nas compras pagas com Aurorinha Pay, creditado em D+30.",
     "expected_retrieved_context": [{"doc_uri": "faq-003"}, {"doc_uri": "pol-003"}]},
]

import pandas as pd
eval_df = pd.DataFrame(eval_examples)
spark.createDataFrame(eval_df).write.mode("overwrite").saveAsTable(EVAL_TABLE)

print(f"✅ {len(eval_df)} exemplos de avaliação salvos em {EVAL_TABLE}")
display(spark.table(EVAL_TABLE).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. ⚡ Dispara criação do Vector Search Endpoint (em background)
# MAGIC
# MAGIC Cold start do endpoint leva **5-10 min**. Criamos agora pra ele tá pronto quando você chegar no Lab 2 — você vai pro Lab 1 enquanto provisiona.

# COMMAND ----------

# MAGIC %pip install -q databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../config/00_config

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vs_client = VectorSearchClient(disable_notice=True)

try:
    info = vs_client.get_endpoint(VS_ENDPOINT_NAME)
    state = info.get("endpoint_status", {}).get("state", "?")
    print(f"✅ Endpoint '{VS_ENDPOINT_NAME}' já existe — state={state}")
except Exception:
    print(f"⏳ Criando endpoint '{VS_ENDPOINT_NAME}' (5-10 min cold start em background)...")
    vs_client.create_endpoint(name=VS_ENDPOINT_NAME, endpoint_type="STANDARD")
    print(f"✅ Criação disparada. NÃO bloqueia esse notebook — vai pro Lab 1.")
    print(f"   O Lab 2 verifica e aguarda o endpoint ficar ONLINE antes de criar o index.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Setup concluído
# MAGIC
# MAGIC | Asset | Local |
# MAGIC |---|---|
# MAGIC | KB Articles | `{SOURCE_TABLE}` |
# MAGIC | Eval set | `{EVAL_TABLE}` |
# MAGIC | VS Endpoint | `{VS_ENDPOINT_NAME}` (provisionando em background) |
# MAGIC
# MAGIC ➡️ Próximo: [`01_chunking_embeddings`](./01_chunking_embeddings) — não precisa esperar o endpoint, esse lab roda em paralelo.
