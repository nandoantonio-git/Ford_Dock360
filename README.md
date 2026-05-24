# Ford VinGuard

Projeto de Machine Learning para prever risco de churn na rede Ford Brasil e apoiar a ficha de abordagem comercial (Dock 360) com perfil de cliente, probabilidade de churn e ação recomendada.


## Objetivo

O Ford VinGuard identifica clientes com maior probabilidade de abandonar a rede de manutenção (churn definido como 18 meses sem serviços). A API retorna:

- Predição de churn: `churn` ou `no_churn`;
- Probabilidade calibrada de churn;
- Nível de risco: `low`, `medium` ou `high`;
- Perfil previsto via clustering comportamental;
- Ação recomendada para abordagem comercial.

## Estrutura do Projeto

```text
.
|-- data/
|   |-- raw/                  # Dataset real (.xlsx)
|   `-- processed/            # VINs agregados e labels de cluster
|-- models/                   # Modelos joblib e checksums sha256
|-- notebooks/                # Notebooks de EDA e Pipeline Completo
|-- reports/                  # Gráficos e relatórios de métricas
|-- src/
|   |-- api/                  # API FastAPI 
|   `-- pipeline/             # Scripts de engenharia e treino (Real)
|-- tests/                    # Testes de leakage, saúde e schema
|-- requirements.txt
`-- README.md
```

## Principais Arquivos

### Pipeline Real

- `src/pipeline/feature_engineering_real.py`: Agrega 600k+ ordens de serviço da base bruta com 175k+ VINs únicos e gera a base modelável final com 118k+ VINs após filtros de qualidade e janela temporal.
- `src/pipeline/clustering_real.py`: Executa K-Means em features comportamentais.
- `src/pipeline/train_churn_real.py`: Treina o modelo calibrado de churn.
- `src/pipeline/mlflow_tracking.py`: Registra experimentos no MLflow.
- `src/pipeline/config.py`: Centraliza constantes e lista de anti-leakage.

### API

- `src/api/main.py`: Ponto de entrada FastAPI.
- `src/api/services/predictor.py`: Lógica de inferência.
- `src/api/models/schemas.py`: Contratos Pydantic com exemplos reais.

## Regra Crítica: Data Leakage

O projeto é de pós-venda. As features de entrada são calculadas somente até a
data de corte (`DATA_CORTE`) e o target de churn usa apenas eventos futuros.

Nunca use no X informações pós-corte, futuro ou target, como
`churn_futuro_18m`, `voltou_pos_corte`, `primeiro_servico_pos_corte`,
`ultimo_servico_pos_corte` ou `qtd_servicos_pos_corte`.

O modelo de churn de produção espera as features comportamentais até corte:
`ano_modelo`, `qtde_revisoes_ate_corte`,
`meses_desde_ultimo_servico_ate_corte`,
`meses_relacionamento_ate_corte`, `n_dealers_usados_ate_corte`,
`km_max_ate_corte`, `pct_agenda_ate_corte`,
`intervalo_medio_revisoes_dias_ate_corte`,
`dias_ate_primeira_revisao`, `idade_veiculo_meses_ate_corte` e `modelo`.

## Como Reproduzir o Pipeline

Execute os scripts na ordem abaixo a partir da raiz:

### 1. Engenharia de Features
```bash
python -m src.pipeline.feature_engineering_real
```
Gera `data/processed/snapshots_pos_venda.csv` e
`data/processed/dataset_churn_pos_venda.csv`.

### 2. Segmentação (Clustering)
```bash
python -m src.pipeline.clustering_real
```
Gera `data/processed/segmentos_pos_venda.csv`.

### 3. Treinamento
```bash
python -m src.pipeline.train_churn_real
```
Gera os arquivos `.joblib` em `models/`.

### 4. Tracking MLflow Opcional
```bash
python -m src.pipeline.mlflow_tracking
mlflow ui --backend-store-uri ./mlruns
```
Registra os artefatos já gerados dos experimentos de churn, segmentação
K-Means. Esse passo é evidência de experimentação; não é necessário para a API
em runtime.

## Notas de Entrega

- Arquivos de experimentos antigos foram removidos do escopo atual.
- O tracking MLflow é apenas evidência de avaliação; a API não depende dele em
  runtime.
- Os notebooks de EDA geram gráficos de apoio em `reports/`, mas não são passos
  obrigatórios do pipeline produtivo.
- Evite reexecutar treino sem necessidade antes da entrega, pois isso altera
  modelos e checksums.

## API FastAPI

Endpoints oficiais da API:

- `GET /health`
- `POST /predict`
- `POST /predict-batch`

O endpoint `/predict/batch` é mantido como alias de compatibilidade.

### Subir a API
```bash
export SECRET_KEY="sua-chave-secreta-de-pelo-menos-32-caracteres"
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Exemplo de Request (/predict)
```json
{
  "features": {
    "ano_modelo": 2020,
    "qtde_revisoes_ate_corte": 2,
    "meses_desde_ultimo_servico_ate_corte": 14.2,
    "meses_relacionamento_ate_corte": 48.0,
    "n_dealers_usados_ate_corte": 1,
    "km_max_ate_corte": 48200,
    "pct_agenda_ate_corte": 0.65,
    "intervalo_medio_revisoes_dias_ate_corte": 220.0,
    "dias_ate_primeira_revisao": 180,
    "idade_veiculo_meses_ate_corte": 54.0,
    "modelo": "KA"
  }
}
```

### Autenticação via JWT

`/health` e `/` são públicos. Os endpoints `/predict`, `/predict-batch` e
`/predict/batch` aceitam dois caminhos de autenticação:

- `Authorization: Bearer <jwt>`: JWT de usuário/demo com role `analyst` ou
  `admin`. O token gerado por `/auth/demo-token` expira conforme
  `ACCESS_TOKEN_EXPIRE_MINUTES` e deve ser usado apenas para Swagger/demo
  manual.
- `X-ML-Service-Token: <token>`: token fixo server-to-server para o Java BFF.
  Esse token não expira pela FastAPI e deve vir somente de secret/variável de
  ambiente. Não registre esse valor em logs, exemplos ou arquivos versionados.

Para compatibilidade técnica, a FastAPI também aceita o service token como
`Authorization: Bearer <service_token>`, mas o header preferido para o BFF é
`X-ML-Service-Token`.

Para gerar um token local usando o mesmo `SECRET_KEY` da API:

```bash
python scripts/create_demo_token.py --role analyst
```

Para gerar token no ambiente publicado, configure `DEMO_TOKEN_SECRET` no Render
e chame:

```bash
curl -X POST "https://ford-vinguard-api.onrender.com/auth/demo-token?role=analyst" \
  -H "X-Demo-Token-Secret: <DEMO_TOKEN_SECRET>"
```

Use o `access_token` retornado como `Authorization: Bearer <token>` na FastAPI
ou como header `X-ML-Demo-Token` no BFF Java para teste manual. O Java BFF não
deve depender desse token demo para produção.

Configure o service token fixo no ambiente da FastAPI:

```env
ML_SERVICE_TOKEN=<service-token-fixo-gerado-fora-do-repositorio>
# fallback aceito, se você quiser usar o mesmo nome no Java e na FastAPI:
FORD_ML_SERVICE_TOKEN=<service-token-fixo-gerado-fora-do-repositorio>
```

No Java BFF, configure o mesmo valor como `FORD_ML_SERVICE_TOKEN`; o BFF envia
esse segredo para a FastAPI com `X-ML-Service-Token`.

## Deploy no Render

O projeto está preparado para deploy como Web Service Python no Render via
`render.yaml`.

Configuração usada:

- Build Command: `pip install --upgrade pip && pip install -r requirements-api.txt && python scripts/fetch_model_artifacts.py`
- Start Command: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
- Health Check Path: `/`
- Python: `3.11.11`

Variáveis criadas/configuradas no Render:

- `SECRET_KEY`: gerada pelo Render, obrigatória para JWT
- `JWT_ISSUER`: `ford-vinguard-api`
- `JWT_AUDIENCE`: `ford-vinguard-api`
- `ACCESS_TOKEN_EXPIRE_MINUTES`: validade dos JWTs de demo
- `DEMO_TOKEN_SECRET`: segredo opcional para habilitar `/auth/demo-token`
- `ML_SERVICE_TOKEN`: service token fixo para Java BFF -> FastAPI ML
- `FORD_ML_SERVICE_TOKEN`: fallback aceito para o service token fixo
- `MODELS_DIR`: `models`
- `CHURN_MODEL_FILENAME`: `churn_pos_venda_rf_calibrated.joblib`
- `PERFIL_MODEL_FILENAME`: `kmeans_segmentador_pos_venda.joblib`
- `CHURN_MODEL_URL`: URL privada/pública para baixar o `.joblib` de churn
- `PERFIL_MODEL_URL`: URL privada/pública para baixar o `.joblib` de perfil
- `CHURN_MODEL_SHA256`: checksum esperado do modelo de churn
- `PERFIL_MODEL_SHA256`: checksum esperado do segmentador K-Means

Observação importante: `models/` e `data/` não são versionados. Por isso, o
deploy tem dois níveis:

- Sem URLs de modelo: `/` e `/docs` sobem para demonstrar a API; `/health`
  retorna `503 degraded` porque os artefatos ainda não existem.
- Com `CHURN_MODEL_URL` e `PERFIL_MODEL_URL`: o script
  `scripts/fetch_model_artifacts.py` baixa os `.joblib` para `MODELS_DIR` antes
  do Uvicorn iniciar. Com os checksums corretos, `/health`, `/predict` e
  `/predict-batch` e `/predict/batch` ficam prontos para demo.

As URLs dos artefatos devem ser configuradas no painel do Render ou preenchidas
no fluxo inicial do Blueprint. Não commitar URLs assinadas, tokens ou datasets
reais no repositório.

## Deploy no Azure Container Apps via Dockerfile

Para evitar o build automático da Azure/Oryx, o projeto também possui um
`Dockerfile` explícito na raiz. A imagem roda a FastAPI na porta `8000` por
padrão e respeita a variável `PORT` quando a plataforma definir outro valor.

Fluxo do container:

```text
pip install -r requirements-api.txt
python scripts/fetch_model_artifacts.py
uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

Arquivos e pastas pesadas ou sensíveis ficam fora do contexto Docker via
`.dockerignore`, incluindo `.env`, `data/`, `models/`, `notebooks/`,
`reports/`, `tests/`, `__pycache__` e `.git`.

Variáveis obrigatórias/recomendadas no Azure Container Apps:

```env
PORT=8000
SECRET_KEY=<chave-com-pelo-menos-32-caracteres>
DEMO_TOKEN_SECRET=<segredo-para-gerar-token-demo>
ML_SERVICE_TOKEN=<service-token-fixo-do-bff>
MODELS_DIR=models
CHURN_MODEL_FILENAME=churn_pos_venda_rf_calibrated.joblib
PERFIL_MODEL_FILENAME=kmeans_segmentador_pos_venda.joblib
CHURN_MODEL_URL=<url-do-artefato-churn>
PERFIL_MODEL_URL=<url-do-artefato-perfil>
CHURN_MODEL_SHA256=<sha256-do-artefato-churn>
PERFIL_MODEL_SHA256=<sha256-do-artefato-perfil>
JWT_ISSUER=ford-vinguard-api
JWT_AUDIENCE=ford-vinguard-api
```

Exemplo de build local:

```bash
docker build -t ford-vinguard-ml-api .
```

Exemplo de execução local sem gravar segredos no repositório:

```bash
docker run --rm -p 8000:8000 \
  -e PORT=8000 \
  -e SECRET_KEY="troque-por-uma-chave-com-32-caracteres" \
  -e DEMO_TOKEN_SECRET="troque-por-um-segredo-de-demo" \
  -e ML_SERVICE_TOKEN="<service-token-fixo-do-bff>" \
  -e MODELS_DIR=models \
  -e CHURN_MODEL_FILENAME=churn_pos_venda_rf_calibrated.joblib \
  -e PERFIL_MODEL_FILENAME=kmeans_segmentador_pos_venda.joblib \
  -e CHURN_MODEL_URL="<url-do-artefato-churn>" \
  -e PERFIL_MODEL_URL="<url-do-artefato-perfil>" \
  -e CHURN_MODEL_SHA256="<sha256-do-artefato-churn>" \
  -e PERFIL_MODEL_SHA256="<sha256-do-artefato-perfil>" \
  -e JWT_ISSUER=ford-vinguard-api \
  -e JWT_AUDIENCE=ford-vinguard-api \
  ford-vinguard-ml-api
```

No Azure Container Apps, configure ingress HTTP externo para a porta alvo
`8000`. O endpoint `/` valida liveness, `/health` valida a disponibilidade dos
modelos e `/docs` abre o Swagger.

Não commitar `.env`, datasets reais, arquivos em `data/raw`,
`data/processed`, `models/*.joblib`, URLs assinadas ou segredos usados no Azure.

## Observação para Avaliação Acadêmica

Por questões de tamanho e segurança, os datasets reais e artefatos `.joblib` não são versionados no GitHub. Para reprodução completa, é necessário disponibilizar os arquivos em `data/raw/`, configurar `CHURN_MODEL_URL` e `PERFIL_MODEL_URL` com os artefatos treinados, ou executar o pipeline completo a partir dos dados brutos autorizados. Sem os artefatos de modelo, a API sobe parcialmente para inspeção do Swagger, mas os endpoints de predição dependem dos arquivos `.joblib`.

## Testes
```bash
pytest tests/ -v
```
