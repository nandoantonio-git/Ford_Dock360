# Entrega IA/ML — Sprint 3

## 1. Compreensão do problema — 1,5 ponto

O Ford VinGuard estima abandono da rede autorizada a partir do histórico de
manutenção por VIN, apoiando retenção no Dock 360. É classificação binária:
`churn_futuro_18m=1` significa nenhum serviço em `(DATA_CORTE, corte + 18 meses]`.
Isso não comprova migração para concorrentes. K-Means complementa a solução
com segmentação comportamental não supervisionada.

Fonte: `data/raw/vin_share_Desafio_02.xlsx`, aba `vin_share`, com 602.788 ordens
e 175.554 VINs anonimizados. Cada linha bruta é um evento; cada linha modelável
é um VIN. Datas, KM, dealer, agendamento e modelo permitem caracterizar uso,
frequência e recência. O notebook registra 118.375 VINs após saneamento,
corte em 2024-10-31 e janela futura até 2026-04-30.

## 2. Preparação dos dados — 2,0 pontos

`feature_engineering_real.py` trata datas e converte KM negativo ou superior
a 500.000 em ausente. Remove VINs com venda posterior ao corte e trata
intervalos negativos como ausentes. A execução salva removeu quatro VINs.
X usa histórico até o corte; y usa somente eventos da janela futura.

As dez variáveis numéricas são ano-modelo, quantidade de revisões, recência,
duração do relacionamento, número de dealers, KM máximo, proporção de
agendamentos, intervalo médio de revisões, dias até primeira revisão e idade
do veículo. A categoria é `modelo`; a lista está em `src/pipeline/config.py`.

Dentro de `sklearn.Pipeline`, numéricas recebem imputação pela mediana e
`StandardScaler`; categorias recebem moda e OneHotEncoder que aceita categorias
desconhecidas. O holdout de churn usa 80% treino e 20% teste, `stratify=y` e
`random_state=42`. A calibração usa cinco folds internos ao treino.
Separar features e target no tempo não transforma esse holdout aleatório em
validação temporal de generalização.

## 3. Desenvolvimento dos modelos — 3,0 pontos

Regressão logística fornece baseline linear; Random Forest captura relações
não lineares; calibração isotônica busca probabilidades úteis à priorização.
Todos os classificadores usam `class_weight="balanced"`.

O comparador em `mlflow_tracking.py` treina no mesmo split:

- Logistic Regression: `max_iter=1000`;
- RF: 50 árvores e `max_depth=8`;
- RF com a mesma configuração e calibração isotônica, `cv=5`.

São configurações fixas comparadas, não uma busca exaustiva de hiperparâmetros.
O treino destinado à API é separado: 200 árvores, `min_samples_leaf=5`,
`max_features="sqrt"`, calibração isotônica com cinco folds.
K-Means compara `k=2..8` por inertia e silhouette e usa quatro clusters por
configuração. O gráfico não prova que quatro é o ótimo estatístico; a escolha
também precisa de validação de utilidade e estabilidade dos segmentos.

## 4. Avaliação e comparação — 2,0 pontos

Resultados de `reports/model_comparison_churn.csv`:

| Modelo | AUC-ROC | AP (`auc_pr`) | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| RF balanceado, 50 árvores | 0,940834 | 0,966649 | 0,945275 | 0,831404 | 0,884690 |
| RF calibrado, 50 árvores | 0,941116 | 0,966619 | 0,905737 | 0,891047 | 0,898332 |
| Regressão logística | 0,937921 | 0,965085 | 0,933918 | 0,841999 | 0,885580 |

AUC-ROC é a métrica principal de avaliação de churn. A coluna `auc_pr` usa
`average_precision_score` (AP), não integração trapezoidal da curva PR.
O comparador implementado ordena por AP, com AUC-ROC no desempate, e marca RF
balanceado como selecionado. O calibrado tem maior AUC-ROC, recall e F1.
A diferença de AP é apenas 0,000030: sem intervalos de confiança, não demonstra
superioridade estatística. O churn é aproximadamente 64,6% do snapshot;
portanto a classe positiva não é rara.

O notebook do treino separado de 200 árvores registra AUC-ROC 0,9438.
Não confundir esse experimento com o CSV. As evidências são resultados salvos,
não uma nova execução realizada nesta revisão. Há um único snapshot, holdout
aleatório e seleção no mesmo holdout; falta teste temporal independente.
Não há prova de impacto causal na retenção nem avaliação de calibração por
Brier score/curva de confiabilidade. Precision e recall devem ser relacionados
ao custo de contatos e oportunidades perdidas.

## 6. Conclusão — 1,5 ponto

A solução final de inferência mantida é **RF calibrado de 200 árvores + K-Means**.
A escolha operacional preserva o contrato da API e produz probabilidades de
risco; seu notebook registra AUC-ROC 0,9438. É uma escolha provisória, não prova
de superioridade: o comparador seleciona RF balanceado de 50 árvores por AP,
sem exportação ou promoção automática desse vencedor.

O BFF pode enviar features à FastAPI e usar risco, perfil e ação recomendada
para apoiar o consultor. Deploy possível: Render ou container no Azure, com
artefatos confiáveis/checksums, segredos no ambiente e `cluster_segment_map.json`
ao lado do segmentador. A documentação não comprova uso em produção.

Melhorias: múltiplos cortes, teste temporal independente, tuning dentro do
treino, intervalos de confiança, avaliação da calibração e limiares por
capacidade de atendimento. Gate de promoção, PSI e rollback automatizado são
trabalhos futuros; a política escrita não equivale à implementação.

## Entregáveis e reprodução

| Entregável | Evidência |
|---|---|
| Notebook/código | `notebooks/00_pipeline_completo.ipynb`, `src/pipeline/` |
| Análise exploratória | `notebooks/01_eda_base1.ipynb`, `notebooks/02_eda_base2.ipynb` |
| Documentação resumida | este documento e `README.md` |
| Comparação e métricas | `reports/model_comparison_churn.csv`, gráficos em `reports/` |
| Conclusão e justificativa | seção 6 |
| Notas do release | `docs/release_sprint3.md` |

Instale `requirements.txt` e `requirements-dev.txt` em ambiente virtual e
obtenha a base por canal autorizado. Na raiz, execute:

```bash
python -m src.pipeline.feature_engineering_real
python -m src.pipeline.clustering_real
python -m src.pipeline.train_churn_real
python -m src.pipeline.mlflow_tracking --only churn-comparison --comparison-output reports/model_comparison_churn.csv
python -m pytest tests/ -q
```

Use cópia isolada: a execução escreve modelos e dados processados. O release
não inclui base nem modelos. Sem a base autorizada, a banca pode inspecionar
código/evidências, mas não reproduzir integralmente o treino. O grupo deve
confirmar a correspondência dos algoritmos com o conteúdo das aulas, que não
foi fornecido para esta revisão.
