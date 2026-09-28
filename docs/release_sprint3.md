# Ford VinGuard — entrega acadêmica IA/ML

Versão proposta: `v1.5-sprint3-ia-ml`. Status: preparação para revisão.

Esta entrega apoia retenção no pós-venda por risco de abandono da rede e
segmentos comportamentais. A [matriz da rubrica](ia_ml_requisitos.md) apresenta
problema, preparação, desenvolvimento, avaliação e conclusão.

## Novidades e melhorias

- Comparação de regressão logística, RF e RF calibrado, com métricas em CSV e MLflow.
- Retreino de candidatos com verificações de autorização, cadência e maturação.
- Ampliação de testes de pipeline, autenticação e predição.
- Evidências visuais e notebook atualizado com 118.375 VINs após saneamento.
- Documentação distingue modelo vencedor do comparador e modelo de inferência,
  reconhece limitações de validação e descreve dependências de deploy.

## Resultados e limites

RF calibrado de 50 árvores: AUC-ROC 0,941116 e F1 0,898332. RF balanceado:
AP 0,966649. O treino separado de 200 árvores registra AUC-ROC 0,9438.
Esses números vêm de evidências salvas, com holdout aleatório, sem teste temporal
independente nem significância estatística demonstrada.

## Evidências visuais da entrega

As figuras abaixo são as mesmas versionadas no README, com links absolutos
fixados ao commit `624d62565ba60421e1bdd7c3b6ff30f588408371`. Assim, também
funcionam quando estas notas são usadas no corpo de um GitHub Release e
preservam os resultados apresentados mesmo após novas execuções do pipeline.

### Compreensão e preparação: distribuição do target

![Distribuição do churn futuro em 18 meses: 35,4% sem churn e 64,6% com churn](https://raw.githubusercontent.com/nandoantonio-git/Ford_Dock360/624d62565ba60421e1bdd7c3b6ff30f588408371/reports/base2_distribuicao_churn.png)

O target representa ausência de retorno à rede dentro da janela futura de
18 meses. A classe positiva corresponde a 64,6% do snapshot; não é uma classe
rara. Essa distribuição contextualiza as métricas de classificação.

### Desenvolvimento: comparação das configurações de K-Means

![Inertia e silhouette para K-Means com dois a oito clusters](https://raw.githubusercontent.com/nandoantonio-git/Ford_Dock360/624d62565ba60421e1bdd7c3b6ff30f588408371/reports/elbow_silhouette_pos_venda.png)

A comparação explora `k=2..8`. O maior silhouette mostrado ocorre em `k=8`;
os quatro segmentos configurados na solução não são o máximo dessa métrica.
A escolha de quatro precisa ser avaliada pela utilidade e estabilidade dos
grupos para ações de retenção.

![Projeção PCA dos segmentos inativo, recorrente, multidealer e baixo engajamento](https://raw.githubusercontent.com/nandoantonio-git/Ford_Dock360/624d62565ba60421e1bdd7c3b6ff30f588408371/reports/clusters_pca_pos_venda.png)

A projeção permite visualizar os quatro segmentos. Os dois componentes
explicam aproximadamente 58,2% da variância (39,7% + 18,5%); sobreposição ou
separação no plano não substitui a avaliação no espaço completo das features.

### Avaliação: classificação de churn

![Curva Precision–Recall do modelo de churn, com AP arredondada de 0,97](https://raw.githubusercontent.com/nandoantonio-git/Ford_Dock360/624d62565ba60421e1bdd7c3b6ff30f588408371/reports/precision_recall_churn_pos_venda.png)

A curva mostra a relação entre precisão e recall ao variar o limiar de
classificação. Atenção à nomenclatura do gráfico original: `AUC = 0.944` no
título corresponde à **AUC-ROC**, enquanto `AP = 0.97` na legenda é average
precision. São métricas distintas. Esta figura pertence ao treino separado
de churn; não representa as três linhas do CSV de comparação de candidatos.

![Matriz de confusão: 6867 verdadeiros negativos, 1517 falsos positivos, 1469 falsos negativos e 13822 verdadeiros positivos](https://raw.githubusercontent.com/nandoantonio-git/Ford_Dock360/624d62565ba60421e1bdd7c3b6ff30f588408371/reports/confusion_matrix_churn_pos_venda.png)

No holdout ilustrado de 23.675 VINs, há 1.517 falsos positivos e 1.469 falsos
negativos. Esses erros ajudam a discutir contatos desnecessários e clientes
em risco que a ação de retenção deixaria de priorizar. O resultado é do
experimento salvo, não uma estimativa comprovada de desempenho futuro.

As demais imagens de EDA permanecem no README como material complementar;
estas cinco cobrem os pontos visuais diretamente relacionados à rubrica.

Distribuição: código, notebooks, documentação e resultados agregados.
Datasets, segredos e modelos não fazem parte do release. Reprodução integral
depende da base autorizada; inferência depende de modelos e mapeamento de
clusters correspondentes. O retreino não agenda sozinho sua execução.
Gate de promoção, drift e rollback automatizado ainda não estão implementados.

## Validação

Parâmetros e métricas documentados foram conferidos contra código e CSV.
A tentativa anterior de pytest foi interrompida sem resultado conclusivo;
esta revisão documental não declara a suíte aprovada. Antes de publicar uma
versão validada, registrar uma execução concluída dos testes e confirmar o
commit/tag da entrega. Não houve promoção de modelos.
