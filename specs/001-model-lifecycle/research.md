# Research: Ciclo de Vida do Modelo Ford VinGuard

## Decision 1: Maturação do rótulo deve ser validada por corte

**Decision**: Um `DATA_CORTE` candidato só é elegível quando `DATA_CORTE + 18 meses <= max(ServiceDate)` disponível no dataset.

**Rationale**: O target de churn é “não retornou em até 18 meses após o corte”. Logo, a janela futura completa precisa estar observável para todos os VINs elegíveis pelo corte. Filtrar VINs por “último evento conhecido há 18 meses” seleciona justamente casos que aparentam abandono e introduz viés silencioso.

**Alternatives considered**:

- Filtrar por VIN com último evento há 18 meses: rejeitado por viés de seleção.
- Empurrar corte semanalmente: rejeitado porque gera rótulos imaturos.
- Online learning: rejeitado por incompatibilidade com latência de rótulo e escopo acadêmico.

## Decision 2: Retreino programado trimestral com monitoramento contínuo de drift

**Decision**: Usar cadência trimestral para retreino e monitorar drift de forma recorrente entre ciclos.

**Rationale**: “Retreino contínuo” sugere atualização online ou frequente, o que conflita com churn de 18 meses. A cadência trimestral é simples, demonstrável e evita retreino prematuro; o monitoramento de drift mantém vigilância entre ciclos.

**Alternatives considered**:

- Retreino semanal: rejeitado por alto risco de rótulo imaturo.
- Retreino manual sem cadência: rejeitado por fragilidade operacional.
- Retreino anual: possível, mas menos convincente para responder ao professor sobre atualização constante.

## Decision 3: Gate de promoção usa AUC-PR com margem mínima

**Decision**: AUC-ROC continua reportada, mas AUC-PR decide promoção. O challenger só é aprovado se `auc_pr_challenger >= auc_pr_champion + 0.005`.

**Rationale**: O uso de churn serve para priorizar clientes em risco, geralmente em cenário desbalanceado. AUC-PR é mais sensível à qualidade de ranking da classe positiva. A margem mínima evita promover modelos equivalentes por ruído.

**Alternatives considered**:

- Accuracy: rejeitada por inadequação a classes desbalanceadas.
- AUC-ROC como gate único: mantida como métrica de compatibilidade, mas menos alinhada à priorização operacional.
- Qualquer empate promove: rejeitado por instabilidade e pouca disciplina de champion/challenger.

## Decision 4: Holdout temporal fixo e versionado

**Decision**: Champion e challenger devem ser comparados no mesmo holdout temporal fixo e versionado.

**Rationale**: Comparações em holdouts diferentes misturam ganho real com mudança de amostra. Um holdout fixo melhora reprodutibilidade e facilita evidência perante a banca.

**Alternatives considered**:

- Gerar novo split em cada ciclo: rejeitado por instabilidade.
- Avaliar só no split interno do treino: rejeitado porque não compara champion e challenger em base comum.

## Decision 5: Gate report-only por padrão e promoção explícita

**Decision**: `promotion_gate.py` deve gerar decisão e evidência por padrão, mas só copiar artefatos com uma autorização explícita adicional, como `--promote`.

**Rationale**: Menor privilégio reduz risco de promoção acidental. A automação deve validar, evidenciar e recomendar; a troca efetiva de produção exige ato explícito.

**Alternatives considered**:

- Promover automaticamente quando métricas passam: rejeitado por excesso de privilégio.
- Promoção totalmente manual sem gate: rejeitado por falta de automação de segurança.

## Decision 6: Evidência dupla em JSON e MLflow

**Decision**: Cada ciclo registra evidência em JSON versionável e em MLflow.

**Rationale**: MLflow facilita visualização e histórico de experimentos. JSON é simples, auditável, diffável e continua disponível mesmo se a UI do MLflow falhar durante a demo.

**Alternatives considered**:

- Somente MLflow: rejeitado por dependência de ferramenta/servidor.
- Somente JSON: aceito como mínimo, mas menos visual para apresentação.

## Decision 7: Rollback por versão aprovada e checksum

**Decision**: Rollback só pode apontar para artefato anterior aprovado, documentado em `CHANGELOG_MODELOS.md`, com URL e SHA256.

**Rationale**: “Backup local” não prova origem, integridade nem aprovação. Checksum e changelog dão rastreabilidade adequada ao escopo acadêmico.

**Alternatives considered**:

- Restaurar último arquivo local: rejeitado por falta de governança.
- Re-treinar modelo anterior sob demanda: rejeitado por não garantir equivalência do artefato.

## Decision 8: Java BFF como consumidor técnico único

**Decision**: A API de ML deve ser consumida tecnicamente pelo Java BFF, autenticado por token de serviço. A UI e o consultor não chamam ML diretamente.

**Rationale**: Menor privilégio e melhor separação entre decisão técnica e experiência de negócio. O consultor recebe `risk_level`, `perfil_previsto`, `acao_recomendada` e prioridade, não probabilidade crua.

**Alternatives considered**:

- UI chama ML diretamente: rejeitado por exposição de token e acoplamento.
- Consultor vê probabilidade crua: rejeitado por baixa ação operacional e risco de má interpretação.
