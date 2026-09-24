# Feature Specification: Ciclo de Vida do Modelo Ford VinGuard

**Feature Branch**: `main`  
**Created**: 2026-09-23  
**Status**: Draft  
**Input**: Plano grillado para retreino programado, gate champion/challenger, evidências MLflow + JSON, drift, rollback e consumidores das predições.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Demonstrar retreino seguro e temporalmente válido (Priority: P1)

Como responsável de ML do Ford VinGuard, quero demonstrar um ciclo de retreino programado que respeita a maturação de 18 meses do rótulo de churn, para que a banca veja que o modelo não é atualizado com rótulos parcialmente observados.

**Why this priority**: A crítica principal do avaliador é sobre atualização contínua do modelo. Sem uma política temporal correta, o ciclo de vida fica tecnicamente frágil.

**Independent Test**: Dado um novo corte candidato, verificar se o ciclo aceita apenas cortes cuja janela futura de 18 meses esteja totalmente observada no dataset disponível.

**Acceptance Scenarios**:

1. **Given** um corte candidato cuja janela futura de 18 meses está coberta pelos dados disponíveis, **When** o ciclo é solicitado com autorização válida, **Then** o sistema permite gerar artefatos candidatos isolados.
2. **Given** um corte candidato cuja janela futura de 18 meses não está coberta pelos dados disponíveis, **When** o ciclo é solicitado, **Then** o sistema rejeita o ciclo sem gerar ou promover artefatos de produção.
3. **Given** uma solicitação de retreino sem autorização explícita, **When** o ciclo é solicitado, **Then** o sistema falha fechado e registra a ausência de autorização quando possível.

---

### User Story 2 - Decidir promoção champion/challenger com evidências (Priority: P1)

Como responsável de ML, quero comparar um candidato contra o modelo champion usando um holdout temporal fixo e versionado, para que a promoção seja baseada em evidência mensurável e reproduzível.

**Why this priority**: A banca precisa ver que o novo modelo não substitui o atual por acaso, por ruído de amostra ou sem validação anti-leakage.

**Independent Test**: Executar o gate em modo relatório e verificar que ele gera decisão, métricas, checks de segurança e evidência persistente sem alterar produção por padrão.

**Acceptance Scenarios**:

1. **Given** um candidato válido, um champion válido e um holdout temporal fixo, **When** o gate é executado sem autorização de promoção, **Then** ele gera decisão e evidência, mas não troca artefatos de produção.
2. **Given** um candidato com AUC-PR inferior ou empate técnico em relação ao champion, **When** o gate é executado, **Then** o champion é mantido e a rejeição é registrada.
3. **Given** um candidato com ganho mínimo de AUC-PR e verificações críticas aprovadas, **When** o gate é executado com autorização explícita de promoção, **Then** o candidato pode substituir o champion e a decisão é registrada.
4. **Given** falha em leakage, holdout ausente, checksum inválido ou autorização ausente, **When** o gate é executado, **Then** o fluxo falha fechado e não altera produção.

---

### User Story 3 - Auditar histórico de ciclos e rollback (Priority: P1)

Como avaliador ou integrante do time, quero consultar evidências do ciclo em MLflow e em arquivos JSON versionáveis, para entender quem autorizou, quais métricas foram obtidas, quais artefatos foram avaliados e como reverter para uma versão anterior aprovada.

**Why this priority**: O professor pediu atualização contínua; evidência auditável mostra maturidade e evita uma resposta apenas conceitual.

**Independent Test**: Após um ciclo, abrir o registro JSON e o histórico visual do experimento para confirmar presença de autorização, métricas, decisão, checksums e motivo de promoção/rejeição.

**Acceptance Scenarios**:

1. **Given** um ciclo executado, **When** o relatório é consultado, **Then** ele contém data do ciclo, corte, autorização, métricas, checks críticos, decisão e checksums.
2. **Given** um ciclo executado, **When** o histórico visual é consultado, **Then** ele apresenta as mesmas tags e métricas principais do ciclo.
3. **Given** necessidade de rollback, **When** o time consulta o changelog de modelos, **Then** encontra a versão anterior aprovada com URL/checksum e procedimento de reversão.

---

### User Story 4 - Explicar quem consome as predições (Priority: P2)

Como time do produto, quero deixar claro quem consome as predições e quais informações são expostas, para demonstrar menor privilégio e utilidade operacional.

**Why this priority**: O avaliador perguntou quem é o cliente final da aplicação; a resposta precisa separar consumidor técnico e consumidor de negócio.

**Independent Test**: Revisar a documentação e confirmar que a UI não consome diretamente a API de ML e que o consultor recebe apenas risco traduzido em ação.

**Acceptance Scenarios**:

1. **Given** a documentação do produto, **When** o fluxo de consumo é lido, **Then** o Java BFF aparece como único cliente técnico autorizado da API de ML.
2. **Given** a visão do consultor, **When** uma predição é apresentada, **Then** aparecem nível de risco, perfil previsto, ação recomendada e prioridade, sem probabilidade crua.

---

### Edge Cases

- Corte candidato muito recente para ter janela futura de 18 meses completa.
- Candidato gerado, mas holdout temporal fixo ausente.
- Candidato melhora AUC-ROC, mas não melhora AUC-PR acima da margem mínima.
- Anti-leakage falha por coluna futura ou target no conjunto de features.
- Solicitação de promoção sem autorização explícita.
- Checksum ausente ou incompatível para artefato candidato ou champion.
- Registro visual indisponível; JSON deve continuar servindo como evidência mínima.
- Rollback solicitado para artefato sem registro prévio aprovado.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: O sistema deve documentar a política de “retreino programado trimestral com monitoramento contínuo de drift”, evitando o termo “retreino contínuo” isolado.
- **FR-002**: O sistema deve validar que um corte candidato só é elegível quando a janela futura de 18 meses do rótulo está completamente observada nos dados disponíveis.
- **FR-003**: O sistema deve calcular features de entrada usando apenas eventos até o corte selecionado.
- **FR-004**: O sistema deve calcular o target de churn usando apenas eventos posteriores ao corte dentro da janela de 18 meses.
- **FR-005**: O sistema deve rejeitar ciclos sem autorização explícita contendo responsável, identificador/ticket, horário e escopo autorizado.
- **FR-006**: O fluxo de retreino deve gerar artefatos candidatos isolados sem alterar a configuração ou os artefatos de produção por padrão.
- **FR-007**: O gate de promoção deve comparar champion e challenger no mesmo holdout temporal fixo e versionado.
- **FR-008**: O gate de promoção deve manter o champion em caso de empate técnico ou ganho insuficiente do challenger.
- **FR-009**: O gate de promoção deve exigir ganho mínimo de AUC-PR do challenger sobre o champion para aprovar promoção técnica.
- **FR-010**: O sistema deve continuar reportando AUC-ROC como métrica de compatibilidade, mas deve usar AUC-PR como métrica decisória do gate.
- **FR-011**: O gate deve executar verificações anti-leakage antes de aprovar qualquer candidato.
- **FR-012**: O gate deve falhar fechado se faltar maturação temporal, holdout, autorização, checksum válido ou aprovação anti-leakage.
- **FR-013**: O gate deve operar em modo relatório por padrão, sem promover artefatos automaticamente.
- **FR-014**: A promoção efetiva de artefatos deve exigir uma autorização explícita adicional no comando ou procedimento operacional.
- **FR-015**: Cada ciclo deve gerar evidência persistente em JSON contendo autorização, corte, métricas, checks, checksums, decisão e motivo.
- **FR-016**: Cada ciclo deve registrar tags, métricas e artefato de evidência no histórico visual de experimentos.
- **FR-017**: O sistema deve manter procedimento de rollback baseado apenas em versões anteriores aprovadas e verificáveis por checksum.
- **FR-018**: O sistema deve monitorar drift leve em features comportamentais críticas e classificar o resultado como OK, observar ou alerta.
- **FR-019**: A documentação deve identificar o Java BFF como único consumidor técnico autorizado da API de ML.
- **FR-020**: A documentação deve identificar o consultor da concessionária como consumidor de negócio, recebendo perfil, nível de risco, ação recomendada e prioridade, sem probabilidade crua.
- **FR-021**: O sistema deve preservar rastreabilidade suficiente para demonstração acadêmica sem expor dados sensíveis desnecessários.

### Key Entities

- **Ciclo de Retreino**: execução autorizada para avaliar um novo corte e gerar candidato.
- **Data de Corte**: marco temporal que separa histórico permitido para features e janela futura usada para rótulo.
- **Modelo Champion**: modelo atualmente aprovado para uso.
- **Modelo Challenger**: modelo candidato treinado em diretório isolado.
- **Holdout Temporal de Gate**: amostra fixa e versionada usada para comparação justa entre champion e challenger.
- **Evidência de Ciclo**: registro JSON e visual contendo autorização, métricas, checks, checksums e decisão.
- **Registro de Rollback**: entrada aprovada contendo artefatos anteriores e checksums para reversão segura.
- **Consumidor Técnico**: Java BFF autorizado a chamar o serviço de ML.
- **Consumidor de Negócio**: consultor de concessionária que recebe recomendação operacional.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% dos ciclos sem autorização explícita são rejeitados antes de qualquer promoção de artefato.
- **SC-002**: 100% dos cortes candidatos sem janela futura completa de 18 meses são rejeitados antes do treino ou promoção.
- **SC-003**: 100% das decisões do gate produzem um registro de evidência JSON com autorização, métricas, checks e decisão quando os insumos mínimos estão disponíveis.
- **SC-004**: 100% das promoções aprovadas registram checksums antes e depois da troca de artefatos.
- **SC-005**: O challenger só é aprovado quando supera o champion em AUC-PR pela margem mínima definida; empates técnicos preservam o champion.
- **SC-006**: A demonstração da banca consegue mostrar, em até 10 minutos, política, candidato isolado, gate, evidência JSON, histórico visual, drift e rollback documentado.
- **SC-007**: A documentação permite identificar em menos de 2 minutos quem consome tecnicamente a predição e quem a consome como negócio.
- **SC-008**: Nenhum requisito de consumo exige que a UI ou o consultor acesse diretamente a API de ML ou probabilidade crua.

## Assumptions

- O projeto permanece como PoC acadêmica demonstrável, não como MLOps corporativo completo.
- O dataset histórico real contém datas suficientes para demonstrar ao menos um corte maduro.
- O histórico visual de experimentos já existe ou pode ser reaproveitado para registrar novos ciclos.
- Evidências JSON pequenas podem ser versionadas quando não contiverem dados sensíveis.
- Artefatos grandes de modelo e dataset bruto não precisam ser versionados no repositório.
- A margem inicial de ganho de AUC-PR é 0.005, podendo ser ajustada por decisão documentada do time.
