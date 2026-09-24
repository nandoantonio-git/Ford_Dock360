# Política de Retreino Programado — Ford VinGuard

## Objetivo

Esta política define como o Ford VinGuard deve atualizar o modelo de churn e o segmentador comportamental em escopo acadêmico demonstrável, sem prometer uma plataforma completa de MLOps de produção.

O termo correto para o projeto é **retreino programado trimestral com monitoramento contínuo de drift**. O monitoramento pode ser recorrente, mas o retreino não deve ser tratado como online ou contínuo, porque o rótulo de churn tem latência de 18 meses.

## Regra temporal obrigatória

O churn é definido como ausência de serviço dentro da janela futura de 18 meses após a data de corte.

Um novo `DATA_CORTE` só é elegível quando:

```text
DATA_CORTE + 18 meses <= maior ServiceDate disponível no dataset
```

As features do modelo devem usar apenas eventos com:

```text
ServiceDate <= DATA_CORTE
```

O target deve usar apenas eventos com:

```text
DATA_CORTE < ServiceDate <= DATA_CORTE + 18 meses
```

Não se deve filtrar VINs pela regra `hoje - ultimo_evento >= 18 meses`, porque isso seleciona VINs que já aparentam abandono e cria viés de seleção.

## Cadência

- Retreino ordinário: trimestral.
- Retreino antecipado: permitido apenas se o monitoramento de drift indicar alerta relevante e houver autorização explícita.
- Retreino semanal/automático sem maturação do rótulo: proibido.

## Desenvolvimento seguro

Cada ciclo deve obedecer aos princípios abaixo:

1. **Menor privilégio**: o gatilho de retreino gera apenas artefatos candidatos em diretórios isolados.
2. **Autorização explícita**: todo ciclo registra `authorized_by`, `authorization_ticket`, data/hora e escopo autorizado.
3. **Automação de verificações de segurança**: maturação temporal, anti-leakage, holdout, checksums e autorização são gates obrigatórios.
4. **Rastreabilidade e evidências**: cada decisão gera evidência em JSON e registro no MLflow quando disponível.
5. **Falha segura**: se qualquer verificação crítica falhar, o fluxo para sem tocar artefatos de produção.

## Fluxo esperado

```text
novo DATA_CORTE candidato
  -> valida autorização e cadência
  -> valida maturação da janela futura de 18 meses
  -> gera datasets e modelos candidatos isolados
  -> compara champion/challenger em holdout temporal fixo
  -> gera evidência JSON + MLflow
  -> promove somente com aprovação técnica e flag explícito de promoção
```

## Gate champion/challenger

O gate de promoção compara o modelo candidato contra o champion no mesmo holdout temporal fixo e versionado.

Critérios mínimos:

- checks anti-leakage aprovados;
- holdout temporal disponível;
- checksums calculáveis;
- autorização explícita presente;
- AUC-PR do candidato maior que a do champion por margem mínima inicial de `0.005`.

AUC-ROC continua sendo reportada para compatibilidade do projeto, mas AUC-PR é a métrica decisória do gate porque a aplicação prioriza VINs em risco.

Empate técnico mantém o champion.

## Evidências obrigatórias

Cada ciclo deve gerar, quando os insumos mínimos estiverem disponíveis:

- JSON em `reports/model_lifecycle/` com autorização, corte, checks, métricas, checksums, decisão e motivo;
- run/tags/métricas no MLflow quando disponível;
- changelog de modelos atualizado quando houver promoção ou rollback;
- relatório de drift quando houver base de referência e base atual.

A evidência não deve incluir linhas brutas do dataset nem segredos.

## Rollback

Rollback só pode usar versão anterior aprovada e verificável por checksum.

Procedimento:

1. consultar `CHANGELOG_MODELOS.md`;
2. escolher o último champion aprovado;
3. conferir URL/caminho e SHA256;
4. repontar variáveis de ambiente `CHURN_MODEL_URL`, `CHURN_MODEL_SHA256`, `PERFIL_MODEL_URL` e `PERFIL_MODEL_SHA256` quando aplicável;
5. reiniciar o serviço;
6. validar `/health`;
7. registrar evidência do rollback.

Backups locais não registrados não são rollback válido.

## Consumidores das predições

O consumidor técnico autorizado da API de ML é o **Java BFF**, usando `X-ML-Service-Token` server-to-server.

A UI e o consultor da concessionária não devem chamar o serviço de ML diretamente.

O consumidor de negócio é o consultor da concessionária. Ele deve receber uma saída operacional, não probabilidade crua:

- `perfil_previsto`;
- `risk_level`;
- `acao_recomendada`;
- prioridade na fila de abordagem.

A probabilidade calibrada pode existir internamente para cálculo e auditoria, mas não deve ser a linguagem principal apresentada ao consultor.

## Escopo acadêmico

Esta política demonstra maturidade de ciclo de vida do modelo para a banca: atualização programada, segurança, rastreabilidade e rollback. Ela não substitui um MLOps corporativo completo com orquestração gerenciada, governança formal de modelos e monitoração produtiva 24x7.
