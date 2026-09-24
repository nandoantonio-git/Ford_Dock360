# Constituição do Projeto — Ford VinGuard / Dock 360

## Escopo acadêmico

Este projeto é uma entrega acadêmica do Desafio 2 — VIN Share / Radar de Evasão. A solução deve ser tecnicamente defensável, demonstrável para banca e compatível com o prazo das Sprints 3 e 4, sem prometer um MLOps corporativo completo.

## Princípios de ML pós-venda

1. O problema é pós-venda: prever risco de abandono da rede autorizada Ford a partir do histórico parcial de serviços por VIN.
2. Features de treino só podem usar eventos até a data de corte.
3. Targets de churn só podem usar eventos posteriores à data de corte dentro da janela definida.
4. A maturação do rótulo de churn de 18 meses é uma restrição central do projeto.
5. Todo pré-processamento de modelo deve permanecer dentro de `sklearn.Pipeline`.
6. Classificadores devem usar `random_state=42` quando aplicável e `class_weight='balanced'` quando aplicável.
7. AUC-ROC deve continuar reportada para compatibilidade do projeto; AUC-PR pode ser usada para decisões de priorização/promoção.

## Princípios de desenvolvimento seguro

1. Menor privilégio: scripts de retreino não devem modificar artefatos de produção por padrão.
2. Autorização explícita: qualquer promoção ou ciclo de retreino deve registrar quem autorizou, quando e qual escopo foi autorizado.
3. Automação de verificações de segurança: leakage temporal, maturação de rótulo, checksums, holdout e contrato de consumo devem ser verificados automaticamente quando possível.
4. Rastreabilidade e evidências: decisões relevantes devem gerar evidência persistente em formato simples e auditável.
5. Falha segura: se uma verificação crítica falhar, o fluxo deve parar sem tocar produção.

## Consumidores

O Java BFF é o consumidor técnico autorizado da API de ML. A UI e o consultor de concessionária não chamam o serviço de ML diretamente. O consultor deve receber risco traduzido em nível, perfil e ação recomendada, não probabilidade crua.
