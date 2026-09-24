# Changelog de Modelos — Ford VinGuard

Este arquivo registra versões aprovadas de modelos e segmentadores para promoção ou rollback seguro.

Rollback válido deve sempre apontar para uma versão anterior aprovada, com URL/caminho e checksum SHA256. Backup local informal não é considerado evidência suficiente.

## Registro de versões

| Data | Versão | Tipo | Data corte | Artefato churn | SHA256 churn | Artefato perfil/KMeans | SHA256 perfil | Aprovado por | Ticket/autorização | Observações |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-23 | baseline-local | champion atual/documentado | 2024-10-31 | `models/churn_pos_venda_rf_calibrated.joblib` | A preencher com artefato aprovado | `models/kmeans_segmentador_pos_venda.joblib` | A preencher com artefato aprovado | Fernando RM555201 | registro-inicial | Entrada inicial para documentar o procedimento; preencher checksums reais antes de usar para rollback. |

## Procedimento de promoção

1. Gerar candidato isolado em `models/candidate/`.
2. Executar gate em modo relatório.
3. Confirmar que todos os checks críticos passaram.
4. Confirmar que o candidato superou o champion em AUC-PR pela margem mínima definida.
5. Executar promoção somente com autorização explícita e flag de promoção.
6. Registrar nesta tabela os novos artefatos e checksums.
7. Guardar o JSON de evidência em `reports/model_lifecycle/` e o run correspondente no MLflow quando disponível.

## Procedimento de rollback seguro

1. Identificar nesta tabela o último champion aprovado.
2. Conferir URL/caminho do artefato e SHA256 registrado.
3. Atualizar variáveis do ambiente de deploy conforme aplicável:
   - `CHURN_MODEL_URL`
   - `CHURN_MODEL_SHA256`
   - `PERFIL_MODEL_URL`
   - `PERFIL_MODEL_SHA256`
4. Reiniciar o serviço.
5. Validar `/health`.
6. Registrar a evidência do rollback em novo JSON de ciclo ou em observação neste changelog.

## Regras

- Não usar modelo sem checksum.
- Não usar URL assinada, token ou segredo em arquivo versionado.
- Não commitar artefatos grandes de modelo no repositório.
- Não considerar rollback válido se o artefato não estiver registrado neste changelog.
- Empate técnico entre challenger e champion mantém o champion.
