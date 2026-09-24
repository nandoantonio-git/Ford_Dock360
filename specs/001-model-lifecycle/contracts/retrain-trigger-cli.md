# Contract: `scripts/retrain_trigger.sh`

## Purpose

Gerar artefatos candidatos isolados para um novo corte, validando maturação temporal, cadência mínima e autorização explícita antes de iniciar o pipeline.

## Invocation

```bash
./scripts/retrain_trigger.sh <NOVO_DATA_CORTE> \
  --authorized-by "Fernando RM555201" \
  --authorization-ticket "Sprint3-demo" \
  --scope "Gerar candidato isolado para gate acadêmico"
```

## Required positional argument

- `NOVO_DATA_CORTE`: data em formato `YYYY-MM-DD`.

## Required options

- `--authorized-by`: nome/RM ou identificador de quem autorizou.
- `--authorization-ticket`: identificador rastreável do pedido, aula, sprint ou demo.
- `--scope`: descrição curta do escopo autorizado.

## Optional behavior

- Pode aceitar variáveis internas para cadência mínima de 3 meses e janela de maturação de 18 meses.
- Pode aceitar caminhos alternativos futuramente, desde que defaults preservem o projeto atual.

## Preconditions

- Dataset bruto existe em `data/raw/vin_share_Desafio_02.xlsx`.
- `src/pipeline/config.py` contém o corte atual de referência.
- Scripts Python parametrizados estão disponíveis.

## Side effects allowed

- Criar ou atualizar `data/processed/candidate/`.
- Criar ou atualizar `models/candidate/`.
- Imprimir resumo da autorização e validações.

## Side effects forbidden

- Alterar `src/pipeline/config.py`.
- Sobrescrever modelos champion em `models/` fora de `models/candidate/`.
- Promover artefatos para produção.
- Criar evidência afirmando promoção.

## Failure behavior

Falhar com exit code diferente de zero quando:

- `NOVO_DATA_CORTE` estiver ausente ou inválido.
- autorização estiver incompleta.
- cadência mínima não for respeitada.
- corte não tiver janela futura completa de 18 meses.
- qualquer etapa do pipeline candidato falhar.

## Success output

Mensagem deve indicar:

- corte atual;
- novo corte;
- janela de maturação;
- autorização;
- diretórios de saída;
- próximo comando sugerido para `promotion_gate.py`.
