# ADR-002 — RandomForest em vez de Deep Learning para a classificação

**Status:** Aceito

## Contexto

O near-RT RIC exige inferência de baixa latência (na ordem de sub-segundo)
e operação simples. O volume de features por amostra é pequeno e
tabular.

## Decisão

**RandomForest** (`services/ml-service/app/model.py`): inferência rápida em
CPU, sem GPU, interpretável (feature importance), robusto a features não
escaladas, baixo custo operacional.

## Alternativas

- **LSTM / Transformer temporal** — capturaria padrões temporais mais
  sofisticados, mas adiciona GPU, tuning, latência e complexidade de deploy
  desproporcionais ao ganho neste escopo.
- **Regras/threshold** — insuficiente para combinações de métricas.

## Consequências

- (+) Latência e operação compatíveis com near-RT; explicável.
- (−) Não modela dependência temporal longa — documentado como evolução
  futura se o padrão de anomalia exigir.
