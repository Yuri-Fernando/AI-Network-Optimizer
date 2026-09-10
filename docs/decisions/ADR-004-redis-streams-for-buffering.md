# ADR-004 — Redis Streams para buffer entre ingestão e ML (não Kafka, neste escopo)

**Status:** Aceito

## Contexto

O V2 adicionou um buffer de streaming entre o gRPC Ingestion Service e o ML
Service, para desacoplar as taxas de produção e consumo e permitir replay
curto.

## Decisão

**Redis Streams** como buffer: consumer groups, ack, `XAUTOCLAIM` para
mensagens presas, latência baixíssima, e o Redis já está no stack (cache).
Para o volume-alvo de um xApp de demonstração, isso é suficiente.

## Alternativas

- **Kafka** — a escolha certa para um event backbone corporativo multi-time
  com retenção longa (é o que o **Argus** e o **RetentIQ** usam). Aqui
  seria peso operacional desproporcional ao escopo de um único xApp.

## Consequências

- (+) Streaming com garantias suficientes e zero infra nova.
- (−) Sem retenção longa / múltiplos domínios; se o projeto crescer para
  vários xApps, migrar para Kafka (caminho documentado).
