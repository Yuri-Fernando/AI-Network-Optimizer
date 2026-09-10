# ADR-001 — gRPC na ingestão, REST no consumo

**Status:** Aceito

## Contexto

O sistema tem dois tipos de comunicação: ingestão contínua e de alta
frequência de métricas de rede (do simulador/coletores para o serviço de
ingestão) e consumo pontual das decisões (por consumidores externos,
dashboards, xApps).

## Decisão

- **gRPC** (`services/grpc-ingestion/app/proto/metrics.proto`) para a
  ingestão: streaming bidirecional, contrato forte em Protocol Buffers,
  HTTP/2 multiplexado, serialização binária compacta.
- **REST/JSON** (`services/api-gateway`) para o consumo: universal,
  cacheável, inspecionável, fácil de integrar por qualquer cliente.

## Alternativas

- **REST em tudo** — overhead de JSON e ausência de streaming nativo pesam
  no caminho de ingestão de alta frequência.
- **gRPC em tudo** — piora a integração externa (sem cache HTTP, tooling de
  browser limitado).

## Consequências

- (+) Cada fronteira usa o protocolo adequado; `.proto` versiona o contrato
  de ingestão.
- (−) Dois estilos de contrato para manter (proto + OpenAPI).
