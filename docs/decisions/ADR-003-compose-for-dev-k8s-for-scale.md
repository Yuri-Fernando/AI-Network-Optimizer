# ADR-003 — Docker Compose no dev, Kubernetes para escala

**Status:** Aceito

## Decisão

- **Docker Compose** (`infra/docker-compose.yml`) para desenvolvimento local
  e demonstração — sobe os três serviços + dependências num comando.
- **Kubernetes** (`infra/k8s/`) para o ambiente de escala: `Deployment` por
  serviço, `Service`, **HPA** (`hpa.yaml`) para autoscaling do ml-service e
  do api-gateway sob carga, namespace dedicado.

## Consequências

- (+) Onboarding trivial (compose) sem abrir mão de um caminho de produção
  real (k8s + autoscaling).
- (−) Dois conjuntos de manifests para manter em paridade.
