# AI Network Optimizer

### O-RAN · 5G · Network Intelligence · Machine Learning · gRPC · Kubernetes

## Status

🟢 **Concluído — Projeto de portfólio / Telecom AI**

Sistema de detecção de anomalias em métricas de rede inspirado na arquitetura **xApp / near-RT RIC do ecossistema O-RAN (Open RAN)**.

O projeto implementa um fluxo distribuído para ingestão contínua de métricas, classificação do estado da rede com Machine Learning e exposição das decisões por API REST.

A arquitetura foi desenvolvida para demonstrar a integração entre **IA, sistemas distribuídos, telecomunicações, streaming de dados e infraestrutura cloud-native**.

Posicionamento no portfólio: **Cloud-native Telecom ML** — três
microsserviços (gRPC ingestion · ML service · API gateway), streaming,
Kubernetes com autoscaling. Trade-offs formalizados em
[`docs/decisions/`](docs/decisions/).

### Status das capacidades

| Capacidade | Status |
|---|---|
| Microsserviços (gRPC ingestion · ML service · API gateway) | ✅ `services/` |
| Contrato gRPC (Protocol Buffers) | ✅ `services/grpc-ingestion/app/proto/metrics.proto` |
| Modelo de detecção de anomalia (RandomForest) | ✅ `services/ml-service/` |
| API REST de consumo | ✅ `services/api-gateway/` |
| Docker Compose (dev) | ✅ `infra/docker-compose.yml` |
| Kubernetes + HPA (autoscaling) | ✅ `infra/k8s/` |
| Streaming buffer (Redis Streams) + métricas Prometheus + logs estruturados | ✅ V2 (`notebooks/..._v2.ipynb`) |
| ADRs de arquitetura | ✅ `docs/decisions/ADR-001..004` |
| Deploy real em cluster | 🗺️ manifests prontos; `kubectl apply` = próximo passo |

---

## Sobre o Projeto

Redes modernas de telecomunicações geram continuamente métricas como:

- Latência;
- Throughput;
- Packet Loss;
- Jitter.

O desafio abordado pelo projeto é detectar degradações da rede e transformar essas métricas em decisões estruturadas em tempo quase real.

O fluxo implementado é:

```text
[Simulador gNB]
      │
      │ gRPC streaming
      ▼
[gRPC Ingestion Service]
      │
      │ métricas contínuas
      ▼
[ML Service — RandomForest]
      │
      │ REST API
      ▼
[API Gateway — FastAPI]
```

---

# Objetivo

O projeto busca demonstrar uma arquitetura inspirada no funcionamento de um **RIC (RAN Intelligent Controller)**, conectando:

- Ingestão contínua de métricas;
- Processamento distribuído;
- Inferência de Machine Learning;
- Detecção de anomalias;
- Exposição de decisões via API;
- Containerização;
- Orquestração com Kubernetes;
- Autoscaling.

---

# Arquitetura

```text
                    ┌──────────────────────┐
                    │      Simulador       │
                    │        gNB           │
                    └──────────┬───────────┘
                               │
                               │ gRPC Streaming
                               ▼
                    ┌──────────────────────┐
                    │  gRPC Ingestion      │
                    │      Service         │
                    └──────────┬───────────┘
                               │
                               │ Network Metrics
                               ▼
                    ┌──────────────────────┐
                    │     ML Service       │
                    │    RandomForest      │
                    │  Real-Time Inference │
                    └──────────┬───────────┘
                               │
                               │ REST
                               ▼
                    ┌──────────────────────┐
                    │    API Gateway       │
                    │       FastAPI        │
                    └──────────┬───────────┘
                               │
                               ▼
                    External Consumers
```

---

# Relação com O-RAN

| Componente do Projeto | Equivalente O-RAN | Função |
|---|---|---|
| `grpc-ingestion` | E2 node / gNB | Simula coleta e envio de métricas |
| `ml-service` | xApp / near-RT RIC | Classificação e decisão baseada em IA |
| `api-gateway` | A1 / O1 | Exposição das decisões para sistemas externos |
| Kubernetes HPA | Elasticidade | Escalonamento automático do serviço |

> A arquitetura é uma **simulação inspirada no ecossistema O-RAN**, não uma implementação completa de uma infraestrutura O-RAN real.

---

# Classificação de Anomalias

O modelo trabalha com três estados de rede:

| Estado | Latência | Throughput | Packet Loss | Ação Recomendada |
|---|---:|---:|---:|---|
| **NORMAL** | 5–30 ms | 80–150 Mbps | 0–1% | NENHUMA |
| **CONGESTIONADO** | 80–250 ms | 5–40 Mbps | 5–20% | BALANCEAR_CARGA |
| **DEGRADADO** | 200–500 ms | 1–15 Mbps | 15–40% | REROUTING_EMERGENCIAL |

### Features utilizadas

- Latência;
- Throughput;
- Packet Loss;
- Jitter.

---

# Stack Tecnológica

| Tecnologia | Papel | Decisão Técnica |
|---|---|---|
| **gRPC + Protocol Buffers** | Ingestão | Streaming contínuo e tipagem forte |
| **Python / FastAPI** | API Gateway | API REST assíncrona e OpenAPI |
| **scikit-learn / RandomForest** | ML | Inferência rápida e interpretável |
| **Docker** | Containerização | Isolamento e reprodutibilidade |
| **Docker Compose** | Ambiente local | Execução integrada dos serviços |
| **Kubernetes** | Orquestração | Deploy e gerenciamento dos serviços |
| **HPA** | Autoscaling | Escalonamento automático do ML Service |

---

# Pipeline de Processamento

```text
Network Metrics
      ↓
gNB Simulator
      ↓
gRPC Streaming
      ↓
Ingestion Service
      ↓
Metric Pipeline
      ↓
Feature Extraction
      ↓
RandomForest
      ↓
Network State
      ↓
Recommended Action
      ↓
FastAPI Gateway
      ↓
External Consumer
```

---

# Funcionalidades

- Simulação de métricas de rede;
- Streaming contínuo via gRPC;
- Processamento desacoplado;
- Classificação de estados da rede;
- Inferência em tempo quase real;
- API REST;
- Monitoramento do estado dos nós;
- Histórico de inferências;
- Detecção de anomalias;
- Recomendação de ações;
- Containerização dos serviços;
- Deploy local via Docker Compose;
- Deploy em Kubernetes;
- Autoscaling com HPA.

---

# Endpoints REST

| Método | Endpoint | Descrição |
|---|---|---|
| `GET` | `/` | Informações do serviço |
| `GET` | `/health` | Health check |
| `GET` | `/network/status` | Estado atual dos nós |
| `GET` | `/network/alerts` | Apenas anomalias detectadas |
| `GET` | `/network/history?limit=N` | Histórico de inferências |
| `GET` | `/network/node/{id}` | Estado de um nó específico |

### Exemplo

```json
{
  "overall_status": "CONGESTIONADO",
  "total_nodes": 3,
  "nodes": [
    {
      "node_id": "gNB-001",
      "status": "CONGESTIONADO",
      "confidence": 0.97,
      "action": "BALANCEAR_CARGA",
      "raw_metrics": {
        "latency": 142.5,
        "throughput": 22.3,
        "packet_loss": 11.2,
        "jitter": 38.7
      }
    }
  ]
}
```

---

# Quick Start

## Opção 1 — Notebook

Instale as dependências:

```bash
pip install jupyter scikit-learn numpy pandas matplotlib seaborn fastapi uvicorn httpx grpcio grpcio-tools
```

Execute:

```bash
jupyter notebook notebooks/ai_network_optimizer.ipynb
```

O notebook percorre o pipeline completo:

| Etapa | Função |
|---|---|
| 0 | Instalação/verificação das dependências |
| 1 | Simulador de métricas de rede |
| 2 | Dataset e visualização |
| 3 | Treinamento e avaliação |
| 4 | Pipeline de inferência |
| 5 | FastAPI e testes de endpoints |
| 6 | Benchmark de latência e ROC-AUC |
| 7 | Compilação `.proto` e streaming gRPC |
| 8 | Docker e Kubernetes |
| 9 | Resumo técnico para entrevista |

---

## Opção 2 — Docker Compose

```bash
cd infra/
docker-compose up --build
```

Serviços:

| Serviço | Acesso |
|---|---|
| API REST | `http://localhost:8000` |
| Swagger | `http://localhost:8000/docs` |
| gRPC | `localhost:50051` |

---

## Opção 3 — Kubernetes

Build das imagens:

```bash
docker build -t ai-network-optimizer/grpc-ingestion:latest services/grpc-ingestion/
docker build -t ai-network-optimizer/ml-service:latest services/ml-service/
docker build -t ai-network-optimizer/api-gateway:latest services/api-gateway/
```

Deploy:

```bash
kubectl apply -f infra/k8s/namespace.yaml
kubectl apply -f infra/k8s/
```

Verificar:

```bash
kubectl get pods -n ai-network
kubectl get hpa -n ai-network
```

Acessar a API:

```bash
kubectl port-forward svc/api-gateway-service 8000:80 -n ai-network
```

> O deployment em Kubernetes faz parte da arquitetura demonstrativa e experimental do projeto.

---

# Estrutura do Projeto

```text
ai-network-optimizer/
│
├── services/
│   ├── grpc-ingestion/
│   │   ├── app/
│   │   │   ├── server.py
│   │   │   ├── simulator.py
│   │   │   └── proto/
│   │   │       └── metrics.proto
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── ml-service/
│   │   ├── app/
│   │   │   ├── model.py
│   │   │   ├── trainer.py
│   │   │   └── inference.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   └── api-gateway/
│       ├── app/
│       │   ├── main.py
│       │   └── routes.py
│       ├── Dockerfile
│       └── requirements.txt
│
├── infra/
│   ├── docker-compose.yml
│   └── k8s/
│       ├── namespace.yaml
│       ├── grpc-deployment.yaml
│       ├── ml-deployment.yaml
│       ├── api-deployment.yaml
│       ├── services.yaml
│       └── hpa.yaml
│
├── notebooks/
│   └── ai_network_optimizer.ipynb
│
├── shared/
│   └── schemas/
│       └── metric_schema.py
│
└── README.md
```

---

# Serviços

## gRPC Ingestion Service

Responsável por:

- Simular nós gNB;
- Gerar métricas;
- Fazer streaming via gRPC;
- Disponibilizar dados ao ML Service.

## ML Service

Responsável por:

- Treinar o RandomForest;
- Extrair features;
- Executar inferência;
- Classificar o estado da rede;
- Recomendar ações;
- Processar o fluxo continuamente.

## API Gateway

Responsável por:

- Expor a API REST;
- Disponibilizar status da rede;
- Expor alertas;
- Consultar histórico;
- Consultar nós individualmente.

---

# Trade-offs Técnicos

> Formalizados como ADRs em [`docs/decisions/`](docs/decisions/):
> ADR-001 (gRPC vs REST), ADR-002 (RandomForest vs Deep Learning),
> ADR-003 (Compose vs Kubernetes), ADR-004 (Redis Streams vs Kafka neste
> escopo). Resumo abaixo.

## gRPC vs REST

**gRPC** foi escolhido para a camada de ingestão por sua adequação a streaming contínuo, contratos definidos por Protocol Buffers e comunicação eficiente entre serviços.

**REST** foi utilizado na camada de exposição porque oferece maior universalidade e facilidade de integração com consumidores externos.

```text
gRPC → ingestão / streaming
REST → consumo / integração
```

---

## RandomForest vs Deep Learning

O **RandomForest** foi utilizado por oferecer:

- Inferência rápida;
- Baixa complexidade operacional;
- Interpretabilidade;
- Execução sem GPU.

Modelos temporais mais complexos, como LSTM, poderiam ser utilizados em cenários com padrões temporais mais sofisticados, mas adicionariam complexidade ao pipeline.

---

## Docker Compose vs Kubernetes

**Docker Compose** é utilizado para desenvolvimento local e demonstração.

**Kubernetes** adiciona:

- Gerenciamento dos serviços;
- Replicação;
- Autoscaling;
- Orquestração;
- Ambiente multi-node.

---

# Contexto O-RAN

O **O-RAN (Open Radio Access Network)** busca ampliar a abertura e interoperabilidade dos componentes das redes móveis.

Dentro desse contexto:

- **xApp** → aplicação executada no near-RT RIC;
- **rApp** → aplicação associada ao non-RT RIC;
- **E2** → comunicação entre RIC e elementos da RAN;
- **A1/O1** → interfaces utilizadas na integração com componentes externos.

Este projeto simula conceitualmente o fluxo:

```text
E2
 ↓
near-RT RIC / xApp
 ↓
A1 / O1
```

---

# O que este projeto demonstra

- Machine Learning aplicado a telecom;
- Análise de métricas de redes;
- Detecção de anomalias;
- Sistemas distribuídos;
- Streaming de dados;
- gRPC;
- Protocol Buffers;
- APIs REST;
- FastAPI;
- Docker;
- Docker Compose;
- Kubernetes;
- Horizontal Pod Autoscaler;
- Arquitetura de microsserviços;
- Inferência em tempo quase real;
- Integração entre IA e infraestrutura de telecom;
- Conceitos de O-RAN e RIC.

---

# Limitações

- As métricas são geradas por um simulador;
- O ambiente não utiliza uma RAN 4G/5G física;
- O projeto não implementa uma pilha O-RAN real;
- As interfaces E2/A1/O1 são representadas conceitualmente;
- O modelo utiliza dados simulados e não telemetria operacional de uma operadora;
- Os resultados não representam desempenho de uma rede comercial;
- O RandomForest não incorpora modelagem temporal profunda;
- O HPA é utilizado como mecanismo demonstrativo de autoscaling.

---

# Melhorias Futuras

- Integração com datasets reais de redes móveis;
- Modelos temporais para detecção de degradação;
- LSTM ou Transformers para séries temporais;
- Reinforcement Learning para otimização de ações;
- Prometheus para métricas;
- Grafana para observabilidade;
- Redis Streams;
- Persistent storage;
- Integração com simuladores O-RAN;
- xApp mais próximo da arquitetura real;
- Policy Engine;
- Alertas automáticos;
- Testes de carga;
- Distributed tracing;
- CI/CD;
- Deployment em ambiente cloud.

---

# Status Final

🟢 **Concluído**

O projeto possui a arquitetura principal implementada, incluindo:

- ✅ Simulador de métricas de rede;
- ✅ gRPC streaming;
- ✅ Serviço de ingestão;
- ✅ Machine Learning com RandomForest;
- ✅ Inferência contínua;
- ✅ API FastAPI;
- ✅ Endpoints de monitoramento;
- ✅ Histórico de inferências;
- ✅ Detecção de anomalias;
- ✅ Recomendação de ações;
- ✅ Docker;
- ✅ Docker Compose;
- ✅ Kubernetes;
- ✅ HPA;
- ✅ Notebook completo;
- ✅ Documentação arquitetural;
- ✅ Relação conceitual com O-RAN.

O projeto permanece disponível como demonstração técnica da integração entre **IA, telecomunicações, sistemas distribuídos e infraestrutura cloud-native**.

---

# Autor

**Yuri Fernando Dubbern**

AI/ML Engineer · Machine Learning · Data Engineering · Telecom AI · Distributed Systems

[LinkedIn](https://www.linkedin.com/in/yuridubbern) · [GitHub](https://github.com/Yuri-Fernando) · [Lattes](http://lattes.cnpq.br/7151392692642166) · [Linktree](https://linktr.ee/yuri.f.dubbern)
