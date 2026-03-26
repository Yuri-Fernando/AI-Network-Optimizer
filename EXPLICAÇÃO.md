# AI Network Optimizer — Explicação Completa

> Teoria, arquitetura, conceitos de telecomunicações e como este projeto simula um xApp O-RAN em tempo real.

---

## Índice

1. [Redes Móveis: da 4G/LTE ao 5G](#1-redes-móveis-da-4glte-ao-5g)
2. [3GPP: O Órgão que Define os Padrões](#2-3gpp-o-órgão-que-define-os-padrões)
3. [Organizações que Lideram IA em Redes](#3-organizações-que-lideram-ia-em-redes)
4. [O-RAN: A Revolução do Rádio Aberto](#4-o-ran-a-revolução-do-rádio-aberto)
5. [RIC: O Cérebro Inteligente do O-RAN](#5-ric-o-cérebro-inteligente-do-o-ran)
6. [xApps e rApps: Aplicações de IA na RAN](#6-xapps-e-rapps-aplicações-de-ia-na-ran)
7. [Interfaces do O-RAN: E2, A1, O1](#7-interfaces-do-o-ran-e2-a1-o1)
8. [Como Este Projeto Mapeia o O-RAN](#8-como-este-projeto-mapeia-o-o-ran)
9. [O Loop de Inferência em Tempo Real](#9-o-loop-de-inferência-em-tempo-real)
10. [Tecnologias Usadas e Por Quê](#10-tecnologias-usadas-e-por-quê)
11. [Métricas de Rede Simuladas](#11-métricas-de-rede-simuladas)
12. [O Modelo de ML: RandomForest na RAN](#12-o-modelo-de-ml-randomforest-na-ran)
13. [Fluxo de Dados Completo](#13-fluxo-de-dados-completo)

---

## 1. Redes Móveis: da 4G/LTE ao 5G

### O que é uma Rede de Acesso por Rádio (RAN)?

Toda vez que seu celular se conecta à internet, ele se comunica com uma torre de celular — chamada de **estação base**. Essa estação base faz parte da **RAN (Radio Access Network)**, a camada responsável por conectar dispositivos móveis à rede central (core) do operador.

A RAN é composta por:

```
[Seu celular / UE]  <--rádio-->  [Torre/Antena]  <--fibra-->  [Core da Operadora]  <-->  [Internet]
     (User Equipment)              (gNB / eNB)                   (EPC / 5GC)
```

- **UE (User Equipment):** Seu celular, tablet, IoT sensor
- **gNB (gNodeB):** Estação base do 5G (Node B evoluído)
- **eNB (eNodeB):** Estação base do 4G/LTE
- **EPC (Evolved Packet Core):** Core do 4G
- **5GC (5G Core):** Core do 5G, totalmente virtualizado

### 4G/LTE — A Base Atual

O **LTE (Long Term Evolution)**, comercializado como 4G, foi padronizado pelo 3GPP a partir de 2008 (Release 8). Características principais:

| Característica | 4G/LTE |
|---|---|
| Frequências | 700 MHz – 2.6 GHz |
| Velocidade máxima teórica | 150 Mbps (DL) / 50 Mbps (UL) |
| Latência | ~30–50 ms |
| Multiplexação | OFDMA (downlink), SC-FDMA (uplink) |
| Estação base | eNodeB (eNB) |
| Arquitetura RAN | RAN monolítica (hardware integrado) |

No 4G, a estação base (eNB) é uma **caixa única**: o hardware de rádio e o software de processamento de sinal ficam juntos, no mesmo equipamento, geralmente fornecido por um único vendor (Ericsson, Nokia, Huawei).

### 5G NR — A Nova Geração

O **5G NR (New Radio)**, padronizado a partir do 3GPP Release 15 (2018), trouxe mudanças revolucionárias:

| Característica | 5G NR |
|---|---|
| Frequências | Sub-6 GHz (FR1) e mmWave 24–100 GHz (FR2) |
| Velocidade máxima teórica | 20 Gbps (DL) |
| Latência | <1 ms (URLLC) |
| Multiplexação | OFDM flexível (múltiplos numerologies) |
| Estação base | gNodeB (gNB) |
| Arquitetura RAN | Desagregada: O-RU / O-DU / O-CU |

A inovação mais importante do 5G para este projeto é a **desagregação da RAN**:

```
4G (monolítica):
[eNB = Hardware + Software num único box]

5G (desagregada):
[O-RU]  <--fronthaul-->  [O-DU]  <--midhaul-->  [O-CU]
 (antena/rádio)           (processamento           (controle /
  hardware físico)         de baixo nível)           user plane)
```

Essa separação abriu a porta para o **Open RAN**: agora cada componente pode ser de um vendor diferente, rodando em hardware genérico (COTS — Commercial Off-The-Shelf).

### Casos de Uso do 5G Definidos pelo 3GPP

O 5G foi projetado para três grandes cenários:

```
          eMBB                    URLLC                   mMTC
  (Enhanced Mobile          (Ultra-Reliable           (Massive Machine
   Broadband)                Low-Latency Comm.)         Type Comm.)

  - Streaming 4K/8K         - Cirurgia remota          - 1 milhão de
  - Realidade virtual       - Veículos autônomos         dispositivos/km²
  - Hotspots densos         - Controle industrial       - Smart cities
                            - Latência < 1ms            - Sensores IoT
```

Este projeto foca no monitoramento de **eMBB e URLLC** — onde métricas como latência, throughput e packet loss são críticas.

---

## 2. 3GPP: O Órgão que Define os Padrões

O **3GPP (3rd Generation Partnership Project)** é o consórcio internacional responsável por criar as especificações técnicas de redes móveis, da 3G ao 5G (e já trabalhando no 6G).

### Como o 3GPP Funciona

O 3GPP não é uma empresa — é uma **aliança de organizações de padrões** (ETSI, ARIB, TSDSI, ATIS, TIRA, CCSA, TTA). Ele organiza seu trabalho em **Releases** numeradas:

| Release | Ano | Principais Inovações |
|---|---|---|
| Rel. 8 | 2009 | 4G LTE básico |
| Rel. 9 | 2010 | LTE-A enhancements |
| Rel. 10 | 2011 | LTE-Advanced (Carrier Aggregation) |
| Rel. 15 | 2018 | 5G NR Phase 1 (NSA + SA) |
| Rel. 16 | 2020 | 5G Phase 2 (URLLC, V2X) |
| Rel. 17 | 2022 | RedCap, NR-U, IoT NTN |
| Rel. 18 | 2024 | 5G-Advanced, IA/ML na RAN |
| Rel. 19 | 2025+ | AI-native RAN, 6G seeds |

### 3GPP e IA/ML: O Release 18 é um Marco

O 3GPP Release 18 trouxe estudos formais sobre **IA/ML nativo na RAN** (TR 38.843, TR 37.817):

- **IA para predição de beam management** (melhora handover)
- **IA para posicionamento** (localização indoor sub-metro)
- **IA para CSI (Channel State Information) feedback compression**
- **AI-native air interface** — o canal de rádio em si usa IA para codificação/decodificação

O Release 19 vai além com **AI-native RAN** onde as próprias camadas PHY/MAC da pilha de protocolo são substituídas por modelos de IA.

**Relevância para este projeto:** O ML Service deste projeto simula exatamente o que o 3GPP Rel. 18 preconiza — um agente de IA que consome métricas da RAN e toma decisões autônomas de otimização (a xApp na near-RT RIC).

---

## 3. Organizações que Lideram IA em Redes

### O-RAN Alliance

**O que é:** Consórcio fundado em 2018 por AT&T, China Mobile, Deutsche Telekom, NTT DOCOMO e Orange. Hoje tem mais de 300 membros.

**Missão:** Criar especificações abertas e interoperáveis para a RAN, quebrando o lock-in de vendors proprietários.

**Principais especificações relevantes para este projeto:**

| Especificação | Conteúdo | Mapeamento no Projeto |
|---|---|---|
| O-RAN.WG2.AIML-v01 | Framework de IA/ML no RIC | ML Service inteiro |
| O-RAN.WG3.E2AP | Interface E2 (RIC ↔ nó RAN) | gRPC Ingestion Service |
| O-RAN.WG3.RICARCH | Arquitetura do near-RT RIC | Estrutura de 3 serviços |
| O-RAN.WG1.OAD | Casos de uso do O-RAN (incluindo anomaly detection) | O problema que o projeto resolve |

**Os 7 Working Groups do O-RAN Alliance:**
- WG1: Use Cases & Architecture
- WG2: Non-RT RIC & A1 Interface
- WG3: Near-RT RIC & E2 Interface
- WG4: Open Fronthaul
- WG5: Open F1/W1/E1/X2/Xn Interfaces
- WG6: Cloudification & Orchestration
- WG7: White-box Hardware

### AI-RAN Alliance

**O que é:** Aliança lançada em 2024 por NVIDIA, Microsoft, Samsung, SoftBank e outros.

**Missão:** Acelerar a adoção de IA nativa em redes de acesso por rádio — não apenas IA para otimizar a rede, mas **IA rodando na infraestrutura da RAN** (compute sharing entre funções de rede e workloads de IA).

**Conceito central:** O mesmo hardware de GPU que processa sinais de rádio 5G também pode executar modelos de IA para outras aplicações, compartilhando recursos de forma dinâmica.

**Relevância para este projeto:** A AI-RAN Alliance é a visão futura do que este projeto demonstra em escala reduzida — IA integrada nativamente ao ciclo de operação da RAN.

### ETSI (European Telecommunications Standards Institute)

**O que é:** Órgão de padronização europeu que produz normas técnicas para telecomunicações.

**Iniciativas relevantes:**
- **ETSI ISG ENI (Experiential Networked Intelligence):** Define como sistemas cognitivos e IA gerenciam redes. O modelo cognitivo do ENI é um loop OBSERVE → ORIENT → DECIDE → ACT — exatamente o que o loop de inferência deste projeto implementa.
- **ETSI ISG MEC (Multi-access Edge Computing):** Computação na borda da rede, habilitando baixa latência para xApps.
- **ETSI NFV (Network Functions Virtualization):** Base para virtualizar funções de rede (como os containers Docker/Kubernetes deste projeto).

**Loop cognitivo ETSI ENI:**
```
OBSERVE → ORIENT → DECIDE → ACT
  ↑                              |
  └──────────────────────────────┘

Neste projeto:
OBSERVE  = gRPC Ingestion (coleta métricas dos gNBs)
ORIENT   = ML Service (feature extraction + normalização)
DECIDE   = RandomForest.predict() (classifica estado da rede)
ACT      = API Gateway expõe ação recomendada (BALANCEAR_CARGA, etc.)
```

### Broadband Forum (BBF)

**O que é:** Consórcio focado em redes de banda larga fixas e convergência fixo-móvel.

**Relevância:** O BBF trabalha em conjunto com o 3GPP e O-RAN Alliance na padronização de interfaces entre redes fixas (fibra, DSL) e RAN para cenários de acesso convergente. O TR-456 define como funções de rede podem ser orquestradas junto com a RAN. Para este projeto, o conceito de **orquestração de serviços via API REST** (O1 interface) vem dessa linha de padronização.

### ITU-T (International Telecommunication Union — Standardization Sector)

**Relevância:** O ITU-T Focus Group on Machine Learning for Future Networks (FG-ML5G) publicou os primeiros frameworks arquiteturais para ML em redes 5G, influenciando diretamente o design de xApps e o conceito de MLFO (ML Function Orchestrator) que inspira o ML Service deste projeto.

---

## 4. O-RAN: A Revolução do Rádio Aberto

### O Problema que o O-RAN Resolve

Antes do O-RAN, a RAN era dominada por poucos vendors (Ericsson, Nokia, Huawei) com soluções proprietárias — hardware e software vinculados. Um operador que comprava Ericsson precisava de todo o ecossistema Ericsson. Isso criava:

- **Lock-in de vendor:** Impossível misturar componentes de diferentes fabricantes
- **Alto custo:** Sem competição = preços altos
- **Inovação lenta:** Só o vendor pode desenvolver novas funcionalidades
- **Sem IA nativa:** Não há hooks para inserir modelos de ML no loop de controle

### A Solução O-RAN: Desagregação + Interfaces Abertas

O O-RAN define como separar a RAN em componentes interoperáveis conectados por interfaces padronizadas:

```
┌──────────────────────────────────────────────────────────────────┐
│                         O-RAN Architecture                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    Service Management &                      │  │
│  │                    Orchestration (SMO)                       │  │
│  │  ┌─────────────────┐    ┌──────────────────────────────┐   │  │
│  │  │   Non-RT RIC    │    │    Outras funções SMO        │   │  │
│  │  │   (rApp host)   │    │  (FCAPS, Inventory, etc.)    │   │  │
│  │  └────────┬────────┘    └──────────────────────────────┘   │  │
│  └───────────┼─────────────────────────────────────────────────┘  │
│              │ A1 Interface (políticas de alto nível)               │
│  ┌───────────┼─────────────────────────────────────────────────┐  │
│  │           ▼                                                   │  │
│  │   ┌──────────────┐         near-RT RIC                       │  │
│  │   │   xApp 1     │  (latência: 10ms a 1s)                    │  │
│  │   ├──────────────┤                                           │  │
│  │   │   xApp 2     │  ← ESTE PROJETO SIMULA ESTE NÍVEL        │  │
│  │   ├──────────────┤                                           │  │
│  │   │   xApp N     │                                           │  │
│  │   └──────────────┘                                           │  │
│  └────────────────────────────┬────────────────────────────────┘  │
│                                │ E2 Interface                       │
│  ┌─────────────────────────────┼────────────────────────────────┐  │
│  │                             ▼                                  │  │
│  │  ┌──────┐    ┌──────┐    ┌──────┐    ← O-CU (Control Plane)  │  │
│  │  │O-CU-C│    │O-CU-U│    │O-DU  │    ← O-DU (Distributed)    │  │
│  │  └──────┘    └──────┘    └──┬───┘                             │  │
│  │                              │ Fronthaul (eCPRI/Open FH)       │  │
│  │                           ┌──┴───┐                             │  │
│  │                           │ O-RU │    ← O-RU (Radio Unit)      │  │
│  └───────────────────────────┴──────┴───────────────────────────┘  │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

### Componentes do O-RAN em Detalhe

**O-RU (O-RAN Radio Unit):**
- Hardware de antena e amplificador de rádio
- Converte sinais digitais em ondas de rádio
- Implementa as funções de baixo nível da camada PHY (Lower-PHY)
- Conecta ao O-DU via fronthaul usando o protocolo eCPRI

**O-DU (O-RAN Distributed Unit):**
- Processa as funções de tempo real da pilha de protocolo (Upper-PHY, MAC, RLC)
- Deve ter latência < 1ms (processa scheduling de recursos de rádio)
- Roda em servidores de propósito geral com aceleradores (FPGAs, GPUs)

**O-CU (O-RAN Central Unit):**
- Gerencia o plano de controle (PDCP, RRC) e plano de usuário (PDCP, SDAP)
- Cobre múltiplos O-DUs
- Menor requisito de latência, pode rodar em cloud regional

**near-RT RIC (Near Real-Time RAN Intelligent Controller):**
- Controlador de IA com latência entre **10ms e 1 segundo**
- Hospeda **xApps** — aplicações de otimização em loop fechado
- Recebe métricas E2 dos nós RAN, executa ML, envia policies
- **Este projeto simula este componente**

**Non-RT RIC (Non Real-Time RIC):**
- Controlador de IA com latência **> 1 segundo** (tipicamente minutos/horas)
- Hospeda **rApps** — aplicações de análise e configuração estratégica
- Envia políticas de alto nível para a near-RT RIC via interface A1
- Parte do SMO (Service Management and Orchestration)

---

## 5. RIC: O Cérebro Inteligente do O-RAN

### near-RT RIC em Detalhe

A near-RT RIC é onde a inteligência artificial da rede vive. Ela foi projetada para ter:

**Latência operacional:** 10ms a 1 segundo
- Suficientemente rápida para reagir a eventos de rede em tempo quasi-real
- Não tão rápida quanto o O-DU (que opera em sub-milissegundo para scheduling)

**Componentes internos da near-RT RIC:**

```
┌─────────────────────────────────────────────────────┐
│                    near-RT RIC                       │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │  xApp 1  │  │  xApp 2  │  │  xApp N  │           │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
│       │              │              │                  │
│  ┌────┴──────────────┴──────────────┴────────────┐  │
│  │           Messaging Infrastructure              │  │
│  │        (publish/subscribe interno)              │  │
│  └────────────────────────┬───────────────────────┘  │
│                            │                          │
│  ┌─────────────────────────┴─────────────────────┐  │
│  │           E2 Termination                        │  │
│  │    (gerencia conexões com nós E2/gNBs)          │  │
│  └─────────────────────────┬─────────────────────┘  │
│                            │ E2 Interface             │
└────────────────────────────┼─────────────────────────┘
                             │
                     [gNB-001] [gNB-002] [gNB-003]
```

**No projeto:**
- O **gRPC Ingestion Service** simula a E2 Termination
- O **ML Service** simula tanto o Messaging Infrastructure quanto uma xApp
- Os 3 nós `gNB-001`, `gNB-002`, `gNB-003` no `simulator.py` são os nós RAN gerenciados

### E2 Service Models (E2SM)

Na especificação O-RAN, os nós RAN publicam dados via **E2 Service Models**:

| E2SM | Conteúdo | Equivalente no Projeto |
|---|---|---|
| E2SM-KPM (KPI Monitoring) | Métricas de performance por célula | `Metric` no `metrics.proto` |
| E2SM-RC (RAN Control) | Control de parâmetros da RAN | `recommended_action` no resultado |
| E2SM-NI (Network Interfaces) | Mensagens de interface X2/Xn | Não implementado (simplificado) |
| E2SM-CCC (Cell Configuration Control) | Configuração de células | Não implementado |

O `metrics.proto` deste projeto define exatamente os campos de um E2SM-KPM simplificado:

```protobuf
message Metric {
    string node_id = 1;       // Identificador da célula/gNB
    float latency = 2;         // KPI: latência em ms
    float throughput = 3;      // KPI: throughput em Mbps
    float packet_loss = 4;     // KPI: taxa de perda de pacotes (%)
    float jitter = 5;          // KPI: variação de latência em ms
    int64 timestamp = 6;       // Timestamp Unix em ms
}
```

---

## 6. xApps e rApps: Aplicações de IA na RAN

### O que são xApps?

**xApps** (x de "e**X**tended")  são microaplicações que rodam dentro da **near-RT RIC** para otimizar a RAN em tempo real.

**Características de uma xApp:**
- **Latência de decisão:** 10ms a 1 segundo
- **Ciclo de vida:** Containerizadas (Docker/Kubernetes)
- **Interface de dados:** E2 interface (recebe KPIs, envia controles)
- **Interface norte:** Expõe resultados via API (para o Non-RT RIC via A1)
- **Autonomia:** Opera em loop fechado sem intervenção humana

**Exemplos de xApps reais:**
| xApp | Função | Métricas Usadas |
|---|---|---|
| Traffic Steering | Redireciona UEs entre células | CQI, RSRP, throughput |
| Anomaly Detection | **Este projeto** | Latência, throughput, packet loss, jitter |
| Admission Control | Decide aceitar/rejeitar novos UEs | Carga da célula, QoS requirements |
| Load Balancing | Distribui carga entre gNBs | PRB utilization, número de UEs |
| Handover Optimization | Melhora decisões de handover | RSRP, RSRQ, velocidade do UE |
| QoS Management | Garante SLAs por slice de rede | Latência por fluxo, throughput por slice |

**A xApp deste projeto (Anomaly Detection xApp):**
```python
# services/ml-service/app/model.py
def predict(metric_dict, clf, scaler):
    features = extract_features(metric_dict)          # E2SM-KPM → feature vector
    features_scaled = scaler.transform([features])    # Normalização
    prediction = clf.predict(features_scaled)[0]      # RandomForest inference
    confidence = max(clf.predict_proba(features_scaled)[0])
    action = _recommend_action(status)                # Mapa status → ação
    return {"status": status, "confidence": confidence, "action": action}
```

### O que são rApps?

**rApps** (r de "**R**easoning" ou "Non-**R**eal-Time") são aplicações que rodam na **Non-RT RIC**, parte do SMO.

**Características de uma rApp:**
- **Latência de decisão:** > 1 segundo (tipicamente minutos, horas ou dias)
- **Função:** Análise estratégica, treinamento de modelos, configuração de políticas
- **Interface:** A1 (envia políticas para near-RT RIC), O1 (configuração de nós)
- **Exemplos:** Retreinamento de modelos de ML, análise de capacidade, previsão de demanda

**Diferença fundamental xApp vs rApp:**

```
                Tempo de Decisão

< 1ms    ─────── O-DU scheduler (tempo real físico)
10ms-1s  ─────── xApp na near-RT RIC   ← ESTE PROJETO
> 1s     ─────── rApp na Non-RT RIC
> 1 hora ─────── Análise OSS/BSS manual
```

**Como xApp e rApp colaboram:**

```
[rApp]  →  A1 Policy: "Priorize latência para slice URLLC"
              ↓
[xApp]  recebe a política → ajusta threshold de detecção de anomalia
              ↓
[xApp]  detecta congestionamento → recomenda BALANCEAR_CARGA
              ↓
[nó RAN] recebe comando E2 Control → executa handover de UEs
```

Neste projeto, a **API Gateway** expõe os resultados da xApp de forma que um sistema externo (Non-RT RIC / rApp) poderia consumir via `/network/alerts` para tomar decisões estratégicas.

---

## 7. Interfaces do O-RAN: E2, A1, O1

### Interface E2 — O Canal de Dados da xApp

A interface **E2** conecta a near-RT RIC aos nós RAN (gNB, en-gNB). É a interface mais crítica para xApps.

**Protocolos:** E2AP (E2 Application Protocol), sobre SCTP/IP

**Funções:**
1. **E2 Setup:** Nó RAN registra na near-RT RIC, anuncia E2 Service Models suportados
2. **Subscription:** xApp solicita receber KPIs específicos de células específicas
3. **Indication:** Nó RAN envia KPIs para a xApp (streaming periódico ou event-triggered)
4. **Control:** xApp envia comandos de controle para o nó RAN
5. **Policy:** xApp configura políticas persistentes no nó RAN

**No projeto — simulação da interface E2:**

```
O-RAN Real:                      Este Projeto:
━━━━━━━━━━━━━━━━━━━━             ━━━━━━━━━━━━━━━━━━━━━━━━━━━
gNB ─── E2AP/SCTP ─→ RIC         simulator.py ─── gRPC ─→ ML Service
                                  (gNB simulado)            (RIC simulado)

E2 Subscription Request   ≈       grpc_client.stream_from_grpc()
E2 Indication (KPM)       ≈       Metric proto (latency, throughput, ...)
E2 Control Request        ≈       recommended_action no InferenceResult
```

O projeto usa **gRPC** com Protocol Buffers em vez do E2AP real, pois:
- gRPC também é baseado em streaming binário eficiente
- Protocol Buffers são semanticamente equivalentes ao ASN.1 do E2AP
- Implementação muito mais rápida para fins de portfólio

### Interface A1 — Políticas do Non-RT para near-RT RIC

A interface **A1** carrega **políticas de alto nível** do Non-RT RIC (rApps) para a near-RT RIC (xApps).

**Tipos de mensagens A1:**
- **A1-P (Policy):** "Dê prioridade a UEs com QCI=1 nas células sobrecarregadas"
- **A1-EI (Enrichment Information):** "Previsão de demanda para as próximas 2 horas"
- **A1-ML (ML Model):** Distribuição de modelos de ML para xApps

**No projeto:** A API Gateway simula a interface A1 northbound — a rota `GET /network/alerts` expõe anomalias que um Non-RT RIC poderia consumir para ajustar políticas.

### Interface O1 — Gerenciamento e Configuração

A interface **O1** é o canal de gerenciamento (FCAPS: Fault, Configuration, Accounting, Performance, Security) do SMO para todos os componentes O-RAN.

**No projeto:** A rota `GET /network/status` simula um report O1 de performance — um operador (ou sistema OSS) pode consultar o estado atual de todos os nós.

---

## 8. Como Este Projeto Mapeia o O-RAN

### Tabela de Mapeamento Completo

| Componente O-RAN Real | Componente deste Projeto | Arquivo Principal | Observação |
|---|---|---|---|
| gNodeB (gNB) | Nós simulados: gNB-001, gNB-002, gNB-003 | `simulator.py` | 3 nós com comportamento normal/congestionado |
| E2 Interface | gRPC `StreamMetrics` | `metrics.proto`, `server.py` | Streaming binário contínuo |
| E2SM-KPM Indication | `Metric` protobuf | `metrics.proto` | latency, throughput, packet_loss, jitter |
| E2 Termination | gRPC Server | `server.py` | Aceita stream subscriptions |
| near-RT RIC | ML Service (container) | `inference.py` | Loop de inferência + estado em memória |
| xApp | RandomForest + lógica de decisão | `model.py` | Classifica estado, recomenda ação |
| xApp Messaging Bus | Thread-safe `_results` list + Lock | `inference.py` | Equivalente ao Internal Messaging Infrastructure |
| A1 Interface (northbound) | REST API `/network/alerts` | `routes.py` | Expõe anomalias para consumo externo |
| O1 Interface (performance) | REST API `/network/status` | `routes.py` | Relatório de performance dos nós |
| Non-RT RIC / SMO | API Gateway (proxy) | `main.py`, `routes.py` | Agrega e expõe dados para operadores/sistemas |
| Kubernetes/Orchestration | `infra/k8s/` + HPA | `k8s/*.yaml` | Escalabilidade automática |

### Diagrama de Arquitetura do Projeto

```
┌─────────────────────────────────────────────────────────────────────┐
│                      AI Network Optimizer                            │
│                                                                      │
│  SIMULAÇÃO DOS gNBs (O-RU/O-DU/O-CU simulados)                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │           grpc-ingestion service (porta 50051)              │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │    │
│  │  │   gNB-001    │  │   gNB-002    │  │   gNB-003    │     │    │
│  │  │ 70%: normal  │  │ 70%: normal  │  │ 70%: normal  │     │    │
│  │  │ 30%: congest │  │ 30%: congest │  │ 30%: congest │     │    │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │    │
│  │         └─────────────────┴─────────────────┘              │    │
│  │                     gRPC StreamMetrics                       │    │
│  └────────────────────────────┬───────────────────────────────┘    │
│                                │ E2 Interface simulada (gRPC)        │
│  NEAR-RT RIC simulada           │                                    │
│  ┌─────────────────────────────┼────────────────────────────────┐  │
│  │           ml-service (porta 8001)                             │  │
│  │                             │                                  │  │
│  │  ┌──────────────────────────▼──────────────────────────────┐ │  │
│  │  │              grpc_client.py (stream consumer)            │ │  │
│  │  │  while True:                                              │ │  │
│  │  │    metric = receive_from_grpc()                           │ │  │
│  │  │    result = model.predict(metric)                         │ │  │
│  │  │    _results.append(result)                                │ │  │
│  │  └──────────────────────────────────────────────────────────┘ │  │
│  │                                                                 │  │
│  │  ┌──────────────────────────────────────────────────────────┐ │  │
│  │  │                     xApp (model.py)                      │ │  │
│  │  │  RandomForest → NORMAL / CONGESTIONADO / DEGRADADO       │ │  │
│  │  │  Confidence score + Recommended Action                    │ │  │
│  │  └──────────────────────────────────────────────────────────┘ │  │
│  └────────────────────────────┬────────────────────────────────┘  │
│                                │ HTTP (A1/O1 simulados)              │
│  SMO / NON-RT RIC simulado      │                                    │
│  ┌─────────────────────────────┼────────────────────────────────┐  │
│  │           api-gateway (porta 8000)                            │  │
│  │                             │                                  │  │
│  │   GET /network/status    ◄──┘                                 │  │
│  │   GET /network/history                                        │  │
│  │   GET /network/alerts   (interface A1 simulada)               │  │
│  │   GET /network/node/{id}                                      │  │
│  │   POST /network/predict                                       │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                │                                      │
│                         Cliente / Operador                            │
│                    (Dashboard, rApp, OSS/BSS)                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 9. O Loop de Inferência em Tempo Real

Esta é a parte mais importante para entender como o projeto funciona como um sistema O-RAN vivo.

### O Conceito: Closed-Loop Automation

O O-RAN define **closed-loop automation** como o ciclo contínuo onde a rede:
1. **Observa** seu próprio estado (métricas KPM)
2. **Analisa** os dados com IA/ML
3. **Decide** qual ação tomar
4. **Executa** a ação
5. **Observa** o resultado da ação → volta ao passo 1

O intervalo desse loop na near-RT RIC é **10ms a 1 segundo**.

### Como o Loop Funciona Neste Projeto

**Passo 0 — Inicialização (acontece uma vez, ao subir o container)**

```python
# services/ml-service/app/inference.py

@app.on_event("startup")
async def startup_event():
    """Dispara o loop de inferência em background quando o container inicia."""
    thread = Thread(target=_background_worker, daemon=True)
    thread.start()
    # Thread daemon = morre junto com o processo principal
```

O loop é uma **thread de background** que roda para sempre enquanto o container estiver vivo. Não bloqueia o servidor HTTP FastAPI.

**Passo 1 — Conexão à fonte de dados (E2 Interface)**

```python
# services/ml-service/app/grpc_client.py

def stream_from_grpc(on_metric_callback):
    """Abre stream gRPC com o serviço de ingestão (gNBs simulados)."""
    channel = grpc.insecure_channel("grpc-ingestion:50051")
    stub = NetworkMetricsStub(channel)

    for metric_proto in stub.StreamMetrics(Empty()):
        # Converte Protobuf → dict Python
        metric_dict = {
            "node_id": metric_proto.node_id,
            "latency": metric_proto.latency,
            "throughput": metric_proto.throughput,
            "packet_loss": metric_proto.packet_loss,
            "jitter": metric_proto.jitter,
            "timestamp": metric_proto.timestamp
        }
        on_metric_callback(metric_dict)  # Chama inferência
```

O `stub.StreamMetrics()` é um **generator gRPC** — ele nunca termina. O Python itera sobre ele indefinidamente, recebendo uma nova métrica a cada ~1 segundo do simulador.

**Passo 2 — Geração das métricas (simulador de gNB)**

```python
# services/grpc-ingestion/app/simulator.py

NODES = ["gNB-001", "gNB-002", "gNB-003"]

def generate_metric():
    node_id = random.choice(NODES)

    if random.random() < 0.30:  # 30% de chance de congestionamento
        return _congested_metric(node_id)
    else:                        # 70% de chance de normal
        return _normal_metric(node_id)

def _normal_metric(node_id):
    return {
        "node_id": node_id,
        "latency": random.uniform(5, 30),         # ms — baixa latência
        "throughput": random.uniform(80, 150),     # Mbps — alta capacidade
        "packet_loss": random.uniform(0, 1),       # % — quase zero
        "jitter": random.uniform(1, 5)             # ms — estável
    }

def _congested_metric(node_id):
    return {
        "node_id": node_id,
        "latency": random.uniform(80, 250),        # ms — alta latência
        "throughput": random.uniform(5, 40),       # Mbps — baixa capacidade
        "packet_loss": random.uniform(5, 20),      # % — alta perda
        "jitter": random.uniform(20, 60)           # ms — instável
    }
```

**Passo 3 — Inferência (xApp ML)**

```python
# services/ml-service/app/model.py

def predict(metric_dict, clf, scaler):
    """Coração da xApp: classifica o estado da rede."""

    # Extrai features relevantes para o modelo
    features = extract_features(metric_dict)
    # → [latency, throughput, packet_loss, jitter]
    # → ex: [145.2, 12.3, 8.7, 35.1]

    # Normaliza usando o scaler treinado (StandardScaler)
    features_scaled = scaler.transform([features])
    # → transforma para média=0, std=1 (mesma escala do treino)

    # Inferência: RandomForest com 100 árvores
    prediction = clf.predict(features_scaled)[0]
    probabilities = clf.predict_proba(features_scaled)[0]
    confidence = float(max(probabilities))

    # Mapeia classe numérica → label
    labels = {0: "NORMAL", 1: "CONGESTIONADO", 2: "DEGRADADO"}
    status = labels[prediction]

    # Recomenda ação de controle (simulando E2 Control)
    action = _recommend_action(status)

    return {
        "node_id": metric_dict["node_id"],
        "status": status,           # Estado classificado
        "confidence": confidence,    # Confiança do modelo (0.0–1.0)
        "recommended_action": action # Ação de otimização
    }

def _recommend_action(status):
    actions = {
        "NORMAL": "NENHUMA",
        "CONGESTIONADO": "BALANCEAR_CARGA",
        "DEGRADADO": "REROUTING_EMERGENCIAL"
    }
    return actions.get(status, "NENHUMA")
```

**Passo 4 — Buffer de resultados (state management da xApp)**

```python
# services/ml-service/app/inference.py

_results = []    # Lista compartilhada entre thread de inferência e HTTP handlers
_lock = Lock()   # Mutex para acesso thread-safe

def _add_result(result):
    """Adiciona resultado ao buffer circular (máx 200 entradas)."""
    with _lock:
        _results.append(result)
        if len(_results) > 200:
            _results.pop(0)   # Remove o mais antigo (FIFO)
```

O buffer de 200 entradas é equivalente ao **state management** de uma xApp real:
- Cada xApp mantém estado em memória (não em banco de dados) para garantir latência < 1s
- O buffer FIFO garante que apenas dados recentes influenciam as decisões

**Passo 5 — Exposição northbound (A1/O1 simulados)**

```python
# services/api-gateway/app/routes.py

@router.get("/network/status")
async def get_network_status():
    """Agrega status atual de todos os nós (equivale a um report O1 de KPMs)."""

    results = await _call_ml("/results?limit=200")

    # Lógica de agregação: último status por nó
    node_status = {}
    for r in reversed(results):
        node_id = r.get("node_id")
        if node_id and node_id not in node_status:
            node_status[node_id] = r

    # Hierarquia de severidade: DEGRADADO > CONGESTIONADO > NORMAL
    overall = "NORMAL"
    for ns in node_status.values():
        if ns.get("status") == "DEGRADADO":
            overall = "DEGRADADO"
            break
        elif ns.get("status") == "CONGESTIONADO":
            overall = "CONGESTIONADO"

    return {
        "overall_status": overall,
        "nodes": list(node_status.values()),
        "total_nodes": len(node_status)
    }
```

### Timeline do Loop Completo

```
t=0ms      Simulador gera métrica do gNB-002 (congestionado)
           latency=187ms, throughput=15Mbps, packet_loss=12%, jitter=45ms

t=1ms      gRPC server envia Metric proto para o ML Service
           (transmissão binária via Protocol Buffers — < 1ms)

t=2ms      grpc_client.py recebe proto, converte para dict Python

t=3ms      model.extract_features() → [187.0, 15.0, 12.0, 45.0]
           scaler.transform() → [2.1, -1.8, 1.9, 2.3] (normalizado)

t=4ms      RandomForest.predict() → classe 1 = "CONGESTIONADO"
           RandomForest.predict_proba() → [0.02, 0.94, 0.04]
           confidence = 0.94

t=5ms      _add_result() → resultado no buffer (thread-safe)

           {
             "node_id": "gNB-002",
             "status": "CONGESTIONADO",
             "confidence": 0.94,
             "recommended_action": "BALANCEAR_CARGA",
             "timestamp": "2026-03-26T..."
           }

t=1000ms   Próxima métrica gerada pelo simulador
           (intervalo configurável via INFERENCE_INTERVAL)

t=qualquer  Cliente HTTP faz GET /network/alerts
           API Gateway chama ML Service → filtra status != NORMAL
           Retorna: gNB-002 CONGESTIONADO (confiança 94%) → BALANCEAR_CARGA
```

**Latência total do loop:** ~5ms de processamento + ~1s de intervalo entre métricas
- Em produção real com E2AP: o intervalo seria definido pelo período de subscription (10ms a 1s)
- O modelo RandomForest contribui com <5ms do total

### Mecanismo de Fallback e Resiliência

```python
# services/ml-service/app/grpc_client.py

def _background_worker():
    consecutive_errors = 0
    MAX_ERRORS = 5

    while True:
        try:
            stream_from_grpc(on_metric_callback=_process_metric)
            consecutive_errors = 0  # Reset após sucesso

        except grpc.RpcError as e:
            consecutive_errors += 1

            if consecutive_errors >= MAX_ERRORS:
                # Fallback: usa simulador local se gRPC indisponível
                logger.warning("gRPC indisponível, usando simulador local")
                _use_local_simulator()
                consecutive_errors = 0
            else:
                # Backoff exponencial: 1s, 2s, 4s, 8s, 16s
                wait = 2 ** consecutive_errors
                time.sleep(wait)
```

Esse padrão de **reconnection com exponential backoff** é um requisito real para xApps em O-RAN: a near-RT RIC precisa continuar operando mesmo que conexões E2 falhem temporariamente.

---

## 10. Tecnologias Usadas e Por Quê

### gRPC + Protocol Buffers — A "Interface E2 Simulada"

**Por que gRPC em vez de REST para a ingestão?**

Na especificação O-RAN, a interface E2 usa **SCTP (Stream Control Transmission Protocol)** com **ASN.1 (Abstract Syntax Notation)** — protocolos de telecomunicações que permitem streaming eficiente de dados binários estruturados.

gRPC é a escolha moderna equivalente para ambientes de software:

| Característica       | E2AP Real           | gRPC (neste projeto)       | REST/JSON           |
| Formato              | ASN.1 PER (binário) | Protocol Buffers (binário) | JSON (texto)        |
| Streaming            | Nativo (SCTP)       | Nativo (HTTP/2)            | Polling ou SSE      |
| Tamanho da mensagem  | ~50 bytes           | ~30 bytes                  | ~200 bytes (JSON)   |
| Latência             | < 1ms               | ~1ms                       | ~5ms                |
| Contrato de schema   | ASN.1 schema        | .proto file                | OpenAPI (opcional)  |
| Geração de código    | ASN.1 compilers     | `protoc`                   | Swagger codegen     |

O arquivo `metrics.proto` é semanticamente equivalente ao E2SM-KPM schema definido pelo O-RAN Alliance.

### FastAPI — O Framework para xApps em Python

FastAPI foi escolhido porque:
- **Async nativo:** Compatível com o modelo de thread de background (loop de inferência)
- **Tipagem:** Pydantic models = equivalente a schemas de mensagens A1/O1
- **OpenAPI automático:** `/docs` gera documentação interativa (útil para demonstrar a API)
- **Performance:** Uvicorn ASGI = alta concorrência para requisições simultâneas

### RandomForest — O Modelo de ML da xApp

Por que RandomForest e não LSTM, Transformer ou outro modelo?

**Requisitos de uma xApp near-RT RIC:**
1. **Latência < 10ms:** RandomForest = ~1-5ms. LSTM = ~50-200ms (com GPU)
2. **Sem GPU:** Hardware de borda pode não ter GPU. RF roda em CPU comum
3. **Interpretabilidade:** Feature importance do RF é útil para auditoria regulatória
4. **Robustez a ruído:** Ensemble de 100 árvores é robusto a métricas ruidosas
5. **Treinamento rápido:** 3000 amostras, treina em segundos (sem necessidade de fine-tuning contínuo)

**Limitação do RF:** Não captura padrões temporais. Para uma xApp real em produção, um **LSTM** ou **Transformer leve** seria mais adequado para detectar tendências (ex: latência subindo progressivamente). O projeto poderia ser estendido com uma janela deslizante de métricas como features temporais.

### Docker + Kubernetes — Infraestrutura de Cloud-Native RAN

O O-RAN define que todos os componentes (near-RT RIC, xApps) devem ser **cloud-native** — containerizados, orquestrados por Kubernetes.

**Por que Kubernetes para xApps?**

```yaml
# infra/k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  scaleTargetRef:
    name: ml-service          # O ml-service (xApp) escala automaticamente
  minReplicas: 2
  maxReplicas: 8              # Até 8 instâncias em carga alta
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        averageUtilization: 70
```

Em O-RAN real, a near-RT RIC (e suas xApps) escalam horizontalmente quando há mais nós E2 conectados ou mais UEs para gerenciar. O HPA do Kubernetes simula exatamente esse comportamento.

---

## 11. Métricas de Rede Simuladas

As quatro métricas simuladas são **KPIs fundamentais de QoS (Quality of Service)** definidos pelo 3GPP e medidos por E2SM-KPM:

### Latência (RTT / One-Way Delay)

**O que é:** Tempo de ida e volta de um pacote de dados entre o UE e o core da rede.

**Relevância 5G:**
- **eMBB:** < 50ms aceitável
- **URLLC:** < 1ms obrigatório
- **Causa mais comum de alta latência:** Congestionamento no backhaul, sobrecarga no O-DU scheduler, interferência de rádio

**No simulador:**
```python
Normal:      5–30 ms   → rede saudável
Congestion: 80–250 ms  → backhaul saturado
Degraded:  200–500 ms  → falha grave (loop de retransmissão, timeout)
```

### Throughput (Taxa de Transferência)

**O que é:** Quantidade de dados transmitidos com sucesso por segundo (Mbps).

**Relevância 5G:**
- **eMBB peak:** 20 Gbps teórico
- **Célula típica carregada:** 100–500 Mbps (compartilhado entre UEs)
- **Causa mais comum de baixo throughput:** Alta carga de UEs, interferência, handovers excessivos

**No simulador:**
```python
Normal:     80–150 Mbps  → célula com carga moderada
Congestion:  5–40 Mbps   → célula sobrecarregada
Degraded:    1–15 Mbps   → cell edge / interferência severa
```

### Packet Loss (Taxa de Perda de Pacotes)

**O que é:** Porcentagem de pacotes que não chegam ao destino.

**Relevância 5G:**
- **Normal:** < 0.1% (com HARQ e ARQ, a camada de rádio compensa)
- **Tolerável:** < 1% para dados best-effort
- **Crítico:** > 1% para VoNR (voz), > 5% causa degradação severa de QoS

**No simulador:**
```python
Normal:     0–1%    → rede saudável com HARQ funcionando
Congestion: 5–20%   → buffers cheios, descarte de pacotes
Degraded:  15–40%   → falha de enlace severa
```

### Jitter (Variação de Latência)

**O que é:** Variação na latência entre pacotes consecutivos. Uma latência média de 10ms com jitter de 50ms significa que alguns pacotes chegam em 10ms e outros em 60ms.

**Relevância 5G:**
- **VoNR (Voice over NR):** Jitter > 30ms causa qualidade de voz degradada
- **Gaming / VR:** Jitter > 20ms causa "lag" perceptível
- **Causa:** Congestionamento variável, interferência intermitente, handovers

**No simulador:**
```python
Normal:    1–5 ms   → transmissão estável e previsível
Congestion: 20–60 ms → variabilidade alta (buffers cheios/vazios)
Degraded:   40–100ms → rede instável (candidato a handover)
```

### Correlação das Métricas com Classes

| Classe        | Latência    | Throughput  | Packet Loss | Jitter      |
|---------------|-------------|-------------|-------------|-------------|
| NORMAL        | 5–30 ms     | 80–150 Mbps | 0–1%        | 1–5 ms      |
| CONGESTIONADO | 80–250 ms   | 5–40 Mbps   | 5–20%       | 20–60 ms    |
| DEGRADADO     | 200–500 ms  | 1–15 Mbps   | 15–40%      | 40–100 ms   |

---

## 12. O Modelo de ML: RandomForest na RAN

### Dataset de Treinamento

```python
# services/ml-service/app/trainer.py

# 3000 amostras geradas sinteticamente:
# - 1000 amostras NORMAL     (classe 0)
# - 1000 amostras CONGESTIONADO (classe 1)
# - 1000 amostras DEGRADADO  (classe 2)

X = [latency, throughput, packet_loss, jitter]   # 4 features
y = [0, 1, 2]                                    # 3 classes

# Split estratificado: 80% treino / 20% teste
# Mantém proporção de classes em ambos os splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalização: StandardScaler
# Transforma cada feature para média=0, desvio_padrão=1
# Essencial porque latência (ms) e throughput (Mbps) têm escalas muito diferentes
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Modelo: RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)
```

### Por que StandardScaler é Crucial

Sem normalização, o modelo seria influenciado desproporcionalmente por features com valores maiores:

```
Sem normalização:
- Throughput: 5–150 Mbps  → diferença de 145 pontos
- Latência: 5–500 ms      → diferença de 495 pontos ← domina o modelo!
- Packet loss: 0–40%      → diferença de 40 pontos
- Jitter: 1–100 ms        → diferença de 99 pontos

Com StandardScaler (média=0, std=1):
- Throughput: -1.8 a 1.9   → diferença de 3.7 unidades
- Latência: -0.8 a 2.1     → diferença de 2.9 unidades ← equilibrado
- Packet loss: -0.9 a 2.0  → diferença de 2.9 unidades
- Jitter: -0.7 a 2.2       → diferença de 2.9 unidades
```

### Feature Importance (Por Que o Modelo Funciona)

O RandomForest calcula automaticamente a importância de cada feature para a classificação. Baseado na distribuição dos dados sintéticos, a importância esperada é:

```
Feature Importance (estimada):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Latency     ██████████████████████  ~35%
Packet Loss ████████████████████    ~30%
Throughput  ████████████            ~22%
Jitter      ████████                ~13%
```

**Interpretação:** Latência e packet loss são os melhores discriminadores porque:
- DEGRADADO tem latência 10x maior que NORMAL (500ms vs 30ms)
- CONGESTIONADO tem packet loss 20x maior que NORMAL (20% vs 1%)

### Decisões de Controle Recomendadas

| Status | Ação Recomendada | Tradução O-RAN |
|---|---|---|
| NORMAL | NENHUMA | Sem intervenção |
| CONGESTIONADO | BALANCEAR_CARGA | E2 Control: redistribuir UEs para células vizinhas menos carregadas |
| DEGRADADO | REROUTING_EMERGENCIAL | E2 Control: handover imediato + alertar NOC |

Em uma xApp real, essas ações seriam enviadas como **E2 Control Request** para o gNB, que executaria o handover de UEs ou ajustaria parâmetros de scheduling.

---

## 13. Fluxo de Dados Completo

### Diagrama de Sequência Detalhado

```
gNB Simulado    gRPC Server    gRPC Client    ML Model    API Gateway    Cliente HTTP
     │               │               │             │            │              │
     │  generate()   │               │             │            │              │
     │───────────────>               │             │            │              │
     │               │               │             │            │              │
     │  Metric proto │               │             │            │              │
     │  stream yield │               │             │            │              │
     │───────────────────────────────>             │            │              │
     │               │               │             │            │              │
     │               │   on_metric() │ predict()   │            │              │
     │               │               │─────────────>            │              │
     │               │               │             │            │              │
     │               │               │  {status,   │            │              │
     │               │               │  confidence,│            │              │
     │               │               │  action}    │            │              │
     │               │               │<────────────            │              │
     │               │               │             │            │              │
     │               │   _add_result()             │            │              │
     │               │               │─────────────────────────>              │
     │               │               │             │  _results  │              │
     │               │               │             │  .append() │              │
     │               │               │             │            │              │
(1 segundo depois)   │               │             │            │              │
     │  generate()   │               │             │            │  GET /status │
     │───────────────>               │             │            │<─────────────│
     │  ...          │               │             │            │              │
     │               │               │             │  GET /results?limit=200   │
     │               │               │             │            │─────────────>│
     │               │               │             │            │   [...]      │
     │               │               │             │            │<─────────────│
     │               │               │             │   aggregate│              │
     │               │               │             │   by node  │              │
     │               │               │             │            │──────────────>
     │               │               │             │            │  {overall,   │
     │               │               │             │            │   nodes:[]}  │
     │               │               │             │            │              │
```

### Estados do Sistema

O sistema evolui através dos seguintes estados a cada ciclo de ~1 segundo:

```
Estado inicial (t=0):
┌────────────┬──────────────┬──────────┐
│ gNB-001    │ gNB-002      │ gNB-003  │
│ NORMAL     │ NORMAL       │ NORMAL   │
│ conf: 0.97 │ conf: 0.95   │ conf:0.98│
└────────────┴──────────────┴──────────┘
→ /network/status: overall = NORMAL

Estado após evento (t=5s):
┌────────────┬──────────────┬──────────┐
│ gNB-001    │ gNB-002      │ gNB-003  │
│ NORMAL     │ CONGESTIONADO│ NORMAL   │
│ conf: 0.96 │ conf: 0.94   │ conf:0.97│
└────────────┴──────────────┴──────────┘
→ /network/status: overall = CONGESTIONADO
→ /network/alerts: [{gNB-002, BALANCEAR_CARGA}]

Estado crítico (t=10s):
┌────────────┬──────────────┬──────────┐
│ gNB-001    │ gNB-002      │ gNB-003  │
│ DEGRADADO  │ CONGESTIONADO│ NORMAL   │
│ conf: 0.91 │ conf: 0.93   │ conf:0.98│
└────────────┴──────────────┴──────────┘
→ /network/status: overall = DEGRADADO (maior severidade)
→ /network/alerts: [{gNB-001, REROUTING_EMERGENCIAL}, {gNB-002, BALANCEAR_CARGA}]
```

### Como Rodar e Observar o Loop em Ação

```bash
# 1. Subir todos os serviços
cd infra/
docker-compose up --build

# 2. Observar o loop de inferência em tempo real (logs do ML Service)
docker-compose logs -f ml-service
# Saída esperada:
# INFO: Conectado ao gRPC (grpc-ingestion:50051)
# INFO: gNB-001 → NORMAL (conf: 0.97) | NENHUMA
# INFO: gNB-003 → CONGESTIONADO (conf: 0.94) | BALANCEAR_CARGA
# INFO: gNB-002 → NORMAL (conf: 0.96) | NENHUMA
# INFO: gNB-001 → DEGRADADO (conf: 0.91) | REROUTING_EMERGENCIAL
# ...

# 3. Consultar status atual via API
curl http://localhost:8000/network/status | python -m json.tool

# 4. Monitorar alertas (apenas anomalias)
watch -n 1 "curl -s http://localhost:8000/network/alerts | python -m json.tool"

# 5. Simular on-demand predict (como se fosse uma mensagem A1)
curl -X POST http://localhost:8000/network/predict \
  -H "Content-Type: application/json" \
  -d '{"latency": 200, "throughput": 10, "packet_loss": 15, "jitter": 45}'

# 6. Ver histórico completo (equivalente a log KPM do O1)
curl "http://localhost:8000/network/history?limit=20" | python -m json.tool
```

---

## Resumo Executivo

Este projeto é uma **simulação fiel de um xApp O-RAN para detecção de anomalias** que implementa:

| Conceito Real | Implementação | Tecnologia |
|---|---|---|
| E2 Interface (KPM streaming) | gRPC StreamMetrics | gRPC + Protobuf |
| near-RT RIC | ML Service container | FastAPI + Thread |
| xApp (Anomaly Detection) | RandomForest classifier | scikit-learn |
| E2 Control (ação recomendada) | `recommended_action` field | JSON response |
| A1 Interface (northbound) | `/network/alerts` REST | FastAPI |
| O1 Interface (management) | `/network/status` REST | FastAPI |
| Cloud-native deployment | Docker + Kubernetes | Docker Compose + k8s |
| Auto-scaling | HPA (2–8 replicas) | Kubernetes |
| Closed-loop automation | Loop de inferência contínuo | Thread daemon |
| Fault tolerance | Exponential backoff + fallback | Python threading |

O loop de inferência roda **continuamente**, com latência de processamento de ~5ms por métrica, consumindo métricas de 3 gNBs simulados a cada ~1 segundo e mantendo um buffer circular de 200 resultados que são expostos via REST API para consumo por dashboards, rApps ou sistemas OSS/BSS.

---

*Documentação gerada para o projeto AI Network Optimizer (simulado) — Portfolio O-RAN xApp*
