# Architecture Decision Memo: Which Decentralized Training Approach Fits Which Context?

**Author:** Blessing Obasi-Uzoma  
**Workstream:** Decentralized Training  
**Date:** June 2026  
**Week:** 1 — Orientation Hands-On Work

---

## The Question

Not all decentralized training problems are the same. The right architecture depends on where the data lives, who owns it, how sensitive it is, and what kind of hardware is available. This memo answers: which architecture would I choose for hospitals, mobile keyboards, home GPUs, and a multi-GPU lab cluster — and why?

---

## Case 1: Hospitals

**Architecture: Cross-Silo Federated Learning**

Hospitals hold the most sensitive data that exists — patient records, diagnoses, pregnancy outcomes, disease history. This data cannot leave the institution, legally or ethically. But hospitals still need to collaborate to build models that generalize across populations.

Cross-silo federated learning is the right fit here. Each hospital is a client that trains locally on its own patients. Only the model updates — not the patient records — are sent to a central aggregator. The number of clients is small (tens to hundreds of hospitals, not millions), connections are stable, and each client has meaningful amounts of data.

This is directly relevant to MamaAlert. If the system were deployed across Nigeria's primary healthcare centres, each PHC would be a cross-silo client. Their maternal health records would never leave the facility.

**Key requirement:** Differential privacy on top of federated learning, to protect against gradient inversion attacks.

---

## Case 2: Mobile Keyboards

**Architecture: Cross-Device Federated Learning**

Mobile keyboards learn from how you type — your vocabulary, your shortcuts, your corrections. This is deeply personal data that users would never consent to uploading to a server. But it is also the exact data needed to improve next-word prediction.

Cross-device federated learning is the architecture Google uses for GBoard. Millions of phones each train a small local model on the user's typing history. Updates are sent to a server only when the phone is idle, charging, and on Wi-Fi. The server aggregates millions of tiny updates into a better global model.

The challenge here is scale and unreliability. Clients drop in and out. Data per device is tiny. The non-IID problem is severe — everyone types differently. Algorithms like FedAvg are designed for exactly this setting.

**Key requirement:** Extremely communication-efficient updates, since sending large model weights from millions of phones is expensive.

---

## Case 3: Home GPUs

**Architecture: Decentralized Compute Markets / Swarm Learning**

Home GPU contributors are not data owners — they are compute providers. The question here is not "how do we protect sensitive data?" but "how do we coordinate untrusted, heterogeneous hardware to train a model without a central authority?"

This is the domain of decentralized compute markets like Gensyn and Bittensor. Contributors offer their GPU power in exchange for tokens or rewards. The training protocol is designed to be permissionless — anyone can join, anyone can leave, and the system must verify that contributors are doing honest work.

This is the most decentralized architecture of the four. There is no trusted central server. Consensus and verification mechanisms replace the aggregator.

**Key requirement:** Incentive alignment and Byzantine fault tolerance — the system must detect and exclude contributors who send dishonest updates.

---

## Case 4: Multi-GPU Lab Cluster

**Architecture: Centralized Distributed Training (Data Parallel / DDP)**

In a lab cluster, the researcher owns all the GPUs, all the data is in one place, and the only goal is to train faster. There is no privacy concern, no untrusted party, and no need to keep data local.

This is standard distributed data parallel (DDP) training — the same data is sharded across multiple GPUs, gradients are synchronised via all-reduce operations, and the model trains as if on one very large GPU. PyTorch's DistributedDataParallel handles this natively.

This is the fastest and simplest of the four cases, but it is only appropriate when data centralisation is acceptable.

**Key requirement:** Fast interconnects between GPUs (NVLink or InfiniBand) to keep communication overhead low.

---

## Summary Table

| Context | Architecture | Data stays local? | Main challenge |
|---|---|---|---|
| Hospitals | Cross-silo FL | Yes | Privacy, non-IID |
| Mobile keyboards | Cross-device FL | Yes | Scale, dropout, communication cost |
| Home GPUs | Decentralized compute market | N/A (compute only) | Trust, verification, incentives |
| Multi-GPU lab cluster | Centralized DDP | No (not needed) | Speed, synchronisation |

---

## Personal Reflection

Before this week, I understood "distributed training" loosely as "training on more than one machine." What I now understand is that the *reason* for distributing matters as much as the *method*. Hospitals distribute because of privacy. Mobile devices distribute because of consent. Home GPUs distribute because of incentives. Lab clusters distribute because of speed. Each reason leads to a different architecture, different algorithms, and different failure modes.

This distinction is what I will carry into my research proposals.
