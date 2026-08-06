# Week 3 Report — Flow Fellowship
**Name:** Blessing Obasi-Uzoma
**Track / Workstream:** Decentralized Training
**Week:** 3
**Mentor:** Daniel

---

## 1. Summary of the Week

Week 3 moved beyond idealized assumptions into the most important practical challenge in Federated Learning: training under non-IID (non-Independent and Identically Distributed) data. Previous experiments assumed every client owned data from the same distribution. This week, I implemented realistic heterogeneous client partitioning using a Dirichlet distribution, observed the effect on global convergence, and compared IID and non-IID training behaviour experimentally. Several significant engineering challenges were encountered and resolved, including Flower version incompatibilities, dependency conflicts, optimizer migration, and runtime configuration errors. The results clearly demonstrated slower convergence, reduced accuracy, and instability caused by client drift under heterogeneous data — directly connecting to the non-IID problem at the centre of my research proposal.

---

## 2. Work Completed

- Studied IID vs. non-IID data distributions and the concept of client drift
- Implemented Dirichlet-based non-IID client partitioning (α = 0.5) on CIFAR-10
- Modified the Flower Quickstart client and server to support heterogeneous data
- Ran two federated simulations: one IID baseline and one non-IID experiment
- Compared global accuracy and cross-entropy loss across both settings
- Generated convergence plots for both experiments
- Resolved multiple debugging challenges across the Flower framework

---

## 3. Evidence Links

- **GitHub Repository:** https://github.com/Code-blize/flow-fellowship-lab
- **Week 2 Folder:** https://github.com/Code-blize/flow-fellowship-lab/tree/main/week%202
- *(Week 3 experiment code and plots to be committed — link to be added)*

---

## 4. Meetings Attended

- Weekly Team Meeting — Attended Daniel's session on Task-Aware Local Data Selection (TALDS), a plug-and-play preprocessing framework for efficient decentralized learning that operates before local training begins. Key insight: instead of training on all local data regardless of relevance, TALDS filters client data by task relevance before each round — directly reducing the non-IID problem I am investigating.

---

## 5. Research Problem Progress

This week's experiments directly advance my primary research proposal: *"How does non-IID data distribution across Nigeria's geopolitical zones affect federated learning model performance?"*

The CIFAR-10 non-IID simulation served as a controlled proof of concept for the exact problem I will later investigate with DHS data partitioned by geopolitical zone. The results were striking. The IID model reached approximately 96% global accuracy after 10 communication rounds. The non-IID model reached only approximately 48% over the same rounds — a gap of nearly 50 percentage points.

This confirms that the non-IID problem is not theoretical. It is severe and measurable. When I later partition Nigeria's DHS data by geopolitical zone — where North West has fundamentally different maternal health indicators from South South — I expect to observe similar degradation. The key question my research will answer is which algorithm, FedAvg or FedProx, recovers more of that lost performance in a real African health data context.

The TALDS framework Daniel presented this week also raised a new question: could task-aware data selection help address client drift in the DHS setting, where certain zones may have irrelevant or noisy records for maternal risk prediction?

---

## 6. Blockers / Questions

- The two convergence plot figures are generated but need to be committed to GitHub with the experiment code.
- I would like Daniel's guidance on the next step: whether to proceed directly to DHS data preprocessing and zone partitioning, or first implement FedProx on this CIFAR-10 non-IID setup as a comparison baseline before switching to real data.
- DHS project approval for the new federated learning use case has been received. Data preprocessing can begin.

---

## 7. Next Week Plan

1. Commit Week 3 experiment code and convergence plots to GitHub
2. Begin DHS Nigeria 2018 data loading and preprocessing — load the Individual Recode dataset, identify maternal health feature columns, and perform the GPS spatial join to assign each cluster to its geopolitical zone
3. Implement FedProx on the CIFAR-10 non-IID setup as a comparison against FedAvg — this builds the comparison methodology I will later apply to DHS data
4. Confirm with Daniel whether to run FedProx experiments first or move directly to DHS partitioning

---
---

# Technical Research Note — Week 3
## Federated Learning Under Non-IID Data Distributions

**Framework:** Flower Federated Learning Framework
**Dataset:** CIFAR-10
**Language:** Python

---

## Introduction

Federated Learning enables multiple devices to collaboratively train a machine learning model without sharing raw data. Each client trains locally and communicates only model updates to a central server.

The standard Federated Averaging (FedAvg) algorithm assumes that local client updates collectively represent the overall population — an assumption that holds when client datasets are independent and identically distributed (IID). Real-world data rarely satisfies this assumption. Different users generate different behaviours. Different hospitals diagnose different diseases. Different regions speak different languages. Week 3 focused on understanding this challenge and experimentally observing how non-identical data distributions affect global model convergence.

---

## Understanding IID vs. Non-IID Data

**The student analogy:**

Suppose ten students are learning to recognize animals. If every student studies exactly the same mixture of dogs, cats, birds, horses, and fish, they gain similar knowledge. When they combine what they learned, the result closely approximates learning from the complete dataset. This is an IID setting.

Now imagine Student 1 studies only cats, Student 2 only dogs, Student 3 only birds. Each becomes an expert in one class. When they combine knowledge, conflicts arise because each local model was optimized for a different objective. This is a non-IID setting — and it is almost always what real Federated Learning resembles.

---

## Client Drift

Client drift occurs when different clients optimize their models toward different objectives because they each possess different data distributions.

Suppose three clients participate in federated training: Client A contains mostly airplanes, Client B mostly birds, Client C mostly trucks. Each client's local updates move the global model in a different direction. When the server averages these conflicting updates, they partially cancel one another. Instead of cooperating, the clients unintentionally compete. The optimization process oscillates rather than converging smoothly.

Client drift becomes more severe as data heterogeneity increases. This is the central challenge my research proposal addresses in the context of Nigeria's geopolitical zones.

---

## Dirichlet Distribution for Non-IID Partitioning

The Dirichlet distribution provides a principled method for simulating heterogeneous client datasets. Controlled by a concentration parameter α, it determines how similar or different client data distributions are from one another.

| α value | Effect |
|---|---|
| α → ∞ | Approaches IID — all clients receive similar class distributions |
| α = 1.0 | Moderate heterogeneity |
| α = 0.5 | Strong heterogeneity — used in this experiment |
| α → 0 | Extreme heterogeneity — each client receives almost one class only |

Using α = 0.5 simulates a realistic federated scenario where clients hold meaningfully different but not extreme distributions.

---

## Implementation

The Flower Quickstart PyTorch application was modified in two ways.

**Client modification:** Instead of uniformly sampling CIFAR-10 data per client, each client receives a Dirichlet-partitioned subset. The partitioning function assigns class proportions to each client based on draws from a Dirichlet distribution, ensuring no two clients see the same class distribution.

**Server modification:** The server was configured to aggregate over 10 communication rounds, collecting accuracy and loss metrics after each round for later visualization.

Both IID and non-IID experiments used identical neural network architecture, learning rate, and number of communication rounds — ensuring that the only variable was client data distribution.

---

## Experimental Results

| Setting | Starting Accuracy | Final Accuracy (Round 10) | Convergence |
|---|---|---|---|
| IID | ~91% | ~96% | Smooth, stable |
| Non-IID (α=0.5) | ~9% | ~48% | Slow, oscillating |

**Figure 1 — Global Accuracy Comparison:**
*(Insert accuracy plot)*

**Figure 2 — Global Loss Comparison:**
*(Insert loss plot)*

The IID model began at approximately 91% and steadily improved to 96% after ten rounds. The convergence curve was smooth, indicating consistent local client updates.

The non-IID model began at approximately 9% and reached only 48% after the same ten rounds. Unlike the IID experiment, the curve displayed larger fluctuations before gradually stabilizing. The loss curves revealed an even stronger contrast — IID loss decreased smoothly while non-IID loss exhibited multiple spikes before declining.

---

## Reflection

Before this experiment, I expected the non-IID model to converge more slowly. I did not anticipate the performance gap to be as large as what I observed.

The IID model converged rapidly at ~96% accuracy. The non-IID model reached only ~48% over the same rounds — nearly half the performance. This reinforced the concept of client drift in a way that reading alone did not.

One question this raises: how can Federated Learning algorithms be modified to reduce client drift while maintaining privacy under highly heterogeneous distributions? Exploring FedProx, SCAFFOLD, and FedNova may provide answers — and this is exactly the comparison my research proposal is designed to investigate using Nigerian health data rather than image classification.

---

## Debugging Journey

Several practical engineering challenges were encountered and resolved during this week:

- Flower version incompatibilities between older tutorials and newer APIs
- Dependency conflicts involving `flwr-datasets`
- Runtime environment creation errors
- Incorrect TOML dependency declarations
- Package installation from the wrong directory
- Import errors caused by accidental removal of the `load_data()` function
- Optimizer configuration issues when switching from SGD to Adam
- Locating experiment output files generated inside Flower's runtime environments
- Recording evaluation metrics for later visualization

Resolving these issues provided practical experience in debugging distributed machine learning systems that reading tutorials alone would not have given.

---

## Lessons Learned

- Real-world Federated Learning rarely operates under IID assumptions
- Client drift is one of the primary causes of slow convergence
- The Dirichlet distribution provides a practical method for simulating heterogeneous client datasets
- Even small implementation changes can significantly alter global optimization behaviour
- Debugging distributed systems requires understanding both machine learning and software engineering — dependency management, runtime environments, and package configuration are not separate skills

---

## Conclusion

Week 3 moved beyond idealized IID assumptions into more realistic heterogeneous Federated Learning environments. Through Dirichlet-based client partitioning, Flower modifications, and IID vs. non-IID comparison, I gained a deeper understanding of the practical challenges facing real Federated Learning systems.

The experiments demonstrated clearly that heterogeneous client datasets slow convergence, reduce global accuracy, and introduce instability through client drift. The performance gap — 96% IID vs. 48% non-IID — is not an edge case. It is the normal condition in real-world deployments, and it is exactly what I will investigate using Nigeria's geopolitical zone structure and DHS health data in the weeks ahead.

---

## References

1. McMahan, B., et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.*
2. Flower Framework Documentation — https://flower.ai/docs/
3. Flower Datasets Documentation — https://flower.ai/docs/datasets/
4. CIFAR-10 Dataset — https://www.cs.toronto.edu/~kriz/cifar.html
5. Kairouz, P., et al. (2021). *Advances and Open Problems in Federated Learning.*
6. Li, T., et al. (2020). *Federated Optimization in Heterogeneous Networks (FedProx).*
