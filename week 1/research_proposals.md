# Week 1 Research Problem Proposals
**Fellow Name:** Blessing Obasi-Uzoma  
**Track / Workstream:** Decentralized Training  
**Mentor:** Daniel  
**Week:** 1 (June 22 – June 28, 2026)

---

## Proposal 1

**Problem Title:** How Does Non-IID Data Distribution Across Nigeria's Geopolitical Zones Affect Federated Learning Model Performance?

---

**1. Problem Statement**

Most federated learning research assumes that data across clients is either independent and identically distributed (IID) or randomly heterogeneous. In reality, health data in Nigeria is structured by deeply unequal geopolitical, infrastructural, and demographic conditions across the country's six geopolitical zones — North West, North East, North Central, South West, South East, and South South. This structural inequality produces a specific kind of non-IID distribution that is not random but geographically and institutionally patterned.

The problem I want to explore is: when a federated learning system is trained across clients representing Nigeria's geopolitical zones using real or realistic health data, how much does this structured non-IID distribution degrade global model performance compared to an IID baseline? And which federated learning algorithms — FedAvg, FedProx, or FedNova — are most robust to this kind of distribution shift?

---

**2. Why It Matters**

Federated learning is increasingly proposed as a solution for training AI models on sensitive health data without centralising that data. But most benchmarks use toy datasets like CIFAR-10 or MNIST, which do not reflect the distributions seen in African health contexts.

If federated models perform poorly under Nigeria's specific data heterogeneity, that is a deployment risk that affects real maternal and child health outcomes. Understanding this gap is a prerequisite for building trustworthy federated health AI in Nigeria and, by extension, across Sub-Saharan Africa where similar structural inequalities exist.

This problem also creates a gap in the current federated learning literature that my background — geospatial data science, Nigerian DHS data, and familiarity with Nigeria's PHC system — uniquely positions me to address.

---

**3. Prior Research**

- Reviewed Flow Fellowship orientation material: *"What Decentralized Training Actually Means"* — this introduced the distinction between IID and non-IID data and why non-IID is the hard problem in FL.
- Reviewed the Flow Fellowship Cohort 1 Roadmap, specifically the decentralised training workstream direction and weekly themes.
- Familiar with Nigeria 2018 Demographic and Health Survey (DHS) Individual Recode dataset, which I have used in my existing MamaAlert project. This dataset includes health, demographic, and facility-access indicators disaggregated at the state and zone level.
- The concept of geopolitically structured data heterogeneity in Nigeria is informed by my geospatial background — I am familiar with how facility density, urbanisation rates, and education levels differ systematically across zones.

---

**4. Proposed Direction**

I want to simulate a federated learning system using the Flower (flwr) framework, where each client represents one of Nigeria's six geopolitical zones. I will partition a health dataset — either real DHS-derived features or a synthetic proxy — by zone, train a simple classification model (logistic regression or small MLP) under FedAvg and FedProx, and compare:

- Global accuracy under IID vs. zone-partitioned non-IID splits
- Per-client accuracy across zones (to reveal which zones are underserved by the global model)
- Convergence speed across settings

The output will make visible whether standard FL algorithms treat Nigeria's zones fairly or systematically disadvantage minority or low-data zones.

---

**5. Possible Outputs**

- A reproducible Flower simulation with zone-partitioned data (GitHub repo)
- Accuracy and convergence comparison charts across IID and non-IID settings
- A short technical write-up or research note explaining the findings
- Potentially a Medium article framing the results for a general AI audience

---

**6. First 3-Month Milestone**

By the end of Month 3 (mid-September 2026), I want to have:
- A working Flower simulation with at least 6 clients and two FL algorithms implemented
- Experiment results showing accuracy curves for IID vs. non-IID splits
- A written research note (2–4 pages) documenting the methodology, results, and implications

---

**7. Mentor Support Needed**

- Confirmation that this problem is relevant to the current decentralised training workstream direction
- Guidance on whether to use real DHS data or a synthetic proxy for the initial experiment
- Feedback on which FL algorithm comparison is most useful for the workstream — FedAvg vs. FedProx, or a different pair
- Recommended papers or benchmarks that treat geographic or structural non-IID as a distinct research object

---
---

## Proposal 2

**Problem Title:** What Privacy Guarantees Are Achievable for a Federated Maternal Health Risk Model in Low-Resource Nigerian Settings, and What Is the Accuracy Cost?

---

**1. Problem Statement**

Federated learning provides data locality — patient records stay at the facility — but it does not automatically guarantee privacy. A curious server can still infer information about individual patients from the model updates sent by each client. Differential privacy (DP) is the standard mechanism for adding formal privacy guarantees on top of federated learning, but it comes at a cost: adding noise to model updates reduces accuracy.

The problem I want to explore is: what is the practical privacy-utility tradeoff for a federated maternal health risk prediction model operating under the resource constraints typical of Nigerian primary healthcare facilities? Specifically, how much accuracy is lost when differential privacy is applied at different noise levels (epsilon values), and is there a setting where the model remains clinically useful?

---

**2. Why It Matters**

Maternal health data — including pregnancy outcomes, antenatal visit records, and risk indicators — is among the most sensitive data that exists. Any federated system that touches this data must be able to answer the question: *what happens if the server is compromised?*

In Nigeria, where patient data governance frameworks are still developing, federated learning without a clear privacy story is not deployable in practice. A credible answer to this question — grounded in actual experiments — would make federated maternal health AI significantly more trustworthy and more fundable.

This problem also has broader implications. The privacy-utility tradeoff under resource constraints is an underexplored area in the African health AI literature.

---

**3. Prior Research**

- Reviewed Flow Fellowship orientation material on decentralised training, which introduced the concept of privacy as a distinct layer on top of federated learning.
- Familiar with Nigeria 2018 DHS data and its maternal health indicators (antenatal visits, skilled birth attendance, facility delivery rates, pregnancy complications) through my MamaAlert project.
- MamaAlert currently has a risk prediction layer trained on DHS-derived features. This existing work provides a baseline model that can be federated and then subjected to differential privacy experiments.
- General awareness of Opacus (PyTorch's differential privacy library) as the implementation framework for DP-SGD.

---

**4. Proposed Direction**

I will train a maternal health risk prediction model (logistic regression baseline) on DHS-derived features, then wrap it in a Flower federated learning simulation with simulated Nigerian state-level clients. I will then apply differential privacy using the Opacus library at multiple epsilon values (ranging from strict to loose privacy budgets) and measure:

- Global model accuracy at each epsilon setting
- The privacy-utility curve (a plot of accuracy vs. epsilon)
- Whether any epsilon setting preserves clinical usefulness (e.g., accuracy above a defined threshold)

I will also write a short privacy threat model document outlining who the adversaries are, what they could learn without DP, and what DP prevents.

---

**5. Possible Outputs**

- A Flower + Opacus experiment with DP-FL on maternal health proxy data (GitHub repo)
- A privacy-utility tradeoff curve
- A privacy threat model document for federated maternal health AI in Nigeria
- A technical research note (3–5 pages) documenting findings

---

**6. First 3-Month Milestone**

By the end of Month 3 (mid-September 2026), I want to have:
- A working DP-FL experiment with at least three epsilon settings
- A privacy-utility tradeoff chart with interpretation
- A written threat model document covering at least three adversary types
- A draft research note ready for mentor review

---

**7. Mentor Support Needed**

- Guidance on what epsilon values are considered realistic for health data contexts
- Feedback on whether the privacy threat model framing is appropriate for this workstream
- Advice on whether to use real DHS features or a fully synthetic dataset for the initial experiments
- Recommended references on DP in federated health AI

---
---

## Proposal 3

**Problem Title:** Can MamaAlert's Risk Prediction Layer Be Restructured as a Federated Learning Prototype with Simulated Nigerian State-Level Clients?

---

**1. Problem Statement**

MamaAlert is a three-layer AI system for maternal health triage in Nigeria that I have been building. Its middle layer is an ML risk prediction model trained on features from Nigeria's 2018 DHS dataset to estimate maternal risk levels based on symptoms, facility access, and demographic indicators. Currently, this model is trained centrally — all data is in one place.

In a real-world deployment, MamaAlert would need to learn from patient data across Nigeria's 774 Local Government Areas and 36 states. That data cannot and should not be centralised — it belongs to each facility. The problem I want to explore is: can the existing MamaAlert risk prediction layer be restructured as a federated learning prototype where each client represents a Nigerian state, and if so, does it perform comparably to the centralised baseline?

---

**2. Why It Matters**

This proposal is the most practically motivated of the three. It asks not just whether federated learning works in theory, but whether it can work for a specific real system that I am building for a specific African health context.

If this prototype succeeds, MamaAlert becomes a demonstrably deployable federated system — one that PHCs and state health ministries could consider without being asked to share sensitive patient data. If it fails or degrades significantly, that is also a useful finding that shapes how MamaAlert should be built going forward.

This work also produces a live, reproducible system — not just a report — which is the highest-value artifact type in a contribution-driven fellowship.

---

**3. Prior Research**

- I built MamaAlert's risk prediction layer using Nigeria 2018 DHS Individual Recode data, with features including antenatal care visits, skilled birth attendance, maternal education, and facility access distance.
- Reviewed the Flow Fellowship orientation on decentralised training, which introduced Flower as the primary simulation framework.
- Flower (flwr) is already installed on my machine. I have confirmed the installation and reviewed the basic client-server architecture in the Flower documentation.
- I understand that restructuring MamaAlert as a federated system means treating each state (or zone) as a Flower client that trains locally and sends model updates — not raw data — to a central aggregator.

---

**4. Proposed Direction**

I will partition MamaAlert's existing DHS-derived feature dataset by Nigerian state (36 clients) or geopolitical zone (6 clients), implement each partition as a Flower client running the existing logistic regression model, and simulate federated training using FedAvg.

I will compare:
- The federated model's accuracy vs. the existing centralised baseline
- Per-client (per-state) accuracy to identify which states the global model serves poorly
- Communication cost: how many rounds are needed to reach baseline-equivalent accuracy

The prototype will be fully reproducible from a public GitHub repository with clear documentation.

---

**5. Possible Outputs**

- A public GitHub repo: `mamaalert-fl` with a clean Flower simulation
- A comparison table: federated vs. centralised accuracy by state/zone
- An architecture diagram showing the MamaAlert-FL client-server structure
- A short technical write-up or demo video walkthrough
- A LinkedIn post connecting the work to federated AI for African health systems

---

**6. First 3-Month Milestone**

By the end of Month 3 (mid-September 2026), I want to have:
- A working Flower simulation with at least 6 clients (one per geopolitical zone)
- A trained federated model with accuracy results compared against the centralised baseline
- A public GitHub repo with documentation clear enough for another fellow to reproduce
- A written summary of what worked, what degraded, and what it means for MamaAlert's real-world deployment path

---

**7. Mentor Support Needed**

- Confirmation that restructuring an existing project as a federated prototype counts as a valid fellowship contribution
- Guidance on whether 6 clients (zones) or 36 clients (states) is the right starting scope for a first prototype
- Feedback on evaluation metrics — is accuracy the right primary metric, or should I also measure fairness across clients?
- Advice on what "team standards and specifications" I should follow for the Flower implementation

---

*All three proposals connect to the Decentralized Training workstream. Proposal 1 is the most research-oriented. Proposal 2 is the most policy-relevant. Proposal 3 is the most systems-oriented and directly extends existing work. I am open to mentor guidance on which one to pursue as my primary year-long research problem.*
