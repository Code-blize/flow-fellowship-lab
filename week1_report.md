# Week 1 Report — Flow Fellowship
**Name:** Blessing Obasi-Uzoma
**Track / Workstream:** Decentralized Training
**Week:** 1 (June 22 – June 28, 2026)
**Mentor:** Daniel

---

## 1. Summary of the Week

This week was focused on orientation, vocabulary building, and research problem selection. I attended the fellowship onboarding session, reviewed the Decentralized Training roadmap in full, and worked through the Week 1 orientation material on what decentralized training actually means. I spent significant time building a clear mental model of the federated learning landscape before touching any frameworks or code. By the end of the week I had produced two written artifacts and three research problem proposals, all linked below.

---

## 2. Work Completed

- Reviewed the full Decentralized Training Roadmap 2026 (all 17 pages)
- Studied the Week 1 orientation material: "What Decentralized Training Actually Means"
- Built a personal reference framework covering FL taxonomy, system lifecycles, data heterogeneity types, and privacy mechanics
- Wrote a one-page architecture decision memo answering: which decentralized training architecture fits hospitals, mobile keyboards, home GPUs, and a multi-GPU lab cluster — and why
- Drew a taxonomy diagram separating: distributed training, federated learning, split learning, decentralized LLM training, and compute marketplaces
- Drafted three research problem proposals connected to the Decentralized Training workstream
- Set up the public fellowship GitHub repository

---

## 3. Evidence Links

- **GitHub Repository:** https://github.com/Code-blize/flow-fellowship-lab
- **Architecture Decision Memo:** https://github.com/Code-blize/flow-fellowship-lab/blob/main/week1_architecture_memo.md
- **Research Proposals (all three):** https://github.com/Code-blize/flow-fellowship-lab/blob/main/week1_research_proposals.md

---

## 4. Meetings Attended

**Flow Fellowship Onboarding — June 24, 2026**

Key takeaways:
- The fellowship is contribution-driven. Output and visible evidence matter more than attendance.
- Fellows are expected to develop a year-long research problem connected to their workstream.
- The first three months are a trial period evaluated on consistency, output quality, research maturity, and collaboration.
- Weekly reports with evidence links are mandatory, not optional.
- Demo Day happens at the end of each quarter — a working demo or proof of concept is expected by end of Month 3.

---

## 5. Research Problem Progress

I proposed three research problems this week, all connected to the Decentralized Training workstream and grounded in my existing work on MamaAlert — a three-layer AI system for maternal health triage in Nigeria.

**Proposal 1:** How does non-IID data distribution across Nigeria's six geopolitical zones affect federated learning model performance, and which FL algorithm is most robust to this structural heterogeneity?

**Proposal 2:** What privacy guarantees are achievable for a federated maternal health risk model in low-resource Nigerian settings, and what is the accuracy cost at different differential privacy noise levels?

**Proposal 3:** Can MamaAlert's existing risk prediction layer be restructured as a federated learning prototype with simulated Nigerian state-level clients, and how does it compare to the centralised baseline?

All three proposals are written in full using the fellowship research proposal template and are linked above. I am open to mentor guidance on which one to develop as my primary year-long research problem.

The key insight from this week that connects all three proposals: federated learning does not change what a model learns or how it learns — it changes where learning happens and what is allowed to move. My existing MamaAlert training pipeline transfers directly into a federated setting. The new layer is coordination, privacy, and heterogeneity — not the ML fundamentals.

---

## 6. Blockers / Questions

- I have not yet had my first mentor sync with Daniel. I would like to use that session to confirm which of the three proposals to prioritise and whether restructuring MamaAlert as a federated prototype counts as a valid fellowship contribution.
- I need guidance on whether to use the real Nigeria 2018 DHS dataset or a synthetic proxy for initial FL experiments — specifically whether there are any data sharing restrictions I should be aware of within the fellowship context.

---

## 7. Next Week Plan

1. Complete Week 2 roadmap material: ML and Deep Learning Foundations — specifically gradient mechanics, loss curves, and optimizers, since these are the building blocks of understanding what FedAvg is actually aggregating.
2. Run the Flower quickstart tutorial end-to-end on my local machine and commit the result to GitHub.
3. Have first mentor sync with Daniel and confirm primary research problem.
4. Begin partitioning a small dataset by geopolitical zone as a first step toward the FL simulation.
