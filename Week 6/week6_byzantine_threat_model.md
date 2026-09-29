# Week 6 — Byzantine Attacks and Byzantine Threat Models

## Federated Learning Fellowship

### Focus
Byzantine attacks, poisoned model updates, threat modelling, and robust aggregation.

---

# 1. Overview

Week 6 focuses on a different kind of Federated Learning failure: a client that is not simply unreliable or disconnected, but actively sends a malicious update to influence the global model.

In the previous weeks, the main challenge was **non-IID data and client drift**. This week asks a different question:

> What happens when one of the clients intentionally sends a harmful update?

Three defenses considered this week are:

- Coordinate-wise Median
- Krum
- Bulyan

For Week 7, I chose **Coordinate-wise Median** as the defense I will implement and test.

---

# 2. Conceptual Exercise — Tracing a Poisoned Round

## Scenario

There are five clients:

- Clients 0–3: honest
- Client 4: Byzantine

The global model parameter is simplified to one value:

```text
w_t = 0.5
```

The honest client updates are:

```text
Client 0 = +0.10
Client 1 = +0.08
Client 2 = +0.12
Client 3 = +0.09
```

The Byzantine client sends:

```text
Client 4 = -2.00
```

## Honest average

The average of the four honest updates is:

```text
(0.10 + 0.08 + 0.12 + 0.09) / 4
= 0.39 / 4
= +0.0975
```

The honest aggregation would therefore move the model to:

```text
0.5 + 0.0975 = 0.5975
```

## FedAvg with the Byzantine update

FedAvg averages all five updates:

```text
(0.10 + 0.08 + 0.12 + 0.09 - 2.00) / 5
= -1.61 / 5
= -0.322
```

So:

```text
FedAvg aggregated update = -0.322
```

The resulting global parameter becomes:

```text
0.5 + (-0.322) = 0.178
```

Compared with the honest result:

```text
Honest result = 0.5975
FedAvg result = 0.1780
```

The poisoned aggregation is therefore:

```text
0.5975 - 0.1780 = 0.4195
```

away from the honest result.

This demonstrates the weakness of ordinary averaging: one extreme update can have a large influence on the global model.

---

## Coordinate-wise Median

The five updates sorted from smallest to largest are:

```text
-2.00, 0.08, 0.09, 0.10, 0.12
```

The middle value is:

```text
0.09
```

Therefore:

```text
Coordinate-wise Median = +0.09
```

Compared with the honest average:

```text
Honest average = +0.0975
Median         = +0.0900
Difference     = 0.0075
```

The median is therefore much closer to the honest aggregation than FedAvg in this poisoned example.

### When can a single Byzantine value influence the median?

With four honest clients and one Byzantine client, simply making an extreme value more negative does not make the median fail. For example, `-2`, `-20`, and `-200` all remain outside the honest cluster.

A more subtle attack is to place the malicious value inside the honest cluster. For example:

```text
0.08, 0.09, 0.095, 0.10, 0.12
```

would make `0.095` the median.

This shows that robustness is not the same as immunity: a median is resistant to extreme outliers, but a carefully constructed update that resembles honest updates can still influence the aggregate.

---

## Krum

Krum evaluates how close each update is to the other updates and selects an update that appears to belong to the consistent group.

With:

```text
n = 5 clients
f = 1 Byzantine client
```

the simplified calculation considers the two closest other updates for each candidate.

| Candidate | Two closest distances | Score |
|---|---:|---:|
| -2.00 | 2.08 + 2.09 | 4.17 |
| 0.08 | 0.01 + 0.02 | 0.03 |
| 0.09 | 0.01 + 0.01 | 0.02 |
| 0.10 | 0.01 + 0.02 | 0.03 |
| 0.12 | 0.02 + 0.03 | 0.05 |

The smallest score belongs to:

```text
+0.09
```

Therefore, Krum would select `+0.09` in this simplified example.

The intuition is that the four honest updates form a tight cluster while `-2.0` is an obvious outlier.

---

# 3. What the Exercise Shows

```text
FedAvg
→ Average everything
→ Extreme updates directly influence the result

Coordinate-wise Median
→ Take the middle value for each coordinate
→ Extreme values have much less influence

Krum
→ Select the update that looks most consistent with the others
→ Return a single selected update
```

The main lesson is that Byzantine robustness depends not only on the aggregation rule, but also on assumptions about the number and behavior of malicious clients.

---

# 4. Threat Model Write-up

## Part 1 — The Adversary

The adversary in this simulation is a single Byzantine client among five participating clients. Clients 0, 1, 2, and 3 are honest, while Client 4 is malicious. The Byzantine client can participate in training like an ordinary client, receive the current global model, perform any local computation it chooses, and return a manipulated model update instead of an honest update. In the conceptual exercise, this is represented by the client sending `-2.0` while the honest clients send updates close to `+0.1`. The attacker's objective is to influence the global model through the aggregation process.

The adversary is assumed to know that it is being aggregated with four honest clients, but it does not know which aggregation method the server is using. It is therefore not assumed to know whether the server uses FedAvg, coordinate-wise median, Krum, or another aggregation rule. The threat model contains only one Byzantine client, so there is no coordination with other malicious clients in this experiment. The attacker also does not receive the other clients' private updates before sending its own update. This gives the experiment a limited adversarial capability: the client can fully control its own update, but it does not have complete information about the server's aggregation process.

## Part 2 — The Attack Surface

FedAvg assumes that client updates can be combined through a weighted average to produce a useful global update. This works reasonably when participating clients are honest, but the assumption breaks when a client deliberately sends a malicious update. In the example, the honest average is `+0.0975`, while the Byzantine client sends `-2.0`. FedAvg combines all five values and produces `-0.322`, moving the simplified global parameter from `0.5` to `0.178`. If the Byzantine client were absent, the model would instead move from `0.5` to `0.5975`. The single malicious update therefore moves the result `0.4195` away from the honest aggregation.

This illustrates the key attack surface in ordinary federated aggregation: the server generally receives model updates rather than the intentions behind those updates. A malicious client can therefore submit an update that is mathematically valid but harmful to the training process. An extreme update can strongly influence the mean because averaging gives every update a direct numerical contribution. Robust aggregation changes this assumption by attempting to reduce the influence of suspicious updates instead of trusting every submitted update equally.

## Part 3 — Chosen Defense

For Week 7, I will implement **coordinate-wise Median**. I chose it because the mechanism is straightforward to understand and directly addresses the weakness demonstrated in the poisoned-round exercise: an extreme malicious value can have a large effect on the mean but becomes an edge value when the updates are sorted, leaving the middle value relatively unaffected. Coordinate-wise Median also provides a useful contrast with the FedAvg baseline used in earlier experiments. The tradeoff is that the median does not preserve information in the same way as averaging, and a malicious client that crafts an update to resemble the honest distribution can be harder to distinguish. A larger or coordinated Byzantine population would also change the threat model and could reduce the protection provided by a simple median-based rule.

## Part 4 — What Robust Aggregation Cannot Fix

Robust aggregation should not be treated as a complete solution to model poisoning. Fang et al. showed that optimization-based local model poisoning attacks can substantially increase the error of models trained with several Byzantine-robust aggregation methods, demonstrating that a defense can still be vulnerable when an attacker deliberately designs an update to exploit the aggregation rule rather than simply sending an obvious outlier. In other words, robust aggregation can reduce the influence of many straightforward malicious updates, but it does not prove that every participating update is honest or correct. The defense can weaken when an attacker understands the geometry or assumptions used by the aggregator, when the attack is carefully targeted, or when the proportion and coordination of Byzantine clients fall outside the defense's assumptions.

---

# 5. Targeted vs Untargeted Byzantine Attacks

## Untargeted attack

An untargeted attack attempts to damage the overall performance of the global model.

Example:

```text
Honest update  → +0.10
Byzantine      → -2.00
```

The attacker's goal is broadly to reduce model quality or prevent convergence.

## Targeted attack

A targeted attack tries to make the model behave incorrectly for a particular class of inputs, task, or behavior while potentially preserving much of its normal performance.

The distinction is:

```text
Untargeted → make the overall model worse

Targeted   → make the model fail in a particular way
```

---

# 6. Week 7 Preparation

The planned Week 7 comparison is:

```text
FedAvg
   ↓
Coordinate-wise Median
   ↓
Same federated learning setup
   ↓
Introduce a Byzantine client
   ↓
Compare model behavior
```

Key questions:

1. How much does one poisoned client damage FedAvg?
2. How does coordinate-wise Median behave under the same attack?
3. Does the defense preserve normal learning performance when clients are honest?
4. What happens when the malicious update becomes more carefully constructed?

---

# 7. Key Takeaways

- A single Byzantine client can strongly distort FedAvg when it submits an extreme update.
- In the five-client conceptual example, FedAvg changes the update from the honest average of `+0.0975` to `-0.322`.
- Coordinate-wise Median returns `+0.09`, much closer to the honest average.
- Krum selects `+0.09` in the simplified example because it belongs to the tight honest cluster.
- Extreme scaling alone does not defeat the median with only one Byzantine client; a more subtle update that lies within the honest cluster can influence the median.
- Robust aggregation reduces certain classes of attacks but does not guarantee complete protection against adaptive or targeted model poisoning.
- For Week 7, the chosen defense is coordinate-wise Median.

---

# References

Fang, M., Cao, X., Jia, J., & Gong, N. Z. *Local Model Poisoning Attacks to Byzantine-Robust Federated Learning*. USENIX Security Symposium / arXiv:1911.11815.
