# Week 5 — Federated Learning: FedAvg vs FedProx on Non-IID Data

## Overview

Week 5 moved the experiment from simply observing the effect of non-IID data to testing a method designed to reduce the resulting **client drift**. The central comparison was between **FedAvg** and **FedProx** under the same non-IID CIFAR-10 setting used previously.

The working idea from the Week 5 notes was straightforward: when clients have different local data distributions, their local models can move in different directions. FedProx addresses this by adding a proximal penalty that discourages a client from moving too far from the global model it received at the beginning of a round.

The proximal objective was:

$$
F_i(w) + \frac{\mu}{2}\|w-w_t\|^2
$$

where:

- $F_i(w)$ is the client's normal local training loss.
- $w$ is the client's current local model.
- $w_t$ is the global model received at the start of the round.
- $\mu$ controls the strength of the restriction.

In the Week 5 notes, $\mu=0$ represents no proximal restriction, while larger values apply a stronger constraint. The notes also emphasize that if $\mu$ becomes too large, it can restrict useful local learning.

This made Week 5 a practical test of an important question:

> **Can controlling local model drift improve federated learning under non-IID data?**

---

## Objectives

The main objectives for Week 5 were to:

1. Understand why FedAvg can struggle when client data distributions are heterogeneous.
2. Understand the role of the FedProx proximal term.
3. Implement the proximal term in the existing Flower/PyTorch application.
4. Compare FedAvg against FedProx with two different values of $\mu$.
5. Use the same non-IID Dirichlet $\alpha=0.5$ setup so that the comparison is fair.
6. Run the experiments for 15 communication rounds and compare the resulting accuracy curves.

---

## 1. From FedAvg to Client Drift

### What FedAvg does

In federated learning, each client trains locally and sends its updated model to the server. The server aggregates the client models to form a new global model, which is then sent back to the clients.

FedAvg works particularly well when the clients have reasonably similar data distributions. Under non-IID data, however, different clients can optimize toward different local objectives.

For example, one client may contain many images from some classes while another contains very different classes. After local training, the resulting model updates can point in different directions.

This is **client drift**.

The Week 5 notes describe client drift as local models moving away from one another because the clients are learning from different data distributions. Large differences between local models can make aggregation less stable and make global convergence more difficult.

---

## 2. What FedProx Changes

FedProx keeps the normal local learning objective but adds a distance penalty relative to the model received from the server.

The loss becomes:

$$
\text{FedProx Loss}
=
\text{Local Classification Loss}
+
\frac{\mu}{2}\|w-w_t\|^2
$$

The proximal term acts like a leash:

| $\mu$ | Interpretation |
|---:|---|
| 0 | No proximal restriction; behaves as the FedAvg local objective |
| 0.01 | Weak restriction on local drift |
| 0.1 | Stronger restriction on local drift |

The practical intuition is that clients should still learn from their own data, but their local models should remain closer to the global model when the proximal penalty is active.

---

## 3. Starting Point: Existing Week 3/4 Project

Rather than building a separate application, Week 5 extended the existing Flower/PyTorch project.

The project already contained:

- CIFAR-10 loading through Hugging Face Datasets.
- A CNN model in `task.py`.
- Dirichlet-based non-IID partitioning.
- Flower `ClientApp` and `ServerApp` architecture.
- Centralized global evaluation.
- CSV experiment logging.

The non-IID split remained:

```text
CIFAR-10
   ↓
Dirichlet partitioning
   ↓
α = 0.5
   ↓
5 simulated clients
```

Keeping this data pipeline unchanged was important because the goal was to compare algorithms under the same heterogeneity conditions rather than changing the dataset between experiments.

---

## 4. Important Implementation Change: Adapting the Provided Code

The Week 5 starter code used the older Flower `NumPyClient` API. The project, however, was already using Flower's newer `ClientApp`/`ServerApp` architecture.

Instead of replacing the application with the older example, the FedProx idea was translated into the existing architecture.

This resulted in relatively small but important changes.

### `task.py`

The training function was extended to accept:

```python
mu=0.0

global_model=None
```

The normal loss is calculated first:

```python
loss = criterion(outputs, labels)
```

Then, when FedProx is enabled, the distance penalty is added:

```python
if mu > 0.0 and global_model is not None:
    proximal_term = 0.0

    for local_param, global_param in zip(
        net.parameters(),
        global_model.parameters(),
    ):
        proximal_term += (
            (local_param - global_param.detach()) ** 2
        ).sum()

    loss = loss + (mu / 2.0) * proximal_term
```

This was the key algorithmic modification.

### `client_app.py`

The client now keeps a copy of the global model before local training begins:

```python
global_model = Net()
global_model.load_state_dict(
    msg.content["arrays"].to_torch_state_dict()
)
```

The local model is trained while the preserved global model is used as the reference point for the proximal term.

This follows the Week 5 learning sequence:

```text
Receive global model
       ↓
Keep a reference copy
       ↓
Train locally
       ↓
Measure distance from original global model
       ↓
Apply proximal penalty
```

### `server_app.py`

The previous Week 4 experiment used `FedAdagrad`. For Week 5, the server strategy was changed to `FedAvg` so that the baseline matched the assignment.

The strategy therefore became:

```python
strategy = FedAvg(
    fraction_evaluate=fraction_evaluate,
)
```

The server continued to perform centralized evaluation and write round-by-round results to CSV.

### `pyproject.toml`

The main experimental configuration was changed to:

```toml
num-server-rounds = 15
fraction-evaluate = 1.0
local-epochs = 2
learning-rate = 0.001
batch-size = 32
save-model = false
```

The value of `mu` and the output filename were then changed between experiments.

---

## 5. Experimental Configuration

Three experiments were performed.

| Experiment | Method | $\mu$ | Output |
|---|---|---:|---|
| 1 | FedAvg baseline | 0.0 | `fedavg_noniid.csv` |
| 2 | FedProx | 0.01 | `fedprox_mu001.csv` |
| 3 | FedProx | 0.1 | `fedprox_mu01.csv` |

All three experiments used:

| Setting | Value |
|---|---|
| Dataset | CIFAR-10 |
| Partitioning | Dirichlet non-IID |
| Dirichlet $\alpha$ | 0.5 |
| Simulated clients | 5 |
| Communication rounds | 15 |
| Local epochs | 2 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Evaluation | Centralized CIFAR-10 test set |

---

## 6. Debugging and Experimental Experience

### Flower initially used only 2 simulated clients

The first Week 5 runs exposed a configuration issue rather than a model issue.

The installed Flower environment reported that the default number of simulated SuperNodes was 2. The terminal repeatedly showed:

```text
Federation `@none/default` (2 simulated SuperNodes)
configure_train: Sampled 2 nodes (out of 2)
```

This did not match the required five-client experiment.

The federation was then explicitly configured with five SuperNodes using:

```powershell
flwr federation simulation-config @none/default local --num-supernodes 5
```

After this, the application reported:

```text
Federation `@none/default` (5 simulated SuperNodes)
configure_train: Sampled 5 nodes (out of 5)
```

The FedAvg baseline was rerun after this correction so the final baseline used the intended five-client configuration.

### Hugging Face connectivity warnings

CIFAR-10 was loaded using Hugging Face Datasets. During some runs, the clients produced warnings about unauthenticated Hub requests, and one later round also showed temporary network/DNS errors while requesting dataset metadata.

Despite those messages, Flower completed the relevant aggregation with five results and the experiments finished successfully.

This reinforced an important practical lesson from the earlier weeks: federated learning experiments depend not only on the learning algorithm but also on the reliability of the surrounding software and data infrastructure.

### Experiment reproducibility

The three runs used the same declared configuration, data partitioning procedure, model, optimizer, and number of rounds. However, the PyTorch model initialization was not explicitly seeded in the current implementation.

Therefore, the comparison should be interpreted as an empirical comparison under the experiment configuration rather than as a fully controlled repeated-trials benchmark. This is especially important when the differences between methods are relatively small.

---

## 7. Results

### Final performance

| Method | Round 15 Accuracy | Best Accuracy | Best Round | Final Loss |
|---|---:|---:|---:|---:|
| FedAvg ($\mu=0$) | **59.14%** | **59.14%** | 15 | 1.2613 |
| FedProx ($\mu=0.01$) | **58.96%** | **58.96%** | 15 | 1.2404 |
| FedProx ($\mu=0.1$) | **60.26%** | **60.26%** | 15 | **1.1685** |

The strongest final result came from **FedProx with $\mu=0.1$**.

It finished at **60.26% accuracy**, compared with **59.14% for FedAvg**, giving a difference of:

$$
60.26 - 59.14 = 1.12
$$

percentage points.

The $\mu=0.01$ configuration, however, finished slightly below the FedAvg baseline at **58.96%**.

This means the experiment does **not** support the simplistic statement that any FedProx setting automatically performs better than FedAvg.

Instead, the results suggest that the effectiveness of the proximal constraint depends on the chosen value of $\mu$.

---

## 8. Round-by-Round Accuracy

### FedAvg ($\mu=0$)

| Round | Accuracy |
|---:|---:|
| 0 | 10.00% |
| 1 | 19.55% |
| 2 | 34.10% |
| 3 | 43.47% |
| 4 | 47.81% |
| 5 | 51.37% |
| 6 | 54.24% |
| 7 | 53.36% |
| 8 | 54.89% |
| 9 | 56.52% |
| 10 | 57.46% |
| 11 | 57.45% |
| 12 | 58.08% |
| 13 | 58.90% |
| 14 | 57.85% |
| 15 | **59.14%** |

FedAvg improved rapidly during the early rounds and continued to improve overall, although there were several small drops at rounds 7, 11, and 14.

### FedProx ($\mu=0.01$)

| Round | Accuracy |
|---:|---:|
| 0 | 10.44% |
| 1 | 28.37% |
| 2 | 34.41% |
| 3 | 44.00% |
| 4 | 49.11% |
| 5 | 52.81% |
| 6 | 52.73% |
| 7 | 53.65% |
| 8 | 54.69% |
| 9 | 55.80% |
| 10 | 57.15% |
| 11 | 57.27% |
| 12 | 58.31% |
| 13 | 58.46% |
| 14 | 58.37% |
| 15 | **58.96%** |

FedProx with $\mu=0.01$ showed strong early improvement and remained close to FedAvg through the later rounds. However, it did not finish above the FedAvg baseline.

### FedProx ($\mu=0.1$)

| Round | Accuracy |
|---:|---:|
| 0 | 10.00% |
| 1 | 27.35% |
| 2 | 37.71% |
| 3 | 41.59% |
| 4 | 47.78% |
| 5 | 50.43% |
| 6 | 52.54% |
| 7 | 52.12% |
| 8 | 55.96% |
| 9 | 55.44% |
| 10 | 57.57% |
| 11 | 58.04% |
| 12 | 59.07% |
| 13 | 58.95% |
| 14 | 59.14% |
| 15 | **60.26%** |

FedProx with $\mu=0.1$ became the strongest configuration in the later rounds and finished at the highest accuracy.

---

## 9. Accuracy Curve

![FedAvg vs FedProx comparison](fedprox_comparison.png)

The three curves follow broadly similar learning trajectories: accuracy rises rapidly during the early communication rounds and then begins to fluctuate around the mid-to-high 50% range.

The most important visual difference is that **FedProx with $\mu=0.1$ becomes increasingly competitive later in training and finishes highest**.

The curves also show that the methods do not separate dramatically from one another during every round. The main difference becomes clearer toward the end of the 15-round experiment.

---

## 10. Interpreting the Results Through Client Drift

The Week 5 hypothesis was that non-IID local data can cause clients to move their models in different directions. FedProx attempts to reduce this effect by penalizing large deviations from the received global model.

The results provide **partial empirical support** for this idea.

FedProx with $\mu=0.1$ achieved the best final accuracy and the lowest final loss among the three configurations:

```text
FedAvg        → 59.14% accuracy, loss 1.2613
FedProx .01   → 58.96% accuracy, loss 1.2404
FedProx .1    → 60.26% accuracy, loss 1.1685
```

The stronger proximal constraint therefore produced the best result in this experiment.

However, $\mu=0.01$ did not outperform FedAvg. This matters because it shows that the proximal term should not be viewed as an automatic improvement simply because it exists. Its benefit depends on how strongly it constrains local optimization.

A useful interpretation is:

- **$\mu=0$:** clients are free to move according to their own local objectives.
- **$\mu=0.01$:** the restriction may have been too weak to produce a measurable advantage in this run.
- **$\mu=0.1$:** the stronger restriction appears to have provided a better balance between local learning and staying close to the global model.

This interpretation remains an inference from the observed experiment rather than proof that $\mu=0.1$ is universally optimal.

---

## 11. What I Learned This Week

### 1. A small code change can represent a major algorithmic change

The central FedProx modification was only a few lines of code, but those lines changed the optimization objective of every participating client.

### 2. The global model must be preserved before local training

FedProx needs a reference point. The client cannot compare its final local model to the global model after the local model has already changed. The original received model must be kept as the reference throughout local training.

### 3. Hyperparameters can change the conclusion

The two FedProx settings behaved differently. $\mu=0.01$ finished below FedAvg, while $\mu=0.1$ finished above it.

That makes hyperparameter selection part of the federated learning problem rather than an implementation detail.

### 4. Experimental configuration matters as much as the model code

The initial 2-SuperNode configuration meant the experiment was not actually matching the intended five-client design. Catching that issue before accepting the first results was important.

### 5. Federated learning continues to look like a distributed-systems problem

Dataset access, simulation configuration, client participation, runtime environments, and aggregation all affect the experiment. The learning algorithm is only one part of the complete system.

---

## 12. Limitations

This experiment has several limitations that should be kept visible in the final interpretation.

### Single run per configuration

Each method was run once. Because model initialization and other stochastic processes were not fully seeded and repeated, the observed differences should not be treated as statistically conclusive.

### Small simulated federation

The experiment used five simulated clients. This is appropriate for the assignment and makes the experiment practical, but it is much smaller than a real federated deployment.

### One heterogeneity level

Only Dirichlet $\alpha=0.5$ was tested. Different levels of heterogeneity could change which algorithm performs best.

### Limited $\mu$ search

Only $\mu=0.01$ and $\mu=0.1$ were tested. A broader sweep could reveal whether the best value lies between them or outside this range.

---

## 13. Conclusion

Week 5 extended the non-IID experiments from simply observing client drift to testing a method designed to control it.

FedAvg was used as the baseline, while FedProx introduced a proximal penalty controlled by $\mu$. All three experiments used the same CIFAR-10 non-IID Dirichlet $\alpha=0.5$ setting, five simulated clients, two local epochs, Adam with a learning rate of 0.001, and 15 communication rounds.

The final results were:

```text
FedAvg (μ=0)       → 59.14%
FedProx (μ=0.01)   → 58.96%
FedProx (μ=0.1)    → 60.26%
```

FedProx with $\mu=0.1$ achieved the best final performance, beating the FedAvg baseline by **1.12 percentage points** and also achieving the lowest final loss.

The result supports the idea that constraining local model movement can help under non-IID data, but it also demonstrates that **the strength of the proximal constraint matters**. A weak proximal penalty did not outperform the baseline in this experiment, while the stronger penalty did.

The most important takeaway from Week 5 is therefore not simply that "FedProx wins." It is that **FedProx provides a mechanism for controlling client drift, and its usefulness depends on choosing an appropriate level of regularization for the degree of heterogeneity in the federation.**

---

## 14. Open Question

> **How does the optimal value of $\mu$ change as the level of non-IID heterogeneity changes, and is there a systematic way to choose $\mu$ instead of tuning it manually?**

---

## 15. Files Produced

- `fedavg_noniid.csv` — FedAvg baseline results.
- `fedprox_mu001.csv` — FedProx with $\mu=0.01$ results.
- `fedprox_mu01.csv` — FedProx with $\mu=0.1$ results.
- `fedprox_comparison.png` — accuracy comparison plot.

