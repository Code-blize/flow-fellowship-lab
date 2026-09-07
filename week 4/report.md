```text
WEEK4_REPORT.md
```

````markdown
# Week 4: Client Dropout and Failure Modes in Federated Learning

## Overview

Week 4 focused on understanding Federated Learning as not only a machine learning problem but also a distributed systems problem.

In previous weeks, the experiments focused on concepts such as IID and Non-IID data distributions, optimizer behaviour, convergence, and global model performance. This week introduced another important challenge in real-world Federated Learning systems: **client unreliability**.

The main objective was to simulate **client dropout** during federated training and study how unreliable clients can affect the training process and convergence of the global model.

In real-world Federated Learning, selected clients do not always successfully complete a training round. Devices can disconnect because of poor internet connectivity, low battery, application shutdowns, hardware limitations, or other unexpected interruptions.

This week's work extended the existing Flower and PyTorch Federated Learning project by introducing configurable client dropout.

---

# Objectives

The objectives for Week 4 were to:

- Simulate client dropout during Federated Learning training.
- Run experiments with different dropout rates.
- Compare model behaviour under different levels of client availability.
- Record global accuracy and loss across communication rounds.
- Generate degradation curves showing the effect of client dropout.
- Understand how system reliability affects Federated Learning.
- Study common Federated Learning failure modes and connect them to real-world scenarios.

The planned experiments included:

| Experiment | Dropout Rate | Results File |
|---|---:|---|
| Baseline | 0% | `dropout_0.csv` |
| Moderate Dropout | 30% | `dropout_30.csv` |
| Severe Dropout | 60% | `dropout_60.csv` |

---

# Project Structure

The project continued using the Flower PyTorch quickstart structure:

```text
quickstart-pytorch/
│
├── pytorchexample/
│   ├── __init__.py
│   ├── client_app.py
│   ├── server_app.py
│   └── task.py
│
├── pyproject.toml
├── week 4.py
├── dropout_0.csv
├── dropout_30.csv
└── dropout_60.csv
````

The main files involved in the Week 4 experiment were:

* `client_app.py` — client-side training and dropout simulation.
* `server_app.py` — server-side Federated Learning strategy and evaluation.
* `pyproject.toml` — experiment configuration.
* `week 4.py` — loading results and plotting the dropout comparison graph.

---

# Implementing Client Dropout

The client dropout experiment was implemented inside the Flower `ClientApp`.

A dropout rate was added to the experiment configuration in `pyproject.toml`.

For example:

```toml
# Week 4: Client Dropout Experiment
dropout-rate = 0.6
results-file = "dropout_60.csv"
```

The client application accessed the dropout rate using:

```python
dropout_rate = context.run_config["dropout-rate"]
```

A random probability was then used to determine whether a client would participate in a particular training round:

```python
if random.random() < dropout_rate:
```

If the condition was true, the client simulated dropout instead of performing normal local training.

This allowed the experiment to represent unreliable clients in a Federated Learning environment.

---

# Normal Client Participation

When a client did not drop out, it followed the normal Federated Learning workflow:

```text
Server sends global model
        ↓
Client receives global parameters
        ↓
Client trains on local data
        ↓
Client sends updated parameters
        ↓
Server aggregates client updates
```

The client loaded its assigned data partition and trained the local PyTorch model before returning its updated parameters.

---

# Simulated Client Dropout

When a client was selected to drop out, it did not perform local training.

Conceptually, the process looked like:

```text
Server selects client
        ↓
Client becomes unavailable
        ↓
Client does not perform local training
        ↓
Server receives fewer useful contributions
```

As more clients become unavailable, fewer local datasets contribute to the global model during each communication round.

This can affect the stability and convergence of Federated Learning.

---

# Configuring the Experiments

The experiments were designed to reuse the same Flower application while changing the dropout rate and output filename through `pyproject.toml`.

## Experiment 1: No Dropout

```toml
dropout-rate = 0.0
results-file = "dropout_0.csv"
```

This experiment serves as the baseline.

All selected clients are expected to participate normally.

---

## Experiment 2: Moderate Dropout

```toml
dropout-rate = 0.3
results-file = "dropout_30.csv"
```

This simulates a Federated Learning environment where approximately 30% of client participation attempts may fail.

---

## Experiment 3: Severe Dropout

```toml
dropout-rate = 0.6
results-file = "dropout_60.csv"
```

This represents a significantly less reliable environment where a large proportion of clients may fail to participate during training.

---

# Recording Experiment Results

The server application was configured to record global evaluation metrics after each communication round.

The recorded metrics include:

* Communication round
* Global accuracy
* Global loss

The results are saved in CSV format:

```csv
round,accuracy,loss
1,...
2,...
3,...
```

At the beginning of each experiment, the server creates a new results file:

```python
with open(RESULTS_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["round", "accuracy", "loss"])
```

After each global evaluation, the results are appended to the CSV file.

This makes it possible to compare different dropout experiments separately.

---

# Challenge Encountered: Aggregation Failure

One of the most important experiences during Week 4 was debugging an aggregation problem caused by the initial dropout implementation.

The first approach simulated a dropped client by returning:

```python
"num-examples": 0
```

The intention was to indicate that the client had not contributed any training data during that communication round.

However, during the simulation, Flower produced the following error:

```text
ZeroDivisionError: division by zero
```

The error occurred during the aggregation of client updates.

This happened because the aggregation process depends on client weights. When the effective total weight of the participating updates became zero, Flower could not complete the weighted aggregation.

This was an important learning experience because it demonstrated that implementing a failure simulation requires understanding not only the Federated Learning concept but also how the framework handles client responses internally.

---

# Debugging the Dropout Simulation

The aggregation error showed that simply returning zero training examples for dropped clients could cause the Federated Learning strategy to fail.

The simulation therefore required changes to ensure that the server could continue aggregating client responses while still representing unreliable participation.

This debugging process reinforced an important lesson:

> Simulating failures in a Federated Learning system requires careful consideration of how the server interprets client responses and performs aggregation.

The problem was not related to VS Code, the project folder structure, or the CSV logging system.

The issue was specifically related to the interaction between the simulated dropout behaviour and Flower's aggregation mechanism.

---

# Key Lessons from Debugging

## 1. Federated Learning Failures Can Affect the Entire System

The original goal was to measure how dropout affects model accuracy and convergence.

However, the experiment demonstrated that failures can also affect whether the Federated Learning process can run successfully at all.

Poorly handled client failures can result in:

* aggregation errors,
* failed communication rounds,
* insufficient client contributions,
* invalid aggregation weights,
* and simulation crashes.

---

## 2. Framework Behaviour Matters

Federated Learning concepts can appear straightforward in theory, but practical implementations depend heavily on the behaviour of the framework being used.

Flower has specific expectations for how client updates are returned and aggregated.

When simulating failures, it is important to understand:

* how clients report participation,
* how client updates are weighted,
* how aggregation is performed,
* and what happens when clients fail to contribute.

---

## 3. Federated Learning Is Also a Distributed Systems Problem

Week 4 reinforced the idea that Federated Learning is not only about training machine learning models.

A Federated Learning system must also deal with:

* unreliable clients,
* network failures,
* hardware differences,
* slow devices,
* incomplete communication,
* and unexpected system interruptions.

This means Federated Learning combines concepts from:

* Machine Learning
* Distributed Systems
* Networking
* Systems Reliability

---

# Plotting the Dropout Degradation Curves

After completing the experiments, the results from the three dropout levels can be compared visually.

The plotting script loads the three CSV files:

```python
r0, a0 = load_log("dropout_0.csv")
r30, a30 = load_log("dropout_30.csv")
r60, a60 = load_log("dropout_60.csv")
```

The accuracy values are then plotted against communication rounds.

The visualization compares:

* No dropout
* 30% dropout
* 60% dropout

The goal of the graph is to answer the following question:

> How does increasing client unreliability affect Federated Learning convergence?

The expected general behaviour is that higher dropout rates may result in:

* slower convergence,
* more unstable training,
* greater variation between rounds,
* and potentially lower final accuracy.

However, the exact results can vary depending on:

* the number of clients,
* the number of selected clients per round,
* the dataset,
* local training,
* data distribution,
* aggregation strategy,
* and randomness in client dropout.

---

# Federated Learning Failure Mode Taxonomy

Client dropout is only one of several failure modes that can affect a Federated Learning system.

## 1. Client Dropout

**What it is:**
Clients become unavailable before successfully completing a training round.

**Effect on training:**
The number of useful client updates decreases, potentially slowing convergence and increasing instability in the global model.

**Real-world scenario:**
Cross-device Federated Learning on smartphones, where clients may disconnect because of poor connectivity, battery saver settings, lock screens, or application shutdowns.

---

## 2. Stragglers

**What it is:**
Some clients take significantly longer than others to complete local training.

**Effect on training:**
Slow clients can delay communication rounds and increase the total training time.

**Real-world scenario:**
A mobile Federated Learning deployment involving devices with very different hardware capabilities, where older phones train much more slowly than newer devices.

---

## 3. Communication Failure

**What it is:**
Clients cannot successfully communicate with the Federated Learning server.

**Effect on training:**
Model updates may be delayed, interrupted, or completely lost.

**Real-world scenario:**
Federated Learning deployments operating across regions with unreliable internet or unstable mobile network infrastructure.

---

## 4. Non-IID Data

**What it is:**
Different clients possess data with significantly different statistical distributions.

**Effect on training:**
Local models may move in different directions during training, making global convergence slower and less stable.

**Real-world scenario:**
A hospital Federated Learning network where different hospitals serve populations with different medical conditions and patient characteristics.

---

## 5. Resource Constraints

**What it is:**
Clients have limited computational power, memory, storage, or battery capacity.

**Effect on training:**
Clients may train slowly, fail during local computation, or participate inconsistently.

**Real-world scenario:**
Federated Learning across low-end mobile devices with limited processing power and battery capacity.

---

## 6. Malicious or Unreliable Updates

**What it is:**
A client sends incorrect, corrupted, manipulated, or unreliable model updates.

**Effect on training:**
The global model may converge poorly or become vulnerable to attacks.

**Real-world scenario:**
A Federated Learning system involving volunteer computing devices where the server cannot fully trust every participating contributor.

---

# Which Failure Mode Is Hardest to Defend Against?

Among the failure modes studied, malicious or unreliable client updates are likely among the most difficult to defend against.

Client dropout and communication failures are often visible because a client simply fails to respond.

Malicious clients are more difficult because they may appear to behave normally.

A malicious client can:

* receive the global model,
* participate in the communication round,
* return a model update,
* and still intentionally attempt to damage the global model.

This makes malicious behaviour more difficult to detect than simple client dropout.

---

# Key Takeaways

The major lessons from Week 4 include:

* Federated Learning clients are not always reliable.
* Client availability can affect both model convergence and system stability.
* Failure simulation must be implemented carefully.
* Framework-specific behaviour matters when building Federated Learning experiments.
* Client responses must remain compatible with the server's aggregation process.
* Configuration files make experiments easier to reproduce.
* CSV logging allows experiments to be compared across multiple runs.
* Federated Learning must account for both machine learning and distributed systems challenges.

Perhaps the most important lesson from this week was:

> Federated Learning is not only a machine learning problem. It is also a distributed systems problem.

A successful Federated Learning system must consider not only:

* model accuracy,
* datasets,
* optimization,
* and convergence,

but also:

* client reliability,
* communication,
* system failures,
* hardware limitations,
* and fault tolerance.

---

# Technologies Used

* Python
* PyTorch
* Flower Federated Learning Framework
* NumPy
* Matplotlib
* CSV
* VS Code
* GitHub

---

# Conclusion

Week 4 focused on understanding how client dropout and system failures can affect Federated Learning.

The existing Flower and PyTorch application was extended by introducing configurable client dropout rates and separate result files for different experiments.

During implementation, an aggregation failure caused by zero-weight client contributions highlighted an important practical challenge. The debugging process provided valuable insight into how Flower handles client updates and how failure simulations interact with server-side aggregation.

This week's work demonstrated that real-world Federated Learning requires more than simply training machine learning models. It requires designing systems that can operate reliably even when clients disconnect, respond slowly, experience communication failures, or behave unpredictably.

The next step is to complete the experiments for the three dropout conditions, generate the degradation curves, and analyse how increasing client unreliability affects the convergence and performance of the global model.

---

# Week 4 Progress

This week's work builds on previous experiments involving:

* IID Federated Learning
* Non-IID data distribution using Dirichlet partitioning
* Optimizer comparisons
* FedAdagrad experiments
* Global accuracy and loss tracking
* CSV experiment logging
* Federated Learning convergence visualization

Week 4 extends this work by introducing **system reliability and client failure simulation** into the Federated Learning workflow.



And yes, when you're ready, I'll also guide you through the **Git commit and push process step by step**. 
