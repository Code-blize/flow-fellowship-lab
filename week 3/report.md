# Week 3 Report
# Federated Learning Under Non-IID Data Distributions

**Author:** Obasi-Uzoma Blessing

**Program:** FLOW Research Fellowship

**Week:** 3

**Framework:** Flower Federated Learning Framework

**Dataset:** CIFAR-10

**Language:** Python

---

# Abstract

Week 3 introduced one of the most important challenges in Federated Learning: training under **non-independent and identically distributed (Non-IID)** data.

Previous experiments assumed that every client owned data drawn from the same probability distribution. While this assumption simplifies experimentation, it does not represent real-world federated systems. Smartphones, hospitals, banks, and IoT devices naturally collect different kinds of data based on their users and environments. Consequently, each client learns from a different local distribution.

The goal of this week's exercise was to simulate this realistic setting using the CIFAR-10 dataset and Flower. Instead of equally distributing samples among clients, the dataset was partitioned using a **Dirichlet distribution**, creating heterogeneous client datasets. The effect of this heterogeneity on global convergence was then observed by comparing IID and Non-IID training behaviour.

During implementation, several engineering challenges were encountered, including dependency incompatibilities between Flower versions, runtime configuration errors, package management issues, optimizer migration from SGD to Adam, and modifications to the data loading pipeline. Solving these issues provided practical experience in debugging distributed machine learning systems.

Finally, experimental results demonstrated the expected behaviour of Federated Learning under heterogeneous data: slower convergence, reduced global accuracy, larger oscillations during optimization, and evidence of client drift.

---

# Table of Contents

1. Introduction
2. Objectives
3. Understanding Data Distribution
4. IID vs Non-IID Data
5. Why Real Federated Learning is Non-IID
6. Client Drift
7. Dirichlet Distribution
8. Implementing Non-IID Partitioning
9. Flower Client Modifications
10. Flower Server Modifications
11. Experimental Setup
12. Results
13. Graph Analysis
14. Debugging Journey
15. Lessons Learned
16. Conclusion
17. References

---

# Introduction

Federated Learning enables multiple devices to collaboratively train a machine learning model without sharing their raw data. Instead of sending datasets to a central server, each client trains locally and only communicates model updates.

The standard Federated Averaging (FedAvg) algorithm assumes that local client updates collectively represent the overall population. This assumption generally holds when client datasets are independent and identically distributed (IID).

However, real-world data rarely satisfies this assumption.

Different users generate different behaviours.

Different hospitals diagnose different diseases.

Different mobile devices contain different applications.

Different regions speak different languages.

Therefore, practical Federated Learning systems must operate under **heterogeneous client data**, commonly known as **Non-IID data**.

Week 3 focused on understanding this challenge and experimentally observing how non-identical data distributions affect global model convergence.

---

# Objectives

The objectives for Week 3 were:

- Understand the concept of IID and Non-IID datasets.
- Learn why heterogeneous data is a major challenge in Federated Learning.
- Implement a realistic Non-IID partition using the Dirichlet distribution.
- Observe the effect of heterogeneous clients on training convergence.
- Compare experimental results between IID and Non-IID training.
- Understand the phenomenon known as client drift.
- Gain practical experience debugging a Flower Federated Learning application.

---

# Understanding Data Distribution

Machine learning models learn patterns from data.

The quality of learning depends not only on the quantity of data but also on **how the data is distributed**.

Suppose ten students are learning to recognize animals.

If every student studies exactly the same mixture of dogs, cats, birds, horses, and fish, then every student gains similar knowledge.

When they later combine what they learned, their combined knowledge closely approximates what would have been learned from the complete dataset.

This represents an IID setting.

Now imagine a different situation.

Student 1 studies only cats.

Student 2 studies only dogs.

Student 3 studies only birds.

Student 4 studies only horses.

Each student becomes an expert in only one class.

When they combine their knowledge, conflicts naturally arise because each local model was optimized for different objectives.

This represents a Non-IID setting.

Federated Learning in practice almost always resembles the second scenario.

---

# Independent and Identically Distributed (IID)

IID stands for **Independent and Identically Distributed**.

Two conditions must hold:

### Independent

Each sample does not depend on another sample.

Learning one image provides no information about another image.

### Identically Distributed

Every client receives data sampled from the same probability distribution.

For example:

Client A

- 10% airplanes
- 10% cars
- 10% birds
- ...
- 10% trucks

Client B

- 10% airplanes
- 10% cars
- 10% birds
- ...
- 10% trucks

Client C

- Similar proportions

Every client effectively sees the same world.

Training therefore becomes stable and global convergence is usually fast.

---

# Non-IID Data

Non-IID data violates one or both IID assumptions.

Different clients observe different distributions.

For example:

Client A

- Mostly airplanes

Client B

- Mostly ships

Client C

- Mostly trucks

Client D

- Mostly birds

Although every client trains correctly on its own data, the local objectives become different.

The server receives updates pointing toward different optimization directions.

Consequently:

- convergence becomes slower
- accuracy improves more gradually
- optimization oscillates
- communication rounds become less efficient

This phenomenon is one of the central research problems in Federated Learning today.

---

# Client Drift

One of the biggest challenges in Federated Learning is **client drift**.

Client drift occurs when different clients optimize their models toward different objectives because they each possess different data distributions.

Consider a simple example involving image classification.

Suppose three clients participate in federated training.

- Client A contains mostly images of airplanes.
- Client B contains mostly images of birds.
- Client C contains mostly images of trucks.

Each client performs local training using only its own data.

Because their datasets differ significantly, each model update moves the global model in a different direction.

Instead of cooperating perfectly, the clients unintentionally compete.

When the server averages these conflicting model updates, the resulting global model improves much more slowly than it would under IID training.

This phenomenon is known as **client drift**.

Client drift becomes more severe as data heterogeneity increases.

Consequently, federated optimization under Non-IID data generally requires:

- more communication rounds,
- improved optimization algorithms,
- adaptive aggregation methods,
- or personalization strategies.

Understanding client drift is fundamental because most modern Federated Learning research attempts to reduce or compensate for its effects.

---

# Dirichlet Distribution

To simulate realistic heterogeneous datasets, this project used the **Dirichlet distribution**.

The Dirichlet distribution is widely used in Federated Learning literature because it provides a controllable way to distribute class labels unevenly across clients.

Rather than assigning equal numbers of each class to every client, the Dirichlet distribution randomly generates proportions for each class.

For example, suppose images of airplanes must be distributed across five clients.

Instead of assigning exactly:

```
20%
20%
20%
20%
20%
```

the Dirichlet distribution may generate

```
60%
20%
10%
5%
5%
```

or

```
90%
5%
2%
2%
1%
```

depending on the value of **α (alpha)**.

This creates realistic client specialization.

Some clients naturally become dominated by certain classes.

Others receive very little data for those classes.

This behaviour closely resembles real-world Federated Learning applications.

---

# Understanding Alpha (α)

The parameter α controls how heterogeneous the data becomes.

Large α values produce datasets that resemble IID training.

Small α values create highly heterogeneous client datasets.

For example:

| Alpha | Distribution |
|--------|--------------|
| α = 10 | Nearly IID |
| α = 1 | Moderately heterogeneous |
| α = 0.5 | Strongly Non-IID |
| α = 0.1 | Extremely Non-IID |

For this experiment,

```
α = 0.5
```

was selected.

This value produces sufficient heterogeneity while still allowing the global model to converge within a reasonable number of communication rounds.

---

# Implementing Non-IID Partitioning

Unlike Week 2, where every client received randomly sampled data, Week 3 required manually partitioning the CIFAR-10 dataset.

A new helper function named `dirichlet_split()` was introduced.

The function begins by extracting every class label from the dataset.

```python
labels = np.array(dataset["label"])
```

The number of unique classes is then determined.

```python
num_classes = len(np.unique(labels))
```

An empty list of client indices is created.

```python
client_indices = [[] for _ in range(num_clients)]
```

The algorithm then processes one class at a time.

For every class:

1. Find all samples belonging to that class.
2. Shuffle them randomly.
3. Generate client proportions using the Dirichlet distribution.
4. Split the class according to those proportions.
5. Assign each portion to the corresponding client.

The critical step is

```python
proportions = np.random.dirichlet(
    np.repeat(alpha, num_clients)
)
```

This single line replaces uniform sampling with probabilistic class allocation.

Every execution produces a realistic heterogeneous distribution while remaining reproducible through the fixed random seed.

---

# Modified Data Loading Pipeline

The `load_data()` function was also modified.

Instead of allowing Flower to partition the dataset automatically, the complete CIFAR-10 training dataset is first loaded.

```python
dataset = load_dataset(
    "uoft-cs/cifar10",
    split="train"
)
```

The dataset is then partitioned using the custom Dirichlet function.

```python
client_indices = dirichlet_split(
    dataset,
    num_partitions,
    alpha=0.5
)
```

Each client retrieves only its assigned subset.

```python
partition = dataset.select(
    client_indices[partition_id]
)
```

Finally, the selected data is divided into training and validation subsets.

```python
partition = partition.train_test_split(
    test_size=0.2,
    seed=42
)
```

The resulting training and validation sets are converted into PyTorch DataLoaders before being returned to the Flower ClientApp.

This modification ensures that every client trains using a unique local data distribution while preserving compatibility with the existing Flower training pipeline.

---

# Why This Change Matters

The transition from IID to Non-IID fundamentally changes how Federated Learning behaves.

Under IID training:

- every client learns approximately the same feature distribution;
- model updates are consistent;
- aggregation produces steady convergence.

Under Non-IID training:

- every client specializes in different classes;
- local gradients diverge;
- aggregation becomes more difficult;
- convergence slows;
- communication efficiency decreases.

Although the model still learns, significantly more rounds are usually required before reaching comparable performance.

This experiment provides a practical demonstration of why handling heterogeneous data remains one of the largest challenges in Federated Learning research.

---

# Implementation and Engineering Journey

Unlike the previous week's implementation, this week's objective required modifying the existing Flower Quickstart project to support heterogeneous client datasets. This meant extending the default data loading pipeline, modifying how clients received data, and recording experiment results for later visualization.

Although the conceptual goal was straightforward, implementing the changes introduced several engineering challenges that required careful debugging and incremental testing.

This section documents the complete implementation process and the lessons learned throughout development.

---

# Modifying the Data Pipeline

The original Flower Quickstart application automatically partitions the CIFAR-10 dataset using the built-in `IidPartitioner`.

```python
partitioner = IidPartitioner(
    num_partitions=num_partitions
)
```

While this approach is suitable for introductory Federated Learning experiments, it assumes every client receives approximately the same class distribution.

For Week 3, this behaviour had to be replaced with a custom Non-IID partitioning strategy.

The first step was implementing a new helper function named `dirichlet_split()` inside `task.py`.

Instead of allowing Flower to partition the dataset, the complete training dataset was first loaded.

```python
dataset = load_dataset(
    "uoft-cs/cifar10",
    split="train"
)
```

Each class label was then extracted and grouped.

The Dirichlet distribution was used to determine what proportion of each class should be assigned to every client.

The resulting indices were stored before constructing client-specific datasets.

This modification completely replaced the default IID sampling process while remaining compatible with Flower's ClientApp interface.

---

# Updating the Data Loader

Once the client indices had been generated, the `load_data()` function also required modification.

Instead of calling Flower's partition loader, each client manually selected only the samples assigned to it.

```python
partition = dataset.select(
    client_indices[partition_id]
)
```

The selected dataset was then divided into local training and validation subsets.

```python
partition = partition.train_test_split(
    test_size=0.2,
    seed=42
)
```

Finally, PyTorch DataLoaders were created for local training.

This allowed every simulated client to train on its own heterogeneous dataset while preserving the remainder of the Flower pipeline.

---

# Client Application

The ClientApp required only minimal changes because Flower's architecture already separates communication from model training.

Each client continued to perform the following sequence:

1. Receive the latest global model.
2. Load its own local dataset.
3. Train locally.
4. Return updated model parameters.
5. Perform local evaluation.

Because the new `load_data()` function preserved the original interface, the remaining client logic remained unchanged.

This demonstrated one of Flower's strengths: data loading strategies can be modified without changing the overall federated workflow.

---

# Server Application

The server application was extended to record evaluation metrics after every communication round.

Instead of simply printing evaluation results to the terminal, the server now saved each round's accuracy and loss into a CSV file.

```python
with open(
    "noniid_results.csv",
    "a",
    newline=""
) as f:
```

Each evaluation appended three values.

- Communication round
- Global accuracy
- Global loss

These CSV files later became the input for plotting convergence curves.

Logging intermediate results also made it easier to compare multiple experiments performed under different configurations.

---

# Visualizing Training Progress

After completing the federated simulations, the recorded CSV files were loaded using Pandas.

Separate plots were generated for

- Global Accuracy
- Global Loss

Matplotlib was used to visualize convergence over communication rounds.

These plots provided a much clearer understanding of how heterogeneous data influences Federated Learning than numerical logs alone.

Rather than simply reading accuracy values from the terminal, the convergence behaviour became immediately visible.

---

# Switching Optimizers

During experimentation, both SGD and Adam optimizers were evaluated.

The original Flower Quickstart implementation used Stochastic Gradient Descent.

```python
optimizer = torch.optim.SGD(
    net.parameters(),
    lr=lr,
    momentum=0.9
)
```

To explore different optimization behaviour, the optimizer was replaced with Adam.

Initially, the implementation incorrectly attempted to reuse the momentum parameter.

```python
optimizer = torch.optim.Adam(
    net.parameters(),
    lr=lr,
    momentum=0.9
)
```

This produced an error because Adam does not accept a momentum argument.

Instead, Adam maintains exponential moving averages using the parameter

```python
betas=(0.9, 0.999)
```

The corrected implementation became

```python
optimizer = torch.optim.Adam(
    net.parameters(),
    lr=lr,
    betas=(0.9, 0.999)
)
```

This debugging process reinforced an important lesson:

Although different optimizers share similar purposes, their parameterizations are not interchangeable.

---

# Recording Experimental Results

To compare IID and Non-IID training fairly, separate CSV files were maintained.

```
iid_results.csv
```

contained experiments performed using IID client partitions.

```
noniid_results.csv
```

contained experiments performed using Dirichlet-based client partitions.

This separation simplified later analysis and prevented accidental mixing of results from different experimental settings.

---

# Engineering Challenges Encountered

The implementation process was significantly more challenging than anticipated.

Although the conceptual modifications required only a few hundred lines of code, most development time was spent debugging software compatibility issues.

Several independent problems arose during implementation.

---

## Flower Version Compatibility

One of the earliest challenges involved incompatibilities between different Flower releases.

Some tutorials referenced APIs that no longer existed in the installed version.

Examples included

- `ServerAppComponents`
- `ServerConfig`

These classes were unavailable because newer Flower releases introduced the `ServerApp` API.

After reviewing the official documentation, the implementation was updated to follow the newer application structure.

This experience highlighted the importance of verifying framework versions before following online examples.

---

## Dependency Conflicts

During installation, runtime dependency resolution repeatedly failed.

The project initially requested

```
flwr-datasets >= 0.7.0
```

However, only version

```
0.6.0
```

was publicly available.

As a result, Flower could not create the runtime environment required for simulation.

Updating the dependency specification to

```
flwr-datasets==0.6.0
```

resolved the issue.

This demonstrated how incorrect version constraints can prevent an otherwise correct application from executing.

---

## TOML Configuration Errors

While modifying the project configuration, an accidental quotation mark introduced an invalid dependency declaration.

```toml
""flwr-datasets[vision]>=0.7.0"
```

This caused the parser to report

```
TOMLDecodeError:
Unclosed array
```

Correcting the syntax immediately restored the build process.

Although simple, this bug emphasized the sensitivity of TOML configuration files to small syntax errors.

---

## Package Installation Errors

At one point, the project was installed from the wrong directory.

Running

```bash
pip install -e .
```

outside the project folder resulted in

```
setup.py or pyproject.toml not found
```

Navigating back into the project directory before installation resolved the problem.

---

## Runtime Environment Confusion

Flower creates isolated runtime environments for every simulation.

Initially, this behaviour made it appear that code modifications were being ignored.

In reality, Flower had cached an earlier version of the application.

Reinstalling the project and rerunning the simulation ensured that the latest source code was packaged into the runtime environment.

Understanding this behaviour significantly improved future debugging.

---

## Missing Function Errors

During refactoring, the `load_data()` function was accidentally removed from `task.py`.

The client application therefore failed with

```
ImportError:
cannot import name 'load_data'
```

Restoring the function resolved the issue immediately.

This reinforced the importance of maintaining stable interfaces between application modules.

---

# Lessons From Debugging

Although debugging consumed considerably more time than writing new code, it proved to be one of the most valuable parts of this week's learning experience.

Rather than viewing errors as setbacks, each debugging session provided deeper insight into

- Flower's application architecture,
- dependency management,
- runtime environments,
- optimizer configuration,
- package installation,
- and distributed machine learning workflows.

By the end of the week, the implementation was capable of performing both IID and Non-IID Federated Learning experiments while automatically recording evaluation metrics for further analysis.

This engineering experience was just as valuable as the theoretical concepts introduced during the week.

---

# Experimental Setup

The experiments were conducted using the Flower Federated Learning framework with the CIFAR-10 image classification dataset.

The objective was to compare the convergence behaviour of Federated Learning under two different client data distributions:

- Independent and Identically Distributed (IID)
- Non-Independent and Identically Distributed (Non-IID)

For the Non-IID experiment, client datasets were generated using a Dirichlet distribution with

```
α = 0.5
```

to create moderate statistical heterogeneity.

The same neural network architecture, communication rounds, and learning rate were used for both experiments to ensure that the only major difference between experiments was the client data distribution.

---

# Evaluation Metrics

Two evaluation metrics were monitored throughout training.

## Global Accuracy

Accuracy measures the percentage of correctly classified images on the global test dataset.

Higher values indicate better model performance.

---

## Cross-Entropy Loss

Loss measures how far the model predictions deviate from the correct labels.

Lower values indicate better optimization.

Monitoring both metrics provides a more complete picture of model convergence than accuracy alone.

---

# Experimental Results

Two convergence plots were generated after completing the federated simulations.

The first plot compares the global accuracy achieved under IID and Non-IID training.

The second plot compares the corresponding loss curves.

## Figure 1

**Global Accuracy Comparison**

> *(Insert Accuracy Plot Here)*

---

## Figure 2

**Global Loss Comparison**

> *(Insert Loss Plot Here)*

---

# Discussion of Results

The experimental results clearly demonstrate the impact of heterogeneous client data on Federated Learning.

## Accuracy Comparison

The IID experiment achieved significantly better performance throughout training.

The global model began with an accuracy of approximately **91%** and steadily improved until reaching nearly **96%** after ten communication rounds.

The convergence curve remained smooth and stable, indicating that local client updates were highly consistent.

In contrast, the Non-IID experiment began with a much lower accuracy of approximately **9%** and gradually increased to around **48%** after ten communication rounds.

Although the model improved over time, convergence was noticeably slower.

Unlike the IID experiment, the Non-IID curve displayed larger fluctuations before gradually stabilizing.

These observations demonstrate that heterogeneous client data significantly slows global learning.

---

## Loss Comparison

The loss curves reveal an even stronger contrast.

For the IID experiment, loss decreased smoothly throughout training.

The model consistently became more confident in its predictions as communication rounds progressed.

The Non-IID experiment exhibited a much more unstable optimization process.

Large spikes appeared during several communication rounds before the loss gradually decreased.

Although the model eventually improved, optimization remained considerably less stable than under IID training.

These oscillations are characteristic of Federated Learning under heterogeneous client data.

---

# Understanding Client Drift

The primary reason for the observed behaviour is **client drift**.

Under IID training, every client observes approximately the same data distribution.

Consequently, local gradients point toward similar optimization directions.

When the server averages these updates, the resulting global model converges efficiently.

Under Non-IID training, clients optimize different local objectives because each client owns different subsets of the data.

For example,

one client may primarily observe airplanes,

while another client mostly observes trucks,

and another client mainly receives bird images.

Each client therefore attempts to improve the model for its own local distribution.

When these conflicting updates are averaged, they partially cancel one another.

Instead of moving consistently toward the global optimum, the optimization process oscillates.

This phenomenon is known as **client drift**.

The experimental results clearly illustrate the effect of client drift on Federated Learning convergence.

---

# Reflection

Before conducting this experiment, I expected the Non-IID model to converge more slowly than the IID model because each client would train on different data distributions.

However, I did not anticipate the performance gap to be as large as the one observed during experimentation.

The IID model converged rapidly, achieving approximately **96%** global accuracy after ten communication rounds.

The Non-IID model, on the other hand, reached only about **48%** accuracy over the same number of communication rounds.

The loss curves also differed substantially.

While IID training produced a smooth decrease in loss, the Non-IID experiment experienced multiple spikes before gradually stabilizing.

These observations reinforced the concept of client drift and demonstrated how heterogeneous client datasets complicate distributed optimization.

One question raised by this experiment is:

> How can Federated Learning algorithms be modified to reduce client drift while maintaining user privacy under highly heterogeneous data distributions?

Exploring algorithms such as **FedProx**, **SCAFFOLD**, and **FedNova** may provide answers to this question.

---

# Challenges Encountered

Although the final implementation successfully produced both IID and Non-IID experiments, several practical challenges were encountered during development.

These included:

- Flower version incompatibilities between older tutorials and newer APIs.
- Dependency conflicts involving `flwr-datasets`.
- Runtime environment creation errors.
- Incorrect TOML dependency declarations.
- Package installation from the wrong directory.
- Import errors caused by accidental removal of the `load_data()` function.
- Optimizer configuration issues when switching from SGD to Adam.
- Locating experiment output files generated inside Flower's runtime environments.
- Recording evaluation metrics for later visualization.

Resolving these issues significantly improved my understanding of the Flower framework and distributed machine learning workflows.

---

# Lessons Learned

This week's activities provided several important insights.

- Real-world Federated Learning rarely operates under IID assumptions.
- Client drift is one of the primary causes of slow convergence.
- The Dirichlet distribution provides a practical method for simulating heterogeneous client datasets.
- Even relatively small implementation changes can significantly alter global optimization behaviour.
- Experimental visualization provides much clearer insight than raw numerical logs.
- Debugging distributed systems requires understanding not only machine learning but also software engineering practices such as dependency management, runtime environments, and package configuration.

---

# Conclusion

Week 3 extended the concepts introduced during Week 2 by moving beyond idealized IID assumptions into more realistic heterogeneous Federated Learning environments.

Through implementing Dirichlet-based client partitioning, modifying the Flower Quickstart application, and comparing convergence under IID and Non-IID settings, I gained a deeper understanding of the practical challenges faced by modern Federated Learning systems.

The experiments clearly demonstrated that heterogeneous client datasets slow convergence, reduce global accuracy, and introduce instability through client drift.

Beyond the theoretical concepts, the implementation process also provided valuable experience debugging framework compatibility issues, managing runtime environments, configuring optimizers, and recording experimental results.

Overall, Week 3 strengthened both my theoretical understanding of Federated Learning and my practical software engineering skills, providing a strong foundation for more advanced topics such as personalized Federated Learning, adaptive aggregation algorithms, and communication-efficient optimization.

---

# Future Work

Future experiments could extend this work by investigating:

- Different Dirichlet alpha values.
- Larger numbers of communication rounds.
- Increasing the number of simulated clients.
- Comparing SGD, Adam, and FedOpt optimizers.
- Evaluating FedAvg against FedProx, FedNova, and SCAFFOLD.
- Measuring communication efficiency under different heterogeneity levels.
- Applying the same methodology to larger computer vision datasets.

---

# References

1. McMahan, B., et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.*

2. Flower Framework Documentation  
https://flower.ai/docs/

3. Flower Datasets Documentation  
https://flower.ai/docs/datasets/

4. CIFAR-10 Dataset  
https://www.cs.toronto.edu/~kriz/cifar.html

5. Kairouz, P., et al. (2021). *Advances and Open Problems in Federated Learning.*

6. Li, T., et al. (2020). *Federated Optimization in Heterogeneous Networks (FedProx).*
