"""
Clean Federated Learning Training Script
with deterministic seeds and configurable parameters
"""

import flwr as fl
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import json
from pathlib import Path

# Import configuration
from config import (
    MODEL_CONFIG, FL_CONFIG, OPTIMIZER_CONFIGS, 
    DATA_CONFIG, EXPERIMENT_CONFIG
)

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed=42):
    """
    Set all random seeds for complete reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    print(f"Random seed set to: {seed}")

# ============================================================================
# MODEL DEFINITION
# ============================================================================

class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for MNIST classification.
    
    Architecture:
        - Input: 784 (28x28 flattened)
        - Hidden: 128 neurons with ReLU
        - Output: 10 classes (digits 0-9)
    """
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        # Initialize weights deterministically
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with deterministic method"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.fc(x)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(client_id, config=None):
    """
    Load and partition MNIST data across clients deterministically.
    
    Args:
        client_id (int): Client ID (0 to num_clients-1)
        config (dict): Configuration dictionary
    
    Returns:
        DataLoader: PyTorch DataLoader for the client's data partition
    """
    if config is None:
        config = {**DATA_CONFIG, **FL_CONFIG}
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    
    dataset = datasets.MNIST(
        config.get("data_dir", "./data"),
        train=True,
        download=config.get("download", True),
        transform=transform
    )
    
    num_clients = config.get("num_clients", 5)
    partition_size = len(dataset) // num_clients
    lengths = [partition_size] * num_clients
    remainder = len(dataset) - sum(lengths)
    lengths[-1] += remainder
    
    # Deterministic split
    generator = torch.Generator().manual_seed(config.get("seed", 42))
    partitions = random_split(dataset, lengths, generator=generator)
    
    return DataLoader(
        partitions[client_id],
        batch_size=config.get("batch_size", 32),
        shuffle=True
    )

# ============================================================================
# FLOWER CLIENT
# ============================================================================

class FlowerClient(fl.client.NumPyClient):
    """
    Federated Learning client with configurable optimizer.
    """
    
    def __init__(self, client_id, config=None):
        """
        Initialize client with model, data, and optimizer.
        
        Args:
            client_id (int): Client identifier
            config (dict): Configuration dictionary
        """
        self.config = config or {}
        self.client_id = client_id
        
        # Initialize model
        self.model = MLP(
            input_size=MODEL_CONFIG["input_size"],
            hidden_size=MODEL_CONFIG["hidden_size"],
            output_size=MODEL_CONFIG["output_size"]
        )
        
        # Load data
        self.trainloader = load_data(client_id, self.config)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer
        self._setup_optimizer()
        
        print(f"Client {client_id} initialized with {self.optimizer_name} optimizer")
    
    def _setup_optimizer(self):
        """Configure optimizer based on settings"""
        opt_config = self.config.get("optimizer_config", {})
        self.optimizer_name = opt_config.get("name", "Adam")
        learning_rate = opt_config.get("learning_rate", 0.01)
        extra_params = opt_config.get("extra_params", {})
        
        if self.optimizer_name.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                **extra_params
            )
        elif self.optimizer_name.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                **extra_params
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
    
    def get_parameters(self, config):
        """Get model parameters as numpy arrays"""
        return [p.detach().cpu().numpy() for p in self.model.parameters()]
    
    def set_parameters(self, parameters):
        """Set model parameters from numpy arrays"""
        for p, new_p in zip(self.model.parameters(), parameters):
            p.data.copy_(torch.from_numpy(new_p))
    
    def fit(self, parameters, config):
        """
        Train the model on local data.
        
        Returns:
            tuple: (parameters, num_examples, metrics)
        """
        self.set_parameters(parameters)
        self.model.train()
        
        for batch_idx, (images, labels) in enumerate(self.trainloader):
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
        
        return (
            self.get_parameters(config),
            len(self.trainloader.dataset),
            {"client_id": self.client_id}
        )
    
    def evaluate(self, parameters, config):
        """
        Evaluate the model on local data.
        
        Returns:
            tuple: (loss, num_examples, metrics)
        """
        self.set_parameters(parameters)
        self.model.eval()
        
        correct = 0
        total = 0
        loss_sum = 0.0
        
        with torch.no_grad():
            for images, labels in self.trainloader:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss_sum += loss.item()
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        average_loss = loss_sum / len(self.trainloader)
        accuracy = correct / total if total > 0 else 0
        
        return (
            float(average_loss),
            total,
            {"accuracy": float(accuracy), "client_id": self.client_id}
        )

# ============================================================================
# STRATEGY AND METRICS
# ============================================================================

def weighted_average(metrics):
    """Compute weighted average of metrics across clients"""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_accuracy = sum(
        num_examples * m["accuracy"] for num_examples, m in metrics
    )
    return {"accuracy": weighted_accuracy / total_examples}

class MetricsLogger(fl.server.strategy.FedAvg):
    """Strategy that logs training metrics"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics_log = []
        self.training_start_time = None
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Log metrics after each evaluation round"""
        if results:
            accuracies = []
            losses = []
            
            for _, evaluate_res in results:
                if hasattr(evaluate_res, 'metrics') and 'accuracy' in evaluate_res.metrics:
                    accuracies.append(evaluate_res.metrics['accuracy'])
                if hasattr(evaluate_res, 'loss'):
                    losses.append(evaluate_res.loss)
            
            avg_accuracy = np.mean(accuracies) if accuracies else 0
            avg_loss = np.mean(losses) if losses else 0
            std_accuracy = np.std(accuracies) if accuracies else 0
            
            self.metrics_log.append({
                'round': server_round,
                'accuracy': avg_accuracy,
                'loss': avg_loss,
                'std_accuracy': std_accuracy,
                'num_clients': len(results)
            })
            
            print(f"Round {server_round}: "
                  f"Acc={avg_accuracy:.4f}±{std_accuracy:.4f}, "
                  f"Loss={avg_loss:.4f}, "
                  f"Active Clients={len(results)}")
        
        return super().aggregate_evaluate(server_round, results, failures)

# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_experiment(experiment_name, config, output_dir="./results"):
    """
    Run a single federated learning experiment.
    
    Args:
        experiment_name (str): Name of the experiment
        config (dict): Complete configuration
        output_dir (str): Directory to save results
    
    Returns:
        dict: Experiment results
    """
    print(f"\n{'='*70}")
    print(f"🔬 Experiment: {experiment_name}")
    print(f"{'='*70}")
    
    # Set seed for reproducibility
    set_seed(config.get("seed", 42))
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create strategy
    strategy = MetricsLogger(
        evaluate_metrics_aggregation_fn=weighted_average,
        min_available_clients=config.get("min_available_clients", 5),
        min_fit_clients=config.get("min_fit_clients", 5),
        min_evaluate_clients=config.get("min_evaluate_clients", 5),
    )
    
    # Client factory function
    def client_fn(cid):
        return FlowerClient(int(cid), config).to_client()
    
    # Run simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config.get("num_clients", 5),
        config=fl.server.ServerConfig(num_rounds=config.get("num_rounds", 5)),
        strategy=strategy
    )
    
    # Save results
    results = {
        "experiment_name": experiment_name,
        "config": config,
        "metrics": strategy.metrics_log,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save to JSON
    json_path = os.path.join(output_dir, f"{experiment_name}_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, f"{experiment_name}_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "accuracy", "loss", "std_accuracy", "num_clients"])
        for entry in strategy.metrics_log:
            writer.writerow([
                entry['round'],
                entry['accuracy'],
                entry['loss'],
                entry.get('std_accuracy', 0),
                entry.get('num_clients', 0)
            ])
    
    print(f"Results saved to {output_dir}")
    return results

def plot_results(all_results, output_dir="./results"):
    """Plot results from multiple experiments"""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    
    for (name, results), color in zip(all_results.items(), colors):
        metrics = results["metrics"]
        rounds = [m['round'] for m in metrics]
        accuracies = [m['accuracy'] * 100 for m in metrics]
        losses = [m['loss'] for m in metrics]
        
        ax1.plot(rounds, accuracies, 'o-', label=name, color=color, linewidth=2)
        ax2.plot(rounds, losses, 'o-', label=name, color=color, linewidth=2)
    
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Model Accuracy Comparison', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Model Loss Comparison', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_plot.png'), dpi=300)
    plt.show()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Set global seed
    set_seed(EXPERIMENT_CONFIG["seed"])
    
    # Create output directory
    output_dir = EXPERIMENT_CONFIG.get("output_dir", "./results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    
    # Define experiments
    experiments_to_run = []
    
    # Adam experiments
    for lr in OPTIMIZER_CONFIGS["adam"]["learning_rates"]:
        config = {
            **DATA_CONFIG,
            **FL_CONFIG,
            **EXPERIMENT_CONFIG,
            "optimizer_config": {
                "name": "Adam",
                "learning_rate": lr,
                "extra_params": OPTIMIZER_CONFIGS["adam"]["extra_params"]
            }
        }
        experiments_to_run.append((f"adam_lr{lr}", config))
    
    # SGD experiments
    for lr in OPTIMIZER_CONFIGS["sgd"]["learning_rates"]:
        config = {
            **DATA_CONFIG,
            **FL_CONFIG,
            **EXPERIMENT_CONFIG,
            "optimizer_config": {
                "name": "SGD",
                "learning_rate": lr,
                "extra_params": OPTIMIZER_CONFIGS["sgd"]["extra_params"]
            }
        }
        experiments_to_run.append((f"sgd_lr{lr}", config))
    
    # Run all experiments
    all_results = {}
    for exp_name, config in experiments_to_run:
        try:
            results = run_experiment(exp_name, config, run_dir)
            all_results[exp_name] = results
        except Exception as e:
            print(f" Failed: {exp_name} - {e}")
    
    # Plot results
    if all_results:
        plot_results(all_results, run_dir)
        
        # Print summary
        print(f"\n{'='*70}")
        print("📊 EXPERIMENT SUMMARY")
        print(f"{'='*70}")
        for name, results in all_results.items():
            final_acc = results["metrics"][-1]["accuracy"] * 100
            print(f"{name:<20}: {final_acc:.2f}% accuracy")
