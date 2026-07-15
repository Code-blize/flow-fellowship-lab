import flwr as fl
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import csv
import matplotlib.pyplot as plt
import numpy as np
import time

# ============================================
# MODEL
# ============================================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.fc(x)

# ============================================
# DATA
# ============================================
def load_data(client_id, num_clients=5, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    
    partition_size = len(dataset) // num_clients
    lengths = [partition_size] * num_clients
    remainder = len(dataset) - sum(lengths)
    lengths[-1] += remainder
    
    generator = torch.Generator().manual_seed(42)
    partitions = random_split(dataset, lengths, generator=generator)
    
    return DataLoader(partitions[client_id], batch_size=batch_size, shuffle=True)

# ============================================
# CLIENT
# ============================================
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, optimizer_name="adam", learning_rate=0.01):
        self.model = MLP()
        self.trainloader = load_data(client_id)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_name = optimizer_name
        
        if optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=learning_rate, 
                momentum=0.9
            )
    
    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in self.model.parameters()]
    
    def set_parameters(self, parameters):
        for p, new_p in zip(self.model.parameters(), parameters):
            p.data.copy_(torch.from_numpy(new_p))
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for images, labels in self.trainloader:
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
        return self.get_parameters(config), len(self.trainloader.dataset), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in self.trainloader:
                outputs = self.model(images)
                loss_sum += self.criterion(outputs, labels).item()
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = loss_sum / len(self.trainloader)
        accuracy = correct / total if total > 0 else 0
        
        return float(avg_loss), total, {"accuracy": float(accuracy)}

# ============================================
# METRICS AGGREGATION
# ============================================
def weighted_average(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_accuracy = sum(num_examples * m["accuracy"] for num_examples, m in metrics)
    return {"accuracy": weighted_accuracy / total_examples}

# ============================================
# RUN SINGLE EXPERIMENT
# ============================================
def run_single_experiment(optimizer_name, learning_rate, num_rounds=5):
    """Run one experiment and return results"""
    print(f"\n{'='*60}")
    print(f"Running: {optimizer_name.upper()} (lr={learning_rate})")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Store results manually
    results_log = []
    
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average,
        min_available_clients=5,
        min_fit_clients=5,
        min_evaluate_clients=5,
    )
    
    # Client function
    def client_fn(cid):
        return FlowerClient(int(cid), optimizer_name, learning_rate).to_client()
    
    # Run simulation with timeout
    try:
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=5,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0},  # Limit resources
        )
        
        # Extract metrics from history
        if hasattr(history, 'metrics_distributed') and history.metrics_distributed:
            for round_num, metrics in history.metrics_distributed.items():
                accuracy = metrics.get('accuracy', 0)
                results_log.append({
                    'round': round_num,
                    'accuracy': accuracy,
                })
                print(f"  Round {round_num}: Accuracy = {accuracy:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        # Return whatever we have
        return results_log
    
    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f} seconds")
    
    return results_log

# ============================================
# MAIN - RUN ALL EXPERIMENTS
# ============================================
if __name__ == "__main__":
    print(" Starting Federated Learning Experiments")
    print("This will take 2-3 minutes per experiment...\n")
    
    # Define experiments
    experiments = [
        ("adam", 0.01, "results_adam_lr0.01.csv"),
        ("adam", 0.001, "results_adam_lr0.001.csv"),
        ("sgd", 0.1, "results_sgd_lr0.1.csv"),
        ("sgd", 0.01, "results_sgd_lr0.01.csv"),
    ]
    
    all_results = {}
    
    # Run each experiment
    for opt_name, lr, filename in experiments:
        print(f"\n Starting {opt_name} lr={lr}...")
        
        results = run_single_experiment(opt_name, lr, num_rounds=5)
        
        if results:
            all_results[f"{opt_name} (lr={lr})"] = results
            
            # Save to CSV
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["round", "accuracy"])
                for entry in results:
                    writer.writerow([entry['round'], entry['accuracy']])
            print(f" Saved: {filename}")
        else:
            print(f" Failed to get results for {opt_name} lr={lr}")
    
    # ============================================
    # CREATE COMPARISON CHART
    # ============================================
    if all_results:
        print(f"\n Creating comparison chart...")
        
        plt.figure(figsize=(10, 6))
        
        colors = {'adam': '#FF6B6B', 'sgd': '#4ECDC4'}
        
        for name, results in all_results.items():
            rounds = [r['round'] for r in results]
            accuracies = [r['accuracy'] * 100 for r in results]
            
            opt_type = name.split()[0].lower()
            color = colors.get(opt_type, 'gray')
            
            plt.plot(rounds, accuracies, marker='o', linewidth=2, 
                    markersize=8, label=name, color=color)
        
        plt.xlabel('Communication Round', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Federated Learning: Optimizer Comparison', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim([80, 100])
        
        plt.tight_layout()
        plt.savefig('optimizer_comparison.png', dpi=150)
        plt.show()
        
        print(" Chart saved: optimizer_comparison.png")
        
        # Print summary table
        print(f"\n{'='*60}")
        print(" FINAL RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"{'Experiment':<20} {'Final Accuracy':<15}")
        print("-" * 35)
        for name, results in all_results.items():
            final_acc = results[-1]['accuracy'] * 100
            print(f"{name:<20} {final_acc:.2f}%")
    
    print(f"\n All experiments complete!")