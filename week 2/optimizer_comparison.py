import flwr as fl
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Set deterministic seeds for reproducibility
def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Simple neural-network model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.fc(x)

# Load and split the MNIST across 5 clients
def load_data(client_id, num_clients=5, seed=42):
    """Load MNIST data and split deterministically across clients"""
    transform = transforms.ToTensor()
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    
    partition_size = len(dataset) // num_clients
    lengths = [partition_size] * num_clients
    remainder = len(dataset) - sum(lengths)
    lengths[-1] += remainder
    
    generator = torch.Generator().manual_seed(seed)
    partitions = random_split(dataset, lengths, generator=generator)
    
    return DataLoader(partitions[client_id], batch_size=32, shuffle=True)

# Flower Client with configurable optimizer
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, optimizer_name="adam", learning_rate=0.01):
        self.model = MLP()
        self.trainloader = load_data(client_id)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        
        # Configure optimizer
        if optimizer_name.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=learning_rate
            )
        elif optimizer_name.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=learning_rate,
                momentum=0.9  # Add momentum for better SGD performance
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
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
        
        return float(average_loss), total, {"accuracy": float(accuracy)}

def weighted_average(metrics):
    """Compute weighted average accuracy across all clients."""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_accuracy = sum(num_examples * m["accuracy"] for num_examples, m in metrics)
    return {"accuracy": weighted_accuracy / total_examples}

class LoggingStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics_log = []
    
    def aggregate_evaluate(self, server_round, results, failures):
        if results:
            accuracies = []
            losses = []
            
            for _, evaluate_res in results:
                if hasattr(evaluate_res, 'metrics') and 'accuracy' in evaluate_res.metrics:
                    accuracies.append(evaluate_res.metrics['accuracy'])
                if hasattr(evaluate_res, 'loss'):
                    losses.append(evaluate_res.loss)
            
            avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            
            self.metrics_log.append({
                'round': server_round,
                'accuracy': avg_accuracy,
                'loss': avg_loss
            })
            
            print(f"Round {server_round}: Accuracy = {avg_accuracy:.4f}, Loss = {avg_loss:.4f}")
        
        return super().aggregate_evaluate(server_round, results, failures)

def run_experiment(optimizer_name, learning_rate, num_rounds=5, seed=42):
    """Run a single experiment with given configuration"""
    set_seed(seed)
    
    print(f"\n{'='*60}")
    print(f"Experiment: {optimizer_name.upper()} | lr={learning_rate} | rounds={num_rounds}")
    print(f"{'='*60}")
    
    strategy = LoggingStrategy(
        evaluate_metrics_aggregation_fn=weighted_average
    )
    
    def client_fn(cid):
        return FlowerClient(
            int(cid), 
            optimizer_name=optimizer_name, 
            learning_rate=learning_rate
        ).to_client()
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=5,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )
    
    return strategy.metrics_log

def create_comparison_charts(all_results, save_path="optimizer_comparison.png"):
    """Create comprehensive comparison charts"""
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Federated Learning: Optimizer Comparison on MNIST', 
                fontsize=16, fontweight='bold')
    
    # Color scheme for different optimizers
    colors = {
        'adam': '#FF6B6B',
        'sgd': '#4ECDC4'
    }
    
    markers = {
        'adam': 'o',
        'sgd': 's'
    }
    
    # Plot 1: Accuracy comparison
    ax1 = axes[0, 0]
    for name, results in all_results.items():
        rounds = [r['round'] for r in results]
        accuracies = [r['accuracy'] * 100 for r in results]
        
        # Extract optimizer type from name
        opt_type = name.split()[0].lower()
        color = colors.get(opt_type, '#95A5A6')
        marker = markers.get(opt_type, 'd')
        
        ax1.plot(rounds, accuracies, marker=marker, linewidth=2, 
                markersize=8, label=name, color=color)
        
        # Add value labels on points
        for i, (x, y) in enumerate(zip(rounds, accuracies)):
            if i % 1 == 0:  # Label every point
                ax1.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8)
    
    ax1.set_xlabel('Communication Round', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Model Accuracy Over Rounds', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([min([r['accuracy']*100 for results in all_results.values() for r in results])-5, 
                max([r['accuracy']*100 for results in all_results.values() for r in results])+2])
    
    # Plot 2: Loss comparison
    ax2 = axes[0, 1]
    for name, results in all_results.items():
        rounds = [r['round'] for r in results]
        losses = [r['loss'] for r in results]
        
        opt_type = name.split()[0].lower()
        color = colors.get(opt_type, '#95A5A6')
        marker = markers.get(opt_type, 'd')
        
        ax2.plot(rounds, losses, marker=marker, linewidth=2, 
                markersize=8, label=name, color=color)
    
    ax2.set_xlabel('Communication Round', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Model Loss Over Rounds', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Learning speed comparison (bar chart)
    ax3 = axes[1, 0]
    optimizers = list(all_results.keys())
    final_accuracies = [results[-1]['accuracy'] * 100 for results in all_results.values()]
    
    bars = ax3.bar(optimizers, final_accuracies, color=[colors.get(opt.split()[0].lower(), '#95A5A6') 
                                                    for opt in optimizers])
    ax3.set_ylabel('Final Accuracy (%)', fontsize=12)
    ax3.set_title('Final Model Accuracy Comparison', fontsize=13, fontweight='bold')
    ax3.set_ylim([min(final_accuracies)-2, max(final_accuracies)+2])
    
    # Add value labels on bars
    for bar, value in zip(bars, final_accuracies):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=15, ha='right')
    
    # Plot 4: Convergence speed (rounds to reach 95% accuracy)
    ax4 = axes[1, 1]
    rounds_to_95 = {}
    
    for name, results in all_results.items():
        for r in results:
            if r['accuracy'] * 100 >= 95:
                rounds_to_95[name] = r['round']
                break
        else:
            rounds_to_95[name] = len(results)  # Never reached 95%
    
    opt_names = list(rounds_to_95.keys())
    opt_rounds = list(rounds_to_95.values())
    
    bars2 = ax4.barh(opt_names, opt_rounds, color=[colors.get(opt.split()[0].lower(), '#95A5A6') 
                                                for opt in opt_names])
    ax4.set_xlabel('Rounds to Reach 95% Accuracy', fontsize=12)
    ax4.set_title('Convergence Speed Comparison', fontsize=13, fontweight='bold')
    
    # Add value labels
    for bar, value in zip(bars2, opt_rounds):
        width = bar.get_width()
        ax4.text(width, bar.get_y() + bar.get_height()/2.,
                f' {value} rounds', ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nCharts saved as '{save_path}'")

def create_summary_table(all_results):
    """Print a summary table of results"""
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Optimizer':<20} {'Final Acc':<12} {'Final Loss':<12} {'Best Acc':<12} {'Best Round':<12}")
    print("-" * 80)
    
    for name, results in all_results.items():
        final_acc = results[-1]['accuracy'] * 100
        final_loss = results[-1]['loss']
        best_acc = max(r['accuracy'] for r in results) * 100
        best_round = [r['round'] for r in results if r['accuracy'] * 100 == best_acc][0]
        
        print(f"{name:<20} {final_acc:<12.2f}% {final_loss:<12.4f} {best_acc:<12.2f}% {best_round:<12}")
    
    print(f"{'='*80}\n")

if __name__ == "__main__":
    # Configuration
    NUM_ROUNDS = 5
    SEED = 42
    
    # Define experiments to run
    experiments = [
        {"optimizer": "adam", "lr": 0.01, "label": "Adam (lr=0.01)"},
        {"optimizer": "adam", "lr": 0.001, "label": "Adam (lr=0.001)"},
        {"optimizer": "sgd", "lr": 0.01, "label": "SGD (lr=0.01)"},
        {"optimizer": "sgd", "lr": 0.1, "label": "SGD (lr=0.1)"},
    ]
    
    all_results = {}
    experiment_data = []
    
    # Run all experiments
    for exp in experiments:
        try:
            results = run_experiment(
                optimizer_name=exp["optimizer"],
                learning_rate=exp["lr"],
                num_rounds=NUM_ROUNDS,
                seed=SEED
            )
            all_results[exp["label"]] = results
            
            # Save individual results
            filename = f"results_{exp['optimizer']}_lr{str(exp['lr']).replace('.', '_')}.csv"
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["round", "accuracy", "loss"])
                for entry in results:
                    writer.writerow([entry['round'], entry['accuracy'], entry['loss']])
            
            print(f"Saved: {filename}")
            
        except Exception as e:
            print(f"Experiment failed: {exp['label']}")
            print(f"Error: {e}")
    
    # Create visualization and summary
    if all_results:
        create_comparison_charts(all_results)
        create_summary_table(all_results)
        
        # Save complete results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"all_results_{timestamp}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["experiment", "round", "accuracy", "loss"])
            for exp_name, results in all_results.items():
                for entry in results:
                    writer.writerow([exp_name, entry['round'], 
                                entry['accuracy'], entry['loss']])
