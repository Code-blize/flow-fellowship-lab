import flwr as fl
import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import csv

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
def load_data(client_id, num_clients=5):
    transform = transforms.ToTensor()

    dataset = datasets.MNIST("./data", 
                            train=True, 
                            download=True, 
                            transform=transform
                            )
    
    partition_size = len(dataset) // num_clients
    lengths = [partition_size] * num_clients

    remainder = len(dataset) - sum(lengths)
    lengths[-1] += remainder

    generator = torch.Generator().manual_seed(42)

    partitions = random_split(
        dataset,
        lengths,
        generator=generator
    )


    return DataLoader(partitions[client_id], 
                    batch_size=32, 
                    shuffle=True)

# Flower Client definition
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.model = MLP()
        self.trainloader = load_data(client_id)

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=0.01
        )
    
    def get_parameters(self, config):
        return [
            p.detach().cpu().numpy()
            for p in self.model.parameters()
            ]
    
    def set_parameters(self, parameters):
        for p, new_p in zip(
            self.model.parameters(),
            parameters
        ):
            p.data.copy_(
                torch.from_numpy(new_p)
                )
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        for images, labels in self.trainloader:
            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                labels
                )
            
            loss.backward()

            self.optimizer.step()

        return (
            self.get_parameters(config), 
            len(self.trainloader.dataset), 
            {}
        )
        
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        correct = 0 
        total = 0 
        loss_sum = 0.0

        with torch.no_grad():
            for images, labels in self.trainloader:
                outputs = self.model(images)

                loss = self.criterion(
                    outputs, 
                    labels
                    )
                
                loss_sum += loss.item()

                predictions = outputs.argmax(dim=1)
                
                correct += (
                    predictions == labels
                    ).sum().item()
                
                total += labels.size(0)

        average_loss = loss_sum / len(self.trainloader)
        accuracy = correct / total if total > 0 else 0
                
        return (average_loss, total,
                {"accuracy": accuracy}
                )
    
def weighted_average(metrics):
    """Compute weighted average accuracy across all clients."""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    weighted_accuracy = sum(
        num_examples * m["accuracy"]
        for num_examples, m in metrics
        )

    return {
        "accuracy": weighted_accuracy / total_examples
        }
# Custom strategy to log metrics
class LoggingStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics_log = []  # <-- THIS WAS MISSING!
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Called after server aggregates evaluation results from clients"""
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
            
            print(f"📊 Round {server_round}: Accuracy = {avg_accuracy:.4f}, Loss = {avg_loss:.4f}")
        
        # Call the parent's aggregate_evaluate which handles the weighted_average
        return super().aggregate_evaluate(server_round, results, failures)
    
                
def client_fn(cid):
    return FlowerClient(int(cid)).to_client()  # cid means client

if __name__ == "__main__":
    # Create strategy with logging
    strategy = LoggingStrategy(
        evaluate_metrics_aggregation_fn=weighted_average
    )
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=5,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy
)        

                    
with open("results_adam1.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["round","accuracy", "loss"])

    for entry in strategy.metrics_log:
            writer.writerow([entry['round'], entry['accuracy'], entry['loss']])
    
    print(f"\n Results saved to results_adam1.csv")
    print(f" Final accuracy: {strategy.metrics_log[-1]['accuracy']:.4f}")             

