import torch
from tqdm import tqdm
import numpy as np
from dataloader import create_dataloaders
from models.registry import REGISTRY
from utils import CreateFolder
import argparse
from pathlib import Path


def main(args):
        
    model_cfg = REGISTRY[args.model]
    input_type = model_cfg["input_type"]
    task_type = model_cfg["task_type"]
    
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_config,
        batch_size=128,
        shuffle=True,
        val_size=0.2,
        input_type=input_type,
        target_type=task_type,
        label_format="index",
        apply_scaling=True,
    )
    
    if args.plot_data:
        from plotting import plot_data_distributions
        plot_data_distributions(train_loader)
    
    model = model_cfg["class"]()
    
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_params}")
    
    
    labels = train_loader.dataset.get_all_class_labels()
    counts = np.bincount(labels)
    weights = 1.0 / counts
    weights = weights / weights.sum()
    print("Class counts:", counts)
    print("Class weights:", weights)
    
    
    if task_type == "classification":
        # criterion = torch.nn.CrossEntropyLoss()
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float,device=args.device))  # Adjust weights for class imbalance
    elif task_type == "regression":
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError(f"Unknown task: {task_type}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(args.device)    
    
    global_step, best_val_loss = 0, float('inf')
    for epoch in range(args.epochs):
        running_loss, running_n = 0.0, 0
    
        pbar = tqdm(train_loader, desc=f"Training epoch {epoch+1}/{args.epochs}", unit="batch")
        for step_in_epoch, (inputs, labels) in enumerate(pbar):
            step = global_step 
            global_step += 1
    
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss
            running_n += 1
    
            pbar.set_postfix({
                "running loss": f"{running_loss.item() / running_n:.3f}",
                "loss": f"{loss.item():.3f}"
            })
    
            if args.eval_freq > 0 and step % args.eval_freq == 0:
                val_loss, val_acc, val_n = 0.0, 0.0, 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs = inputs.to(args.device)
                        labels = labels.to(args.device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss
                        if task_type == "classification":
                            preds = outputs.argmax(dim=1)
                            val_acc += (preds == labels).float().mean().item()
                        val_n += 1
    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model, f"{args.output}/best_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SmartPixels models")

    # Dataset and architecture
    parser.add_argument(
        "--data-config", "-c", type=str, default="config/baseline.yml"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="towards-model-2", choices=list(REGISTRY.keys()), help="Model architecture."
    )

    # Training parameters
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", "-bs", type=int, default=1024, help="Batch size")
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--val-size", type=float, default=0.01)
    parser.add_argument(
        "--num-iterations", "-ni", type=int, default=100_000, help="Number of training iterations"
    )
    parser.add_argument(
        "--eval-freq", "-ef", type=int, default=200, 
        help="Evaluation frequency. Evaluation is deactivated if set to 0."
    )

    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cuda", "cpu", "mps"],
        help="Device to use (cuda is faster)"
    )

    parser.add_argument(
        "--output", "-o", action=CreateFolder, type=Path, default="output/",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--plot-data", action='store_true', help="Plot data distributions after loading"
    )

    args = parser.parse_args()

    main(args)
