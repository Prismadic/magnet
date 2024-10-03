import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import asyncio
from datetime import datetime
from magnet.utils.data_classes import Status, Run  # Import the Run data class
from magnet.utils.prism.models.cmamba.model.model import CMamba
from magnet.utils.prism.models.cmamba.data_classes import CMambaArgs

# Define the training function
async def train_model(magnet, run, train_loader, test_loader):
    num_epochs = run.params.training_options.num_epochs
    lr = run.params.training_options.learning_rate
    final_model_path = f'/tmp/{run.params.resource_id}'

    # Initialize the model, loss function, and optimizer
    model = CMamba(run.params.training_options, magnet)  # Pass the CMambaArgs object and magnet to the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        magnet.status_callback(Status(datetime.now(), "info", f"\nEpoch [{epoch+1}/{num_epochs}]"))
        magnet.status_callback(Status(datetime.now(), "info", f"Learning Rate: {optimizer.param_groups[0]['lr']}"))

        for i, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            
            loss = criterion(output, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            grad_norms = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm(2).item()
                    grad_norms[f"grad_norm_{name}"] = grad_norm
            
            await magnet.runs_kv.put(key=run.id,value=grad_norms)

            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 10 == 9:
                magnet.status_callback(Status(datetime.now(), "info", f"Step [{i+1}/{len(train_loader)}], Batch Loss: {loss.item():.4f}"))
                await magnet.runs_kv.put(key=run.id,value=loss.item())

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        magnet.status_callback(Status(datetime.now(), "info", f"Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {epoch_loss:.4f}"))
        
        await magnet.runs_kv.put(key=run.id,value=epoch_loss)

        with torch.no_grad():
            weight_stats = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    weight_stats[f"weight_mean_{name}"] = param.mean().item()
                    weight_stats[f"weight_std_{name}"] = param.std().item()
            
            await magnet.runs_kv.put(key=run.id, value=weight_stats)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for j, (X_batch, y_batch) in enumerate(test_loader):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                
                loss = criterion(output, y_batch)
                val_loss += loss.item()
                
                if j % 10 == 9:
                    magnet.status_callback(Status(datetime.now(), "info", f"Validation Batch [{j+1}], Batch Loss: {loss.item():.4f}"))
                    await magnet.runs_kv.put(key=run.id, value=loss.item())
        
        val_loss /= len(test_loader)
        val_losses.append(val_loss)
        magnet.status_callback(Status(datetime.now(), "info", f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}"))
        
        await magnet.runs_kv.put(key=run.id, value=val_loss)

        if (epoch + 1) % 5 == 0:
            model_path = f'c_mamba_model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), model_path)
            await magnet.runs_kv.put(key=run.id, value=model_path)

    if final_model_path:
        torch.save(model.state_dict(), final_model_path)
        await magnet.runs_kv.put(key=run.id, value=final_model_path)

    # Update run with final metrics and status
    run.status = "completed"
    run.end_time = datetime.now()
    run.metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses
    }
    await magnet.runs_kv.put(key=run.id, value=run)