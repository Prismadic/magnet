import json
import pickle
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
    train_params = CMambaArgs(**run.params.training_options)  # Convert the run parameters to a CMambaArgs object
    num_epochs = train_params.num_epochs
    lr = train_params.learning_rate
    final_model_path = f'/tmp/{run._id}'
    print("Final model path:", final_model_path)
    # Initialize the model, loss function, and optimizer
    model = CMamba(train_params, magnet)  # Pass the CMambaArgs object and magnet to the model
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
        magnet.status_callback(Status(datetime.now(), "info", f"Epoch [{epoch+1}/{num_epochs}]"))
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
            
            # Serialize grad_norms dictionary to bytes using json
            await magnet.runs_kv.put(key=run._id, value=json.dumps(grad_norms).encode())

            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 10 == 9:
                magnet.status_callback(Status(datetime.now(), "info", f"Step [{i+1}/{len(train_loader)}], Batch Loss: {loss.item():.4f}"))
                # Serialize loss value to bytes using json
                await magnet.runs_kv.put(key=run._id, value=json.dumps({"loss": loss.item()}).encode())

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        magnet.status_callback(Status(datetime.now(), "info", f"Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {epoch_loss:.4f}"))
        
        # Serialize epoch_loss to bytes
        await magnet.runs_kv.put(key=run._id, value=json.dumps({"epoch_loss": epoch_loss}).encode())

        with torch.no_grad():
            weight_stats = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    weight_stats[f"weight_mean_{name}"] = param.mean().item()
                    weight_stats[f"weight_std_{name}"] = param.std().item()
            
            # Serialize weight_stats dictionary to bytes
            await magnet.runs_kv.put(key=run._id, value=json.dumps(weight_stats).encode())
        
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
                    # Serialize validation loss to bytes
                    await magnet.runs_kv.put(key=run._id, value=json.dumps({"val_loss": loss.item()}).encode())
        
        val_loss /= len(test_loader)
        val_losses.append(val_loss)
        magnet.status_callback(Status(datetime.now(), "info", f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}"))
        
        # Serialize val_loss to bytes
        await magnet.runs_kv.put(key=run._id, value=json.dumps({"val_loss": val_loss}).encode())

        if (epoch + 1) % 5 == 0:
            model_path = f'c_mamba_model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), model_path)
            # Serialize the model path (string) to bytes
            await magnet.runs_kv.put(key=run._id, value=model_path.encode())

    if final_model_path:
        torch.save(model.state_dict(), final_model_path)
        # Serialize the final model path (string) to bytes
        await magnet.runs_kv.put(key=run._id, value=final_model_path.encode())

    # Update run with final metrics and status
    run.status = "completed"
    run.end_time = datetime.now()
    run.metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses
    }
    # Serialize the `run` object using pickle
    await magnet.runs_kv.put(key=run._id, value=pickle.dumps(run))
