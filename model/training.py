import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import matplotlib.pyplot as plt
from torch.nn import functional as F
import numpy as np

def train(
    model,
    train_loader,
    val_loader,
    num_epochs,
    learning_rate=1e-3,
    device="cuda",
    model_path="trained_model_belief.pth",
):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1)

    # Initialize lists to track losses
    train_losses = []
    val_losses = []

    epochs_without_improvement = 0
    best_val_loss = 100

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        train_loader_tqdm = tqdm.tqdm(
            train_loader, desc=f"Train Epoch {epoch + 1}/{num_epochs}", unit="batch"
        )

        for images, belief_maps, _ in train_loader_tqdm:
            images, belief_maps = images.to(device), belief_maps.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = torch.tensor(0).float().cuda()

            for stage in range(len(outputs)):
                loss += (
                    (outputs[stage] - belief_maps) * (outputs[stage] - belief_maps)
                ).mean()

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_loader_tqdm.set_postfix(loss=running_loss / len(train_loader))

        avg_training_loss = running_loss / len(train_loader)
        train_losses.append(avg_training_loss)

        # Validation
        model.eval()
        val_loss = 0.0

        val_loader_tqdm = tqdm.tqdm(
            val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}", unit="batch"
        )

        with torch.no_grad():
            for images, belief_maps, _ in val_loader_tqdm:
                images, belief_maps = images.to(device), belief_maps.to(device)
                outputs = model(images)
                loss = torch.tensor(0).float().cuda()

                for stage in range(len(outputs)):
                    loss += (
                        (outputs[stage] - belief_maps) * (outputs[stage] - belief_maps)
                    ).mean()

                val_loss += loss.item()

                val_loader_tqdm.set_postfix(loss=val_loss / len(val_loader))

        avg_val_loss = val_loss / len(val_loader)

        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_training_loss}, Validation Loss: {avg_val_loss}"
        )

        if len(val_losses) >= 5:
            moving_avg_val_loss = np.mean(val_losses[-5:])
            if moving_avg_val_loss < best_val_loss:
                best_val_loss = moving_avg_val_loss
                epochs_without_improvement = 0
                # Save the best model
                torch.save(
                    {
                        "model_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": num_epochs,
                        "learning_rate": learning_rate,
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "loss": avg_val_loss,
                    },
                    model_path,
                )
            else:
                epochs_without_improvement += 1

        if epochs_without_improvement >= 7:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    torch.save(
        {
            "model_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": num_epochs,
            "learning_rate": learning_rate,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "loss": avg_val_loss,
        },
        model_path,
    )

    print(f"Model saved to {model_path}")

    # Plot and save the training and validation loss
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_plot_belief.png")
    plt.close()
    print(f"Plot saved to loss_plot_belief.png")

    return model
