import torch
from tqdm import tqdm

def train(model, output_path, train_loader, val_loader, optimizer, criterion, num_epochs=10, device="cpu"):
    best_accuracy = 0.0
    best_model_weights = None

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

        # Validation
        val_accuracy = evaluate(model, val_loader, device)
        print(f"Validation accuracy: {val_accuracy:.2f}%")
        if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                print(f"New best model saved with accuracy: {best_accuracy:.2f}%")
                best_model_weights = {k: v.clone() for k, v in model.state_dict().items()}  

    model.load_state_dict(best_model_weights)
    torch.save(model, output_path)
    print(f"Best model saved at {output_path}")

    return model


def evaluate(model, data_loader, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    accuracy = 0.0
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total              
    return accuracy