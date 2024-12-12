import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import timm
from imagenet import ImageNetDataset
from tqdm import tqdm
from codecarbon import EmissionsTracker
import pandas as pd

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

imagenet_data = ImageNetDataset(root=r'D:\桌面\evaluate\Imagenet1', transform=transform)
train_size = int(0.8 * len(imagenet_data))
val_size = len(imagenet_data) - train_size
train_dataset, val_dataset = random_split(imagenet_data, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = timm.create_model('gcvit_small', pretrained=True)

if hasattr(model, 'head') and hasattr(model.head, 'fc') and isinstance(model.head.fc, nn.Linear):
    num_ftrs = model.head.fc.in_features
    model.head.fc = nn.Linear(num_ftrs, 1000)
else:
    print("Available model attributes:", dir(model))
    raise ValueError("Unexpected model attributes.")

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
tracker = EmissionsTracker()

best_accuracy = 0.0
best_model_wts = model.state_dict()

num_epochs = 10
batch_interval = 100

results = []
batch_counter = 0

for epoch in range(num_epochs):
    tracker.start()
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(train_loader, desc=f'Epoch [{epoch + 1}/{num_epochs}]'):
        inputs, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        batch_counter += 1

        if batch_counter % batch_interval == 0:
            accuracy = 100 * correct / total
            epoch_loss = running_loss / len(train_dataset)
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_inputs, val_labels = val_batch[0].to(device), val_batch[1].to(device)
                    val_outputs = model(val_inputs)
                    _, val_predicted = torch.max(val_outputs, 1)
                    val_total += val_labels.size(0)
                    val_correct += (val_predicted == val_labels).sum().item()

            val_accuracy = 100 * val_correct / val_total
            power = tracker._do_measurements()
            tracker.stop()
            tracker.start()

            print(f"Epoch [{epoch + 1}], Batch [{batch_counter}], Loss: {epoch_loss:.4f}, Train Accuracy: {accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%")
            print(f"Power Consumption (kWh): {power:.4f}")

            results.append({
                'Epoch': epoch + 1,
                'Batch': batch_counter,
                'Loss': epoch_loss,
                'Train Accuracy (%)': accuracy,
                'Validation Accuracy (%)': val_accuracy,
                'Power Consumption (kWh)': power
            })

            df = pd.DataFrame(results)
            df.to_excel('result.xlsx', index=False)

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model_wts = model.state_dict()

df = pd.DataFrame(results)
df.to_excel('result.xlsx', index=False)

torch.save(best_model_wts, 'best.pt')
torch.save(model.state_dict(), 'last.pt')
