import pathlib
import pathlib
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn
import torch
from dataset import ImageDataset
from torch.utils.data import DataLoader
from model import Model
from dataset import ImageDataset

base_dir = pathlib.Path("./dataset")
folders = ["Megan Fox", "Robert Downey Jr", "Will Smith"]
train_file_list = []
test_file_list = []
train_labels = []
test_labels = []
for folder in folders:
    imgdir_path = base_dir / folder
    individual_file_list = sorted([str(path) for path in imgdir_path.glob("*.png")])
    train_file_list.extend(individual_file_list[:80])
    test_file_list.extend(individual_file_list[80:])
    
    # Assign labels based on folder name
    if folder == "Megan Fox":
        train_labels.extend([0] * 80)
        test_labels.extend([0] * (len(individual_file_list) - 80))
    elif folder == "Robert Downey Jr":
        train_labels.extend([1] * 80)
        test_labels.extend([1] * (len(individual_file_list) - 80))
    else:
        train_labels.extend([2] * 80)
        test_labels.extend([2] * (len(individual_file_list) - 80))

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize([300, 300]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transform = transform = transforms.Compose([
    transforms.Resize([300, 300]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
batch_size = 20
#torch.manual_seed(1) #torch.manual_seed(seed) is a function in PyTorch that sets the seed for generating random numbers.
#torch.cuda.manual_seed_all(1) #for GPU.
train_dataset = ImageDataset(train_file_list, train_labels, transform_train)
valid_dataset = ImageDataset(test_file_list, test_labels, transform_train)
train_dl = DataLoader(train_dataset,batch_size, shuffle=True)
valid_dl = DataLoader(valid_dataset,batch_size, shuffle=False)

# Train the model.
model = Model()
num_epochs = 10
loss_hist_train = [0] * num_epochs
accuracy_hist_train = [0] * num_epochs
loss_hist_valid = [0] * num_epochs
accuracy_hist_valid = [0] * num_epochs
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(num_epochs):
    model.train()
    for x_batch, y_batch in train_dl:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_hist_train[epoch] += loss.item()*y_batch.size(0)
        is_correct = (pred.argmax(dim=1) == y_batch).float()
        accuracy_hist_train[epoch] += is_correct.sum()
    loss_hist_train[epoch] /= len(train_dl.dataset)
    accuracy_hist_train[epoch] /= len(train_dl.dataset)

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in valid_dl:
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss_hist_valid[epoch] += \
                loss.item() * y_batch.size(0)
            is_correct = (pred.argmax(dim=1) == y_batch).float()
            accuracy_hist_valid[epoch] += is_correct.sum()
    loss_hist_valid[epoch] /= len(valid_dl.dataset)
    accuracy_hist_valid[epoch] /= len(valid_dl.dataset)

    print(f'Epoch {epoch+1} accuracy: '
          f'{accuracy_hist_train[epoch]:.4f} val_accuracy: '
          f'{accuracy_hist_valid[epoch]:.4f}')
    
# Plotting loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), loss_hist_train, label='Training Loss')
plt.plot(range(1, num_epochs + 1), loss_hist_valid, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plotting accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), accuracy_hist_train, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), accuracy_hist_valid, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()