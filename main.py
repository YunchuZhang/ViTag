import torch
import torch.nn as nn
from transformers import ViTModel
import torch.optim as optim
from dataset import ArucoDataset
import torchvision.transforms as trans
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import cv2
import os
from datetime import datetime


class ViTForTagDetection(nn.Module):
    def __init__(self):
        super(ViTForTagDetection, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.fc = nn.Linear(self.vit.config.hidden_size, 8)  # 8 coordinates 
        self.fc2 = nn.Linear(196*8, 12*8)  # compress to 8 coordinates 
        self.gelu = torch.nn.GELU()

    def forward(self, x):
        b = x.shape[0]
        outputs = self.vit(x).last_hidden_state[:, 1:]  # Exclude [CLS] token  [32, 196, 768]
        tag_predictions = self.gelu(self.fc(outputs))
        output = torch.sigmoid(self.fc2(tag_predictions.view(b, -1)))
        return output.view(-1, 12, 8)  # Reshape to (batch_size, 12, 8)



def train(model, train_loader, optimizer, criterion, device, writer, epoch):
    model.train()
    train_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Log the training loss
        writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)

    return train_loss / len(train_loader)


def evaluate(model, test_loader, criterion, device, writer, epoch, dataset, log_dir):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            
            if batch_idx % 30 == 0:
                # Save visualizations
                for idx, image in enumerate(images):
                    image = images[idx].cpu().numpy().transpose(1, 2, 0)
                    image = image * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]
                    batch_labels = outputs[idx].cpu().numpy()
                    dataset.save_and_visualize(image, batch_labels*224, f'{log_dir}/batch{batch_idx}_image{idx}_visualization.png',vis=False)

            # Log the evaluation loss
            writer.add_scalar('Loss/test', loss.item(), epoch * len(test_loader) + batch_idx)

    return test_loss / len(test_loader)



def main():
    # Set up TensorBoard writer with a specific log directory and name
    log_dir = os.path.join("runs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    writer = SummaryWriter(log_dir=log_dir)
    train_set = '/home/yunchuz/deeptag/dataset/train.txt'
    test_set = '/home/yunchuz/deeptag/dataset/test.txt'
    train_input, test_input = [],[]
    with open(train_set, 'r') as f:
        lines = f.readlines()
        for line in lines:
            train_input.append(line[:-1])
    with open(test_set, 'r') as f:
        lines = f.readlines()
        for line in lines:
            test_input.append(line[:-1])
    train_input = train_input[:100000]
    test_input = test_input[:5000]
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)

    transform = trans.Compose([
        trans.ToPILImage(),
        trans.Resize((224, 224)),
        trans.ToTensor(),
        trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # based on ViT paper
    ])

    # Create datasets and dataloaders
    train_dataset = ArucoDataset(train_input, aruco_dict, transform=transform)
    test_dataset = ArucoDataset(test_input, aruco_dict, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize model, optimizer, and criterion
    model = ViTForTagDetection()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        # Train the model
        train_loss = train(model, train_dataloader, optimizer, criterion, device, writer, epoch)

        # Evaluate the model
        test_loss = evaluate(model, test_dataloader, criterion, device, writer, epoch, test_dataset,log_dir)
        # Print epoch statistics
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    writer.close()

if __name__ == "__main__":
    main()
