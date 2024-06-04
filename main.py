import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel
import torch.optim as optim
from dataset_newformat import ArucoDataset
import torchvision.transforms as trans
import torchvision
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import cv2
import os
from datetime import datetime


class ViTForTagDetection(nn.Module):
    def __init__(self):
        super(ViTForTagDetection, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.fc = nn.Linear(self.vit.config.hidden_size, 256)  # 5 conf + 4coordinates 
        self.fc2 = nn.Linear(256, 5)  # compress to 8 coordinates 
        self.gelu = torch.nn.GELU()

    def forward(self, x):
        b = x.shape[0]
        outputs = self.vit(x).last_hidden_state[:, 1:]  # Exclude [CLS] token  [B, 196, 768]
        tag_predictions = self.gelu(self.fc(outputs))
        output = torch.sigmoid(self.fc2(tag_predictions))
        return output.view(b, 14, 14, 5)  # Reshape 

class ResForTagDetection(nn.Module):
    def __init__(self):
        super(ResForTagDetection, self).__init__()
        self.model = models.resnet18(pretrained = True)
        self.fc = nn.Linear(1000, 980)  # 5 conf + 4coordinates 

    def forward(self, x):
        b = x.shape[0]
        outputs = self.model(x) # B 1000
        output = torch.sigmoid(self.fc(outputs))
        return output.view(b, 14, 14, 5)  # Reshape 

def train(model, train_loader, optimizer, criterion, device, writer, epoch):
    model.train()
    train_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Extract predictions and targets
        pred_coords = outputs[..., 1:5]  # Predicted bounding box coordinates (x, y, w, h)
        pred_conf = outputs[..., 0]  # Predicted confidence score
       
        target_coords = labels[..., 1:5]  # Ground truth bounding box coordinates (x, y, w, h)
        target_conf = labels[..., 0]  # Ground truth confidence score (1 for object presence, 0 for no object)
        
        # Mask to ignore confidence loss for grid cells without objects
        noobj_mask = target_conf < 0.5  # Adjust threshold as needed
        
        # Bounding box coordinates loss (Smooth L1 loss)
        bbox_loss = criterion(pred_coords, target_coords)
        
        # Confidence loss (Binary cross-entropy)
        conf_loss = F.binary_cross_entropy(pred_conf, target_conf, reduction='none')
        
        # Apply the weight for grid cells without objects
        conf_loss = torch.where(noobj_mask, 0.5 * conf_loss, conf_loss).sum()
        
        # Total loss
        total_loss = (bbox_loss * 5 + conf_loss) / outputs.size(0)  # Normalize by batch size
        
        # Backward pass and optimize
        total_loss.backward()
        optimizer.step()
        train_loss += total_loss.item()
        if batch_idx % 10 == 0:
            # Log the training loss
            writer.add_scalar('Loss/train', total_loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Conf_Loss/train', conf_loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Bbox_Loss/train', bbox_loss.item(), epoch * len(train_loader) + batch_idx)
        print(train_loss)
    return train_loss


def evaluate(model, test_loader, criterion, device, writer, epoch, dataset, log_dir):
    model.eval()
    test_loss = 0.0
    # with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            # Extract predictions and targets
            pred_coords = outputs[..., 1:5]  # Predicted bounding box coordinates (x, y, w, h)
            pred_conf = outputs[..., 0]  # Predicted confidence score
        
            target_coords = labels[..., 1:5]  # Ground truth bounding box coordinates (x, y, w, h)
            target_conf = labels[..., 0]  # Ground truth confidence score (1 for object presence, 0 for no object)
            
            # Mask to ignore confidence loss for grid cells without objects
            noobj_mask = target_conf < 0.5  # Adjust threshold as needed
            
            # Bounding box coordinates loss (Smooth L1 loss)
            bbox_loss = criterion(pred_coords, target_coords)
            
            # Confidence loss (Binary cross-entropy)
            conf_loss = F.binary_cross_entropy(pred_conf, target_conf, reduction='none')
            
            # Apply the weight for grid cells without objects
            conf_loss = torch.where(noobj_mask, 0.5 * conf_loss, conf_loss).sum()
            
            # Total loss
            total_loss = (bbox_loss * 5 + conf_loss) / outputs.size(0)  # Normalize by batch size

            test_loss += total_loss.item()
            
            if batch_idx != 0 and batch_idx % 10 == 0:
                # Save visualizations
                for idx, image in enumerate(images):
                    image = images[idx].cpu().numpy().transpose(1, 2, 0)
                    image = image * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]
                    batch_labels = labels[idx].cpu().detach().numpy()
                    dataset.save_and_visualize(image, batch_labels, f'{log_dir}/batch{batch_idx}_image{idx}_visualization.png',vis=False)

            # Log the evaluation loss
            writer.add_scalar('Loss/test', total_loss.item(), epoch * len(test_loader) + batch_idx)
            writer.add_scalar('Conf_Loss/test', conf_loss.item(), epoch * len(test_loader) + batch_idx)
            writer.add_scalar('Bbox_Loss/test', bbox_loss.item(), epoch * len(test_loader) + batch_idx)
    print(f"test: {test_loss}")
    return test_loss



def main():
    # Set up TensorBoard writer with a specific log directory and name
    log_dir = os.path.join("runs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    writer = SummaryWriter(log_dir=log_dir)
    train_set = '/home/yunchuz/deeptag/dataset/train.txt'
    test_set = '/home/yunchuz/deeptag/dataset/test.txt'
    train_set = '/home/yunchuz/deeptag/dataset/train_with_noise.txt'
    test_set = '/home/yunchuz/deeptag/dataset/test_with_noise.txt'
    train_input, test_input = [],[]
    with open(train_set, 'r') as f:
        lines = f.readlines()
        for line in lines:
            train_input.append(line[:-1])
    with open(test_set, 'r') as f:
        lines = f.readlines()
        for line in lines:
            test_input.append(line[:-1])
    train_input = train_input
    test_input = test_input
    print(f"Train set:{len(train_input)}")
    print(f"Test set:{len(test_input)}")

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
    # model = ViTForTagDetection()
    model = ResForTagDetection()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.MSELoss(reduction='sum')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        # Train the model
        train_loss = train(model, train_dataloader, optimizer, criterion, device, writer, epoch)
        # Evaluate the model
        if epoch % 10 == 0:
            test_loss = evaluate(model, test_dataloader, criterion, device, writer, epoch, test_dataset,log_dir)
            # Print epoch statistics
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        else:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')

    writer.close()

if __name__ == "__main__":
    main()
