import torch
from torch import nn
from torchvision import transforms, datasets, models
from torch.utils.tensorboard import SummaryWriter
import random
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.models import resnet101
from torch.optim.lr_scheduler import ReduceLROnPlateau
from timeit import default_timer as timer

def main():
    epochs = 10

    writer = SummaryWriter("/scratch/g/plaviole/srubenstein/DeepLearning/logs")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # import dataset
    data_path = Path("/scratch/g/plaviole/srubenstein/DeepLearning/bigger_dataset")

    # transform data
    data_transform = transforms.Compose([
        # resize to be half as large
        transforms.Resize(250),
        transforms.RandomCrop(240),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # crop to resnet input size
        transforms.CenterCrop(224),
        # Flip images randomly on the horizontal
        transforms.RandomHorizontalFlip(p=0.5),
        # Turn the image into a torch.Tensor
        transforms.ToTensor()
    ])

    train_dir = data_path / "train"
    test_dir = data_path / "test"

    train_data = datasets.ImageFolder(root=train_dir, transform=data_transform, target_transform=None)
    test_data = datasets.ImageFolder(root=test_dir, transform=data_transform)

    # get class names as a list
    class_names = train_data.classes

    # put data in DataLoaders and get model
    model = resnet101(weights='DEFAULT')
    model.to(device)
    train_dataloader = DataLoader(dataset=train_data, batch_size=1, num_workers=0, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=1, num_workers=0, shuffle=True)
    if(next(model.parameters()).is_cuda):   
        print("Model on GPU")
    else:
        print("Model on CPU")

    # define loss function, optimizer, and scheduler
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=6, min_lr=0.001)

    for epoch in range(epochs):
        print(f"Epoch: {epoch}\n----")
        # train
        train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer, device)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)

        # test
        test_loss, test_acc = test(test_dataloader, model, loss_fn, device)
        scheduler.step(test_loss)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)

        print(f"\nTrain loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

    writer.close()
    print("program finished")

# function to calculate accuracy
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """

    # for testing:
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

# function for printing train time
def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
  """Prints difference between start and end time."""
  total_time = end-start
  print(f"Train time on {device}: {total_time:.3f} seconds")
  return total_time

def train(train_dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    total_acc = 0
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        # forward pass
        y_pred = model(X)
        # calculate loss
        loss = loss_fn(y_pred, y)
        
        l2_reg_lambda = 0.0001
        l2_reg = 0
        for param in model.parameters():
            l2_reg += torch.sum(param**2)
            
        # Calculate the total loss, including the MSE loss and the L2 regularization term
        loss = loss + l2_reg_lambda * l2_reg
        total_loss += loss
        
        accuracy += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        total_acc += accuracy
        # optimizer zero grad
        optimizer.zero_grad()
        # loss backward
        loss.backward()
        # optimizer step
        optimizer.step()

        # print what's happening
        if(batch % 100 == 0):
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)}")

    total_loss /= len(train_dataloader)
    total_acc /= len(train_dataloader)

    return total_loss, total_acc

def test(test_dataloader, model, loss_fn, device):
    test_loss, test_acc = 0, 0
    model.eval()
    
    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            # forward pass
            test_pred = model(X_test)
            # calculate the loss
            test_loss += loss_fn(test_pred, y_test)
            # calculate the accuracy
            test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))
            
        # calculate the test loss average per batch
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

    return test_loss, test_acc

if __name__ == '__main__':
    main()