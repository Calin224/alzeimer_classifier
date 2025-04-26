```python
import torch
```


```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
```

    cuda
    


```python
%%writefile data_setup.py

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloader(train_dir: str,
                      test_dir: str,
                      transforms: transforms.Compose,
                      batch_size: int = 32,
                      num_workers: int = NUM_WORKERS):
    train_data = datasets.ImageFolder(root=train_dir, transform=transforms)
    test_data = datasets.ImageFolder(root=test_dir, transform=transforms)

    class_names = train_data.classes

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, test_dataloader, class_names
```

    Overwriting data_setup.py
    


```python
import os
from pathlib import Path

train_dir = Path("data/train")
test_dir = Path("data/val")
train_dir, test_dir
```




    (WindowsPath('data/train'), WindowsPath('data/val'))




```python
%%writefile engine.py

import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    train_loss, train_acc = 0, 0

    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):
    test_loss, test_acc = 0, 0

    model.eval()
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred = model(X)

            loss = loss_fn(test_pred, y)
            test_loss += loss.item()

            test_pred_labels = test_pred.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device):
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)

        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)

        print(f"Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
```

    Overwriting engine.py
    


```python
import torchvision
from torchvision import transforms

weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
weights
```




    EfficientNet_B2_Weights.IMAGENET1K_V1




```python
auto_transforms = weights.transforms()
auto_transforms
```




    ImageClassification(
        crop_size=[288]
        resize_size=[288]
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        interpolation=InterpolationMode.BICUBIC
    )




```python
import data_setup

train_dataloader, test_dataloader, class_names = data_setup.create_dataloader(train_dir=train_dir,
                                                                              test_dir=test_dir,
                                                                              transforms=auto_transforms,
                                                                              batch_size=32)

train_dataloader, test_dataloader, class_names
```




    (<torch.utils.data.dataloader.DataLoader at 0x1ccc46ef800>,
     <torch.utils.data.dataloader.DataLoader at 0x1ccc7533080>,
     ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented'])




```python
weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
model = torchvision.models.efficientnet_b2(weights=weights)
model.to(device)
```




    EfficientNet(
      (features): Sequential(
        (0): Conv2dNormActivation(
          (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): SiLU(inplace=True)
        )
        (1): Sequential(
          (0): MBConv(
            (block): Sequential(
              (0): Conv2dNormActivation(
                (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
                (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(32, 8, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(8, 32, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (2): Conv2dNormActivation(
                (0): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.0, mode=row)
          )
          (1): MBConv(
            (block): Sequential(
              (0): Conv2dNormActivation(
                (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
                (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(16, 4, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(4, 16, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (2): Conv2dNormActivation(
                (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.008695652173913044, mode=row)
          )
        )
        (2): Sequential(
          (0): MBConv(
            (block): Sequential(
              (0): Conv2dNormActivation(
                (0): Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): Conv2dNormActivation(
                (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
                (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(96, 4, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(4, 96, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): Conv2dNormActivation(
                (0): Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.017391304347826087, mode=row)
          )
          (1): MBConv(
            (block): Sequential(
              (0): Conv2dNormActivation(
                (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): Conv2dNormActivation(
                (0): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
                (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): Conv2dNormActivation(
                (0): Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.026086956521739136, mode=row)
          )
          (2): MBConv(
            (block): Sequential(
              (0): Conv2dNormActivation(
                (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): Conv2dNormActivation(
                (0): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
                (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): Conv2dNormActivation(
                (0): Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.034782608695652174, mode=row)
          )
        )
        (3): Sequential(
          (0): MBConv(
            (block): Sequential(
              (0): Conv2dNormActivation(
                (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): Conv2dNormActivation(
                (0): Conv2d(144, 144, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=144, bias=False)
                (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): Conv2dNormActivation(
                (0): Conv2d(144, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.043478260869565216, mode=row)
          )
          (1): MBConv(
            (block): Sequential(
              (0): Conv2dNormActivation(
                (0): Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): Conv2dNormActivation(
                (0): Conv2d(288, 288, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=288, bias=False)
                (1): BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(288, 12, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(12, 288, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): Conv2dNormActivation(
                (0): Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.05217391304347827, mode=row)
          )
          (2): MBConv(
            (block): Sequential(
              (0): Conv2dNormActivation(
                (0): Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): Conv2dNormActivation(
                (0): Conv2d(288, 288, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=288, bias=False)
                (1): BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(288, 12, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(12, 288, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): Conv2dNormActivation(
                (0): Conv2d(288, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.06086956521739131, mode=row)
          )
        )
        (4): Sequential(
          (0): MBConv(
            (block): Sequential(
              (0): Conv2dNormActivation(
                (0): Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): Conv2dNormActivation(
                (0): Conv2d(288, 288, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=288, bias=False)
                (1): BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(288, 12, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(12, 288, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): Conv2dNormActivation(
                (0): Conv2d(288, 88, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(88, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.06956521739130435, mode=row)
          )
          (1): MBConv(
            (block): Sequential(
              (0): Conv2dNormActivation(
                (0): Conv2d(88, 528, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(528, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): Conv2dNormActivation(
                (0): Conv2d(528, 528, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=528, bias=False)
                (1): BatchNorm2d(528, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(528, 22, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(22, 528, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): Conv2dNormActivation(
                (0): Conv2d(528, 88, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(88, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.0782608695652174, mode=row)
          )
          (2): MBConv(
            (block): Sequential(
              (0): Conv2dNormActivation(
                (0): Conv2d(88, 528, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(528, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): Conv2dNormActivation(
                (0): Conv2d(528, 528, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=528, bias=False)
                (1): BatchNorm2d(528, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(528, 22, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(22, 528, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): Conv2dNormActivation(
                (0): Conv2d(528, 88, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(88, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.08695652173913043, mode=row)
          )
          (3): MBConv(
            (block): Sequential(
              (0): Conv2dNormActivation(
                (0): Conv2d(88, 528, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(528, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): Conv2dNormActivation(
                (0): Conv2d(528, 528, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=528, bias=False)
                (1): BatchNorm2d(528, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(528, 22, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(22, 528, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): Conv2dNormActivation(
                (0): Conv2d(528, 88, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(88, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.09565217391304348, mode=row)
          )
        )
        (5): Sequential(
          (0): MBConv(
            (block): Sequential(
              (0): Conv2dNormActivation(
                (0): Conv2d(88, 528, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(528, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): Conv2dNormActivation(
                (0): Conv2d(528, 528, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=528, bias=False)
                (1): BatchNorm2d(528, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(528, 22, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(22, 528, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): Conv2dNormActivation(
                (0): Conv2d(528, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.10434782608695654, mode=row)
          )
          (1): MBConv(
            (block): Sequential(
              (0): Conv2dNormActivation(
                (0): Conv2d(120, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): Conv2dNormActivation(
                (0): Conv2d(720, 720, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=720, bias=False)
                (1): BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(720, 30, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(30, 720, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): Conv2dNormActivation(
                (0): Conv2d(720, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.11304347826086956, mode=row)
          )
          (2): MBConv(
            (block): Sequential(
              (0): Conv2dNormActivation(
                (0): Conv2d(120, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): Conv2dNormActivation(
                (0): Conv2d(720, 720, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=720, bias=False)
                (1): BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(720, 30, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(30, 720, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): Conv2dNormActivation(
                (0): Conv2d(720, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.12173913043478261, mode=row)
          )
          (3): MBConv(
            (block): Sequential(
              (0): Conv2dNormActivation(
                (0): Conv2d(120, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): Conv2dNormActivation(
                (0): Conv2d(720, 720, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=720, bias=False)
                (1): BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(720, 30, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(30, 720, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): Conv2dNormActivation(
                (0): Conv2d(720, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.13043478260869565, mode=row)
          )
        )
        (6): Sequential(
          (0): MBConv(
            (block): Sequential(
              (0): Conv2dNormActivation(
                (0): Conv2d(120, 720, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): Conv2dNormActivation(
                (0): Conv2d(720, 720, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=720, bias=False)
                (1): BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(720, 30, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(30, 720, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): Conv2dNormActivation(
                (0): Conv2d(720, 208, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.1391304347826087, mode=row)
          )
          (1): MBConv(
            (block): Sequential(
              (0): Conv2dNormActivation(
                (0): Conv2d(208, 1248, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(1248, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): Conv2dNormActivation(
                (0): Conv2d(1248, 1248, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1248, bias=False)
                (1): BatchNorm2d(1248, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(1248, 52, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(52, 1248, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): Conv2dNormActivation(
                (0): Conv2d(1248, 208, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.14782608695652175, mode=row)
          )
          (2): MBConv(
            (block): Sequential(
              (0): Conv2dNormActivation(
                (0): Conv2d(208, 1248, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(1248, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): Conv2dNormActivation(
                (0): Conv2d(1248, 1248, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1248, bias=False)
                (1): BatchNorm2d(1248, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(1248, 52, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(52, 1248, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): Conv2dNormActivation(
                (0): Conv2d(1248, 208, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.1565217391304348, mode=row)
          )
          (3): MBConv(
            (block): Sequential(
              (0): Conv2dNormActivation(
                (0): Conv2d(208, 1248, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(1248, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): Conv2dNormActivation(
                (0): Conv2d(1248, 1248, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1248, bias=False)
                (1): BatchNorm2d(1248, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(1248, 52, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(52, 1248, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): Conv2dNormActivation(
                (0): Conv2d(1248, 208, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.16521739130434784, mode=row)
          )
          (4): MBConv(
            (block): Sequential(
              (0): Conv2dNormActivation(
                (0): Conv2d(208, 1248, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(1248, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): Conv2dNormActivation(
                (0): Conv2d(1248, 1248, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1248, bias=False)
                (1): BatchNorm2d(1248, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(1248, 52, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(52, 1248, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): Conv2dNormActivation(
                (0): Conv2d(1248, 208, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.17391304347826086, mode=row)
          )
        )
        (7): Sequential(
          (0): MBConv(
            (block): Sequential(
              (0): Conv2dNormActivation(
                (0): Conv2d(208, 1248, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(1248, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): Conv2dNormActivation(
                (0): Conv2d(1248, 1248, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1248, bias=False)
                (1): BatchNorm2d(1248, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(1248, 52, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(52, 1248, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): Conv2dNormActivation(
                (0): Conv2d(1248, 352, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(352, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.1826086956521739, mode=row)
          )
          (1): MBConv(
            (block): Sequential(
              (0): Conv2dNormActivation(
                (0): Conv2d(352, 2112, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(2112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): Conv2dNormActivation(
                (0): Conv2d(2112, 2112, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2112, bias=False)
                (1): BatchNorm2d(2112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(2112, 88, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(88, 2112, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): Conv2dNormActivation(
                (0): Conv2d(2112, 352, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(352, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.19130434782608696, mode=row)
          )
        )
        (8): Conv2dNormActivation(
          (0): Conv2d(352, 1408, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(1408, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): SiLU(inplace=True)
        )
      )
      (avgpool): AdaptiveAvgPool2d(output_size=1)
      (classifier): Sequential(
        (0): Dropout(p=0.3, inplace=True)
        (1): Linear(in_features=1408, out_features=1000, bias=True)
      )
    )




```python
len(class_names)
```




    4




```python
for param in model.features.parameters():
    param.requires_grad = False
```


```python
from torch import nn

model.classifier = torch.nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1408, out_features=len(class_names), bias=True).to(device)
)
```


```python
from torchinfo import summary

summary(model=model.to(device),
        input_size=(1, 3, 288, 288),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"],
        device=device)
```




    ============================================================================================================================================
    Layer (type (var_name))                                      Input Shape          Output Shape         Param #              Trainable
    ============================================================================================================================================
    EfficientNet (EfficientNet)                                  [1, 3, 288, 288]     [1, 4]               --                   Partial
    ├─Sequential (features)                                      [1, 3, 288, 288]     [1, 1408, 9, 9]      --                   False
    │    └─Conv2dNormActivation (0)                              [1, 3, 288, 288]     [1, 32, 144, 144]    --                   False
    │    │    └─Conv2d (0)                                       [1, 3, 288, 288]     [1, 32, 144, 144]    (864)                False
    │    │    └─BatchNorm2d (1)                                  [1, 32, 144, 144]    [1, 32, 144, 144]    (64)                 False
    │    │    └─SiLU (2)                                         [1, 32, 144, 144]    [1, 32, 144, 144]    --                   --
    │    └─Sequential (1)                                        [1, 32, 144, 144]    [1, 16, 144, 144]    --                   False
    │    │    └─MBConv (0)                                       [1, 32, 144, 144]    [1, 16, 144, 144]    (1,448)              False
    │    │    └─MBConv (1)                                       [1, 16, 144, 144]    [1, 16, 144, 144]    (612)                False
    │    └─Sequential (2)                                        [1, 16, 144, 144]    [1, 24, 72, 72]      --                   False
    │    │    └─MBConv (0)                                       [1, 16, 144, 144]    [1, 24, 72, 72]      (6,004)              False
    │    │    └─MBConv (1)                                       [1, 24, 72, 72]      [1, 24, 72, 72]      (10,710)             False
    │    │    └─MBConv (2)                                       [1, 24, 72, 72]      [1, 24, 72, 72]      (10,710)             False
    │    └─Sequential (3)                                        [1, 24, 72, 72]      [1, 48, 36, 36]      --                   False
    │    │    └─MBConv (0)                                       [1, 24, 72, 72]      [1, 48, 36, 36]      (16,518)             False
    │    │    └─MBConv (1)                                       [1, 48, 36, 36]      [1, 48, 36, 36]      (43,308)             False
    │    │    └─MBConv (2)                                       [1, 48, 36, 36]      [1, 48, 36, 36]      (43,308)             False
    │    └─Sequential (4)                                        [1, 48, 36, 36]      [1, 88, 18, 18]      --                   False
    │    │    └─MBConv (0)                                       [1, 48, 36, 36]      [1, 88, 18, 18]      (50,300)             False
    │    │    └─MBConv (1)                                       [1, 88, 18, 18]      [1, 88, 18, 18]      (123,750)            False
    │    │    └─MBConv (2)                                       [1, 88, 18, 18]      [1, 88, 18, 18]      (123,750)            False
    │    │    └─MBConv (3)                                       [1, 88, 18, 18]      [1, 88, 18, 18]      (123,750)            False
    │    └─Sequential (5)                                        [1, 88, 18, 18]      [1, 120, 18, 18]     --                   False
    │    │    └─MBConv (0)                                       [1, 88, 18, 18]      [1, 120, 18, 18]     (149,158)            False
    │    │    └─MBConv (1)                                       [1, 120, 18, 18]     [1, 120, 18, 18]     (237,870)            False
    │    │    └─MBConv (2)                                       [1, 120, 18, 18]     [1, 120, 18, 18]     (237,870)            False
    │    │    └─MBConv (3)                                       [1, 120, 18, 18]     [1, 120, 18, 18]     (237,870)            False
    │    └─Sequential (6)                                        [1, 120, 18, 18]     [1, 208, 9, 9]       --                   False
    │    │    └─MBConv (0)                                       [1, 120, 18, 18]     [1, 208, 9, 9]       (301,406)            False
    │    │    └─MBConv (1)                                       [1, 208, 9, 9]       [1, 208, 9, 9]       (686,868)            False
    │    │    └─MBConv (2)                                       [1, 208, 9, 9]       [1, 208, 9, 9]       (686,868)            False
    │    │    └─MBConv (3)                                       [1, 208, 9, 9]       [1, 208, 9, 9]       (686,868)            False
    │    │    └─MBConv (4)                                       [1, 208, 9, 9]       [1, 208, 9, 9]       (686,868)            False
    │    └─Sequential (7)                                        [1, 208, 9, 9]       [1, 352, 9, 9]       --                   False
    │    │    └─MBConv (0)                                       [1, 208, 9, 9]       [1, 352, 9, 9]       (846,900)            False
    │    │    └─MBConv (1)                                       [1, 352, 9, 9]       [1, 352, 9, 9]       (1,888,920)          False
    │    └─Conv2dNormActivation (8)                              [1, 352, 9, 9]       [1, 1408, 9, 9]      --                   False
    │    │    └─Conv2d (0)                                       [1, 352, 9, 9]       [1, 1408, 9, 9]      (495,616)            False
    │    │    └─BatchNorm2d (1)                                  [1, 1408, 9, 9]      [1, 1408, 9, 9]      (2,816)              False
    │    │    └─SiLU (2)                                         [1, 1408, 9, 9]      [1, 1408, 9, 9]      --                   --
    ├─AdaptiveAvgPool2d (avgpool)                                [1, 1408, 9, 9]      [1, 1408, 1, 1]      --                   --
    ├─Sequential (classifier)                                    [1, 1408]            [1, 4]               --                   True
    │    └─Dropout (0)                                           [1, 1408]            [1, 1408]            --                   --
    │    └─Linear (1)                                            [1, 1408]            [1, 4]               5,636                True
    ============================================================================================================================================
    Total params: 7,706,630
    Trainable params: 5,636
    Non-trainable params: 7,700,994
    Total mult-adds (Units.GIGABYTES): 1.09
    ============================================================================================================================================
    Input size (MB): 1.00
    Forward/backward pass size (MB): 259.12
    Params size (MB): 30.83
    Estimated Total Size (MB): 290.94
    ============================================================================================================================================




```python
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```


```python
import engine

results = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       loss_fn=loss_fn,
                       optimizer=optimizer,
                       epochs=5,
                       device=device)
```


      0%|          | 0/5 [00:00<?, ?it/s]


    Epoch: 1/5, Train Loss: 0.9917, Train Acc: 0.5597, Test Loss: 0.9140, Test Acc: 0.5672
    Epoch: 2/5, Train Loss: 0.9883, Train Acc: 0.5605, Test Loss: 0.9242, Test Acc: 0.5686
    

    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x000001CCB9210860>
    Traceback (most recent call last):
      File "C:\Users\sandu\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\utils\data\dataloader.py", line 1618, in __del__
        self._shutdown_workers()
      File "C:\Users\sandu\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\utils\data\dataloader.py", line 1576, in _shutdown_workers
        if self._persistent_workers or self._workers_status[worker_id]:
                                       ^^^^^^^^^^^^^^^^^^^^
    AttributeError: '_MultiProcessingDataLoaderIter' object has no attribute '_workers_status'
    Exception ignored in: <function _MultiProcessingDataLoaderIter.__del__ at 0x000001CCB9210860>
    Traceback (most recent call last):
      File "C:\Users\sandu\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\utils\data\dataloader.py", line 1618, in __del__
        self._shutdown_workers()
      File "C:\Users\sandu\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\utils\data\dataloader.py", line 1576, in _shutdown_workers
        if self._persistent_workers or self._workers_status[worker_id]:
                                       ^^^^^^^^^^^^^^^^^^^^
    AttributeError: '_MultiProcessingDataLoaderIter' object has no attribute '_workers_status'
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[34], line 3
          1 import engine
    ----> 3 results = engine.train(model=model,
          4                        train_dataloader=train_dataloader,
          5                        test_dataloader=test_dataloader,
          6                        loss_fn=loss_fn,
          7                        optimizer=optimizer,
          8                        epochs=5,
          9                        device=device)
    

    File ~\Desktop\medical_ai\engine.py:74, in train(model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs, device)
         66 results = {
         67     "train_loss": [],
         68     "train_acc": [],
         69     "test_loss": [],
         70     "test_acc": []
         71 }
         73 for epoch in tqdm(range(epochs)):
    ---> 74     train_loss, train_acc = train_step(model=model,
         75                                        dataloader=train_dataloader,
         76                                        loss_fn=loss_fn,
         77                                        optimizer=optimizer,
         78                                        device=device)
         80     test_loss, test_acc = test_step(model=model,
         81                                     dataloader=test_dataloader,
         82                                     loss_fn=loss_fn,
         83                                     device=device)
         85     print(f"Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    

    File ~\Desktop\medical_ai\engine.py:18, in train_step(model, dataloader, loss_fn, optimizer, device)
         15 for batch, (X, y) in enumerate(dataloader):
         16     X, y = X.to(device), y.to(device)
    ---> 18     y_pred = model(X)
         20     loss = loss_fn(y_pred, y)
         21     train_loss += loss.item()
    

    File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\module.py:1739, in Module._wrapped_call_impl(self, *args, **kwargs)
       1737     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1738 else:
    -> 1739     return self._call_impl(*args, **kwargs)
    

    File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\module.py:1750, in Module._call_impl(self, *args, **kwargs)
       1745 # If we don't have any hooks, we want to skip the rest of the logic in
       1746 # this function, and just call forward.
       1747 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
       1748         or _global_backward_pre_hooks or _global_backward_hooks
       1749         or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1750     return forward_call(*args, **kwargs)
       1752 result = None
       1753 called_always_called_hooks = set()
    

    File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\torchvision\models\efficientnet.py:343, in EfficientNet.forward(self, x)
        342 def forward(self, x: Tensor) -> Tensor:
    --> 343     return self._forward_impl(x)
    

    File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\torchvision\models\efficientnet.py:333, in EfficientNet._forward_impl(self, x)
        332 def _forward_impl(self, x: Tensor) -> Tensor:
    --> 333     x = self.features(x)
        335     x = self.avgpool(x)
        336     x = torch.flatten(x, 1)
    

    File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\module.py:1739, in Module._wrapped_call_impl(self, *args, **kwargs)
       1737     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1738 else:
    -> 1739     return self._call_impl(*args, **kwargs)
    

    File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\module.py:1750, in Module._call_impl(self, *args, **kwargs)
       1745 # If we don't have any hooks, we want to skip the rest of the logic in
       1746 # this function, and just call forward.
       1747 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
       1748         or _global_backward_pre_hooks or _global_backward_hooks
       1749         or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1750     return forward_call(*args, **kwargs)
       1752 result = None
       1753 called_always_called_hooks = set()
    

    File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\container.py:250, in Sequential.forward(self, input)
        248 def forward(self, input):
        249     for module in self:
    --> 250         input = module(input)
        251     return input
    

    File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\module.py:1739, in Module._wrapped_call_impl(self, *args, **kwargs)
       1737     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1738 else:
    -> 1739     return self._call_impl(*args, **kwargs)
    

    File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\module.py:1750, in Module._call_impl(self, *args, **kwargs)
       1745 # If we don't have any hooks, we want to skip the rest of the logic in
       1746 # this function, and just call forward.
       1747 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
       1748         or _global_backward_pre_hooks or _global_backward_hooks
       1749         or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1750     return forward_call(*args, **kwargs)
       1752 result = None
       1753 called_always_called_hooks = set()
    

    File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\container.py:250, in Sequential.forward(self, input)
        248 def forward(self, input):
        249     for module in self:
    --> 250         input = module(input)
        251     return input
    

    File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\module.py:1739, in Module._wrapped_call_impl(self, *args, **kwargs)
       1737     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1738 else:
    -> 1739     return self._call_impl(*args, **kwargs)
    

    File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\module.py:1750, in Module._call_impl(self, *args, **kwargs)
       1745 # If we don't have any hooks, we want to skip the rest of the logic in
       1746 # this function, and just call forward.
       1747 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
       1748         or _global_backward_pre_hooks or _global_backward_hooks
       1749         or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1750     return forward_call(*args, **kwargs)
       1752 result = None
       1753 called_always_called_hooks = set()
    

    File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\torchvision\models\efficientnet.py:164, in MBConv.forward(self, input)
        163 def forward(self, input: Tensor) -> Tensor:
    --> 164     result = self.block(input)
        165     if self.use_res_connect:
        166         result = self.stochastic_depth(result)
    

    File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\module.py:1739, in Module._wrapped_call_impl(self, *args, **kwargs)
       1737     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1738 else:
    -> 1739     return self._call_impl(*args, **kwargs)
    

    File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\module.py:1750, in Module._call_impl(self, *args, **kwargs)
       1745 # If we don't have any hooks, we want to skip the rest of the logic in
       1746 # this function, and just call forward.
       1747 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
       1748         or _global_backward_pre_hooks or _global_backward_hooks
       1749         or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1750     return forward_call(*args, **kwargs)
       1752 result = None
       1753 called_always_called_hooks = set()
    

    File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\container.py:250, in Sequential.forward(self, input)
        248 def forward(self, input):
        249     for module in self:
    --> 250         input = module(input)
        251     return input
    

    File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\module.py:1739, in Module._wrapped_call_impl(self, *args, **kwargs)
       1737     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1738 else:
    -> 1739     return self._call_impl(*args, **kwargs)
    

    File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\module.py:1750, in Module._call_impl(self, *args, **kwargs)
       1745 # If we don't have any hooks, we want to skip the rest of the logic in
       1746 # this function, and just call forward.
       1747 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
       1748         or _global_backward_pre_hooks or _global_backward_hooks
       1749         or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1750     return forward_call(*args, **kwargs)
       1752 result = None
       1753 called_always_called_hooks = set()
    

    File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\container.py:250, in Sequential.forward(self, input)
        248 def forward(self, input):
        249     for module in self:
    --> 250         input = module(input)
        251     return input
    

    File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\module.py:1739, in Module._wrapped_call_impl(self, *args, **kwargs)
       1737     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1738 else:
    -> 1739     return self._call_impl(*args, **kwargs)
    

    File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\module.py:1750, in Module._call_impl(self, *args, **kwargs)
       1745 # If we don't have any hooks, we want to skip the rest of the logic in
       1746 # this function, and just call forward.
       1747 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
       1748         or _global_backward_pre_hooks or _global_backward_hooks
       1749         or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1750     return forward_call(*args, **kwargs)
       1752 result = None
       1753 called_always_called_hooks = set()
    

    File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\modules\activation.py:432, in SiLU.forward(self, input)
        431 def forward(self, input: Tensor) -> Tensor:
    --> 432     return F.silu(input, inplace=self.inplace)
    

    File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\nn\functional.py:2379, in silu(input, inplace)
       2377     return handle_torch_function(silu, (input,), input, inplace=inplace)
       2378 if inplace:
    -> 2379     return torch._C._nn.silu_(input)
       2380 return torch._C._nn.silu(input)
    

    KeyboardInterrupt: 



```python

```
