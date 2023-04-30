# Imports
import torch
import torch
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms import *
from models.vgg import VGG
from opti import optimize
assert torch.cuda.is_available()

device = "cuda" if torch.cuda.is_available() else "cpu"

# load le modèle
checkpoint = torch.load('models/vgg.cifar.pretrained.pth')
model = VGG().to(device)
model.load_state_dict(checkpoint["state_dict"])

# load les données
NORMALIZE_DICT = {
    'cifar10':  dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) )
    }
image_size = 32
transforms = {
    "train": Compose([
        RandomCrop(image_size, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        #Normalize(**NORMALIZE_DICT["cifar10"]),
    ]),
    "test": Compose([
        ToTensor(),
        #Normalize(**NORMALIZE_DICT["cifar10"]),
    ]),
}

train_dataset = CIFAR10(
        root="data/cifar10",
        train=True,
        download=True,
        transform=transforms["train"],
    )
test_dataset = CIFAR10(
        root="data/cifar10",
        train=False,
        download=True,
        transform=transforms["test"],
    )


train_loader = DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
test_loader= DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

# définition des paramètres 
speed_up = 2 # basé par rapport au nombre de MACs du modèle
schedule = "oneshot" # oneshot, iterative
epochs_finetuning = 100
method = "group_norm" # random, l1, lamp, slim, group_norm, group_sl
num_classes = 10
global_pruning = True 
save_path = "test.pth"


# pruning le modèle 
optimize(model, train_loader, test_loader, speed_up, schedule, epochs_finetuning, method, num_classes, global_pruning, save_path)

