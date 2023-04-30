# Imports
import torch.nn.functional as F
import torch
import torch_pruning as tp
import os
import copy
import random
import numpy as np
import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm
import argparse
from functools import partial
from models.vgg import VGG
assert torch.cuda.is_available()


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fixer le seed pour la reproductibilité
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# Evaluation loop
@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    verbose=True,
) -> float:
    model.eval()

    num_samples = 0
    num_correct = 0
    loss = 0

    for inputs, targets in tqdm(dataloader, desc="eval", leave=False, disable=not verbose):
        # Move the data from CPU to GPU
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Inference
        outputs = model(inputs)
        # Calculate loss
        loss += F.cross_entropy(outputs, targets, reduction="sum")
        # Convert logits to class indices
        outputs = outputs.argmax(dim=1)
        # Update metrics
        num_samples += targets.size(0)
        num_correct += (outputs == targets).sum()
    return (num_correct / num_samples * 100).item(), (loss / num_samples).item()

# training loop
def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: int,
    # for pruning
    weight_decay=5e-4,
    pruner=None,
    callbacks=None,
    save=None,
    save_only_state_dict=False,
) -> None:

    optimizer = torch.optim.SGD(model.parameters(
    ), lr=lr, momentum=0.9, weight_decay=weight_decay if pruner is None else 0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,80], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    best_acc = -1
    best_checkpoint = dict()

    for epoch in range(epochs):
        model.train()
        for inputs, targets in tqdm(train_loader, leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Reset the gradients (from the last iteration)
            optimizer.zero_grad()

            # Forward inference
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward propagation
            loss.backward()

            # Pruner regularize for sparsity learning
            if pruner is not None:
                pruner.regularize(model)

            # Update optimizer
            optimizer.step()

            if callbacks is not None:
                for callback in callbacks:
                    callback()

        acc, val_loss = evaluate(model, test_loader)
        print(
            f'Epoch {epoch + 1}/{epochs} | Val acc: {acc:.2f} | Val loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')

        if best_acc < acc:
            best_checkpoint['state_dict'] = copy.deepcopy(model.state_dict())
            best_acc = acc
        # Update LR scheduler
        scheduler.step()
    model.load_state_dict(best_checkpoint['state_dict'])
    if save:
        # on veut sauvegarder le meilleur modèle
        path = os.path.join(os.getcwd(), "results", save)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if save_only_state_dict:
            torch.save(model.state_dict(), path)
        else:
            torch.save(model, path)     
    print(f'Best val acc: {best_acc:.2f}')


# Pruner
# définir le nbre de classses => évite de pruner la dernière couche
def get_pruner(model, method, example_input, num_classes, global_pruning=False):
    sparsity_learning = False
    if method == "random":
        imp = tp.importance.RandomImportance()
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=global_pruning)
    elif method == "l1":
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=global_pruning)
    elif method == "lamp":
        imp = tp.importance.LAMPImportance(p=2)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=global_pruning)
    elif method == "slim":
        sparsity_learning = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=1e-5, global_pruning=global_pruning)
    elif method == "group_norm":
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=global_pruning)
    elif method == "group_sl":
        sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=1e-5, global_pruning=global_pruning)
    else:
        raise NotImplementedError

    num_classes = num_classes
    unwrapped_parameters = []
    ignored_layers = []
    ch_sparsity_dict = {}
    # ignore output layers
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == num_classes:
            ignored_layers.append(m)
        elif isinstance(m, torch.nn.modules.conv._ConvNd) and m.out_channels == num_classes:
            ignored_layers.append(m)
    
    # Here we fix iterative_steps=200 to prune the model progressively with small steps 
    # until the required speed up is achieved.
    pruner = pruner_entry(
        model,
        example_input,
        importance=imp,
        iterative_steps=200,
        ch_sparsity=1.0,
        ch_sparsity_dict=ch_sparsity_dict,
        max_ch_sparsity=1,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters,
    )
    return pruner

# pruning jusqu'à atteindre le speed up voulu
def progressive_pruning(pruner, model, speed_up, example_inputs):
    model.eval()
    base_ops, _ = tp.utils.count_ops_and_params(
        model, example_inputs=example_inputs)
    current_speed_up = 1
    while current_speed_up < speed_up:
        pruner.step(interactive=False)
        pruned_ops, _ = tp.utils.count_ops_and_params(
            model, example_inputs=example_inputs)
        current_speed_up = float(base_ops) / pruned_ops
        # print(current_speed_up)
    return current_speed_up


# pruning jusqu'à atteindre le speed up voulu + fine tuning après chaque étape de pruning
def progressive_pruning_fine_tuning(pruner, model, speed_up, train_loader, test_loader, example_inputs):
    model.eval()
    base_ops, _ = tp.utils.count_ops_and_params(
        model, example_inputs=example_inputs)
    current_speed_up = 1
    while current_speed_up < speed_up:
        pruner.step(interactive=False)
        pruned_ops, _ = tp.utils.count_ops_and_params(
            model, example_inputs=example_inputs)
        current_speed_up = float(base_ops) / pruned_ops
        # print(current_speed_up)
        # après avoir pruné : fine tuning
        train(model, train_loader,
              test_loader, epochs=5, lr=0.01)
    return current_speed_up


def optimize(model, train_loader, test_loader, speed_up, epochs_finetuning, method, num_classes, global_pruning, save_path):
    # Avant pruning
    # Get the size of the first image tensor in the first batch of the DataLoader
    images, labels = next(iter(train_loader))
    example_input = torch.rand(images[0].shape).unsqueeze(0).to(device)
    start_macs, start_params = tp.utils.count_ops_and_params(model, example_input)
    start_acc, start_loss = evaluate(model, test_loader)
    print('----- Avant pruning -----')
    print(f'Nombre de MACs = {start_macs/1e6:.3f} M')
    print(f'Nombre de paramètres = {start_params/1e6:.3f} M')
    print(f'Précision = {start_acc:.2f} %')
    print(f'Loss = {start_loss:.3f}')
    print('')


    speed_up = speed_up
    pruner = get_pruner(model, method, example_input, num_classes, global_pruning)
    progressive_pruning(pruner, model, speed_up, example_input)
    pruned_macs, pruned_params = tp.utils.count_ops_and_params(
        model, example_input)
    pruned_acc, pruned_loss = evaluate(model, test_loader)
    print('----- Après pruning -----')
    print(f'Nombre de MACs = {pruned_macs/1e6:.3f} M')
    print(f'Nombre de paramètres = {pruned_params/1e6:.3f} M')
    print(f'Précision = {pruned_acc:.2f} %')
    print(f'Loss = {pruned_loss:.3f}')
    print('')

    # Results
    print('----- Results before fine tuning -----')
    print(f'Params: {start_params/1e6:.2f} M => {pruned_params/1e6:.2f} M')
    print(f'MACs: {start_macs/1e6:.2f} M => {pruned_macs/1e6:.2f} M')
    print(f'Accuracy: {start_acc:.2f} % => {pruned_acc:.2f} %')
    print(f'Loss: {start_loss:.2f} => {pruned_loss:.2f}')
    print('')

    # Fine tuning
    print('----- Fine tuning -----')
    path = save_path
    train(model, train_loader, test_loader,
        epochs=epochs_finetuning, lr=0.01, save=path)

    # Post fine tuning
    end_macs, end_params = tp.utils.count_ops_and_params(model, example_input)
    end_acc, end_loss = evaluate(model, test_loader)
    print('----- Results after fine tuning -----')
    print(f'Params: {start_params/1e6:.2f} M => {end_params/1e6:.2f} M')
    print(f'MACs: {start_macs/1e6:.2f} M => {end_macs/1e6:.2f} M')
    print(f'Accuracy: {start_acc:.2f} % => {end_acc:.2f} %')
    print(f'Loss: {start_loss:.2f} => {end_loss:.2f}')

