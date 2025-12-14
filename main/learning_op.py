import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassPrecision, MulticlassRecall, MulticlassAccuracy, MulticlassF1Score
from tqdm import tqdm
from typing import Tuple, List, Any

from traffic_loader import TrafficDataset
from torch.utils.data import DataLoader
import pdb
from focal_loss import multi_class_analysis, FocalLoss

def evaluate_op(
        model: nn.Module,
        data_loader: DataLoader,
        n_class,
        device: str = 'cuda:0'
) -> List[Tuple[float, Any, Any, Any, Any]]:
    """ Perform evaluation.

    Args:
        model: Model instance.
        data_loader: Data loader in PyTorch.
        device: Device name/number for usage. The desired device of the parameters
                and buffers in this module.

    Returns:
        Task metrics
    """
    if not torch.cuda.is_available():
        print('Fail to use GPU')
        device = 'cpu'

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculation as it's unnecessary during evaluation
    pred = []
    with torch.no_grad():
        # Iterate through each batch in the data_loader with a progress bar for evaluation
        pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Evaluation")
        for batch_idx, (input, label) in pbar:
            input = input.to(device)
            scale = True
            if(scale):
                input *= 10
            a, b = input.shape
            input = input.reshape(a, 1, b)
            output = model(input)
            output = output.cpu()
            pred.append((output, label))

        
    # Initialize a list for task metrics
    task_metrics = []
    total_loss = 0.0
    total_batches = 0

    # Initialize metrics for multiclass
    prec_metric = MulticlassPrecision(average=None, num_classes=n_class)
    recall_metric = MulticlassRecall(average=None, num_classes=n_class)
    f1_metric = MulticlassF1Score(average=None, num_classes=n_class)
    accuracy_metric = MulticlassAccuracy(num_classes=n_class)

    # Calculate metrics for each output and label in the task
    for output, label in pred:
        # Calculate cross-entropy loss
        loss = F.cross_entropy(output, label)
        total_loss += loss.item()
        total_batches += 1

        # Update metrics
        prec_metric.update(torch.argmax(output, dim=1), label)
        recall_metric.update(torch.argmax(output, dim=1), label)
        f1_metric.update(torch.argmax(output, dim=1), label)
        accuracy_metric.update(torch.argmax(output, dim=1), label)

    # Compute average loss and metrics
    avg_loss = total_loss / total_batches
    avg_precision = prec_metric.compute().mean()
    avg_recall = recall_metric.compute().mean()
    avg_f1 = f1_metric.compute().mean()
    accuracy = accuracy_metric.compute().detach().cpu().numpy()

    # print("prec_metric: ", prec_metric.compute())
    # print("recall_metric: ", recall_metric.compute())
    # print("f1_metric: ", f1_metric.compute())
    # print("accuracy_metric: ", accuracy_metric.compute())
    # Append computed metrics to the task metrics list
    task_metrics.append((avg_loss, avg_precision, avg_recall, avg_f1, accuracy))

    for _, m in enumerate(task_metrics):
        print(f" Validation Loss: {m[0]:.4f}, "
                f"Precision: {m[1]:.4f}, Recall: {m[2]:.4f}, F1: {m[3]:.4f} , Accuracy: {m[4]:.4f}")

    return task_metrics


def train_op(
        model: nn.Module,
        batch_size: int = 512,
        n_epochs: int = 10,
        device: str = 'cuda:0',
        use_wandb: bool = False,
        n_class = 2
):
    """ Perform training iteration in PyTorch.

    Args:
        model: Model instance.
        batch_size: Batch size.
        n_epochs: Epochs.
        device: Device name/number for usage. The desired device of the parameters
                and buffers in this module.
        task_weights: Assign weights of loss calculation for multi-class classification.
        use_wandb: Enable wandb to record training log.

    Returns:

    """
    if not torch.cuda.is_available():
        print('Fail to use GPU')
        device = 'cpu'


    train_data_path = '/mnt/winter/UTFormer_COMNET/cross-platform/dataset/tensor_1/ios/train_data.pt'
    train_label_path = '/mnt/winter/UTFormer_COMNET/cross-platform/dataset/tensor_1/ios/train_label.pt'
    test_data_path = '/mnt/winter/UTFormer_COMNET/cross-platform/dataset/tensor_1/ios/test_data.pt'
    test_label_path = '/mnt/winter/UTFormer_COMNET/cross-platform/dataset/tensor_1/ios/test_label.pt'



    # Create DataLoader to load the data in batches
    train_dataset = TrafficDataset(train_data_path, train_label_path)
    val_dataset = TrafficDataset(test_data_path, test_label_path)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    # Move model parameters to specified devices
    model = model.to(device)

    # Create optimizer with gradient decent method for training
    # The author of `MTC` used weight decay to avoid over-fitting
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

    # Initialize accuracy to save best model
    best_accuracy = 0.0
    avg_loss = 0.0
    avg_val_loss = 0.0

    # Training loop by epoch
    for epoch in range(n_epochs):
        # Store
        running_loss = 0.0

        # Set training mode
        model.train()

        # Initialize progress bar
        pbar = tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc=f"Epoch {epoch + 1}, Loss: 0.000")

        # Loop training dataset
        for batch_idx, (input, label) in pbar:
            a, b = input.shape
            scale = True
            if(scale):
                input *= 10
            input = input.reshape(a, 1, b)
            
            # Move model parameters to specified devices
            input = input.to(device)
            label = label.to(device)
            # Initialize gradient
            optimizer.zero_grad()

            output = model(input)

            loss = F.cross_entropy(output, label)
            # loss = focal_loss(output, label)
            batch_loss = loss
            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.item()
            avg_loss = running_loss / (batch_idx + 1)

            # Update the description of progress bar with the average loss
            pbar.set_description(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
            pbar.set_postfix(loss=avg_loss)

        # Evaluation after each epoch
        metrics = evaluate_op(model, val_data_loader, n_class)

        # Log metrics
        task_accuracy = []
        #val_loss = 0.0
        for _, m in enumerate(metrics):
            # print(f" Validation Loss: {m[0]:.4f}, "
            #       f"Precision: {m[1]:.4f}, Recall: {m[2]:.4f}, F1: {m[3]:.4f} , Accuracy: {m[4]:.4f}")
            avg_val_loss = avg_val_loss + m[0]
            
            # Record task accuracy
            task_accuracy.append(m[4])

        

        # Save best model according to the accuracy of 'application'
        target_accuracy = task_accuracy[0]
        if target_accuracy >= best_accuracy:
            torch.save(model.state_dict(), './best/best_ios_80.pt')
            print('saving best model ...')
            # Update best accuracy
            best_accuracy = target_accuracy

        # Update scheduler to modify learning rate
        scheduler.step(avg_loss)

    avg_val_loss = avg_val_loss / n_epochs
    print("best accuracy: ", str(best_accuracy))

    return avg_val_loss

        
