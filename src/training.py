
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

_cross_entropy_loss = nn.CrossEntropyLoss()


def train_loop(train_dataloader, model, optimizer, binary=False, val_dataloader = None, num_epochs = 10, verbose = False, writer: SummaryWriter | None = None):
    model = model.to(device)
    
    train_loss_history = []
    train_accuracy_history = []

    test_loss_history = []
    test_accuracy_history = []

    for epoch in range(num_epochs):
        epoch_loss, epoch_accuracy = train_one_epoch(train_dataloader, model, optimizer, binary)

        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train loss: {epoch_loss:>7f}, accuracy: {100*epoch_accuracy:>0.2f}%")

        if writer:
            writer.add_scalar("Train loss", epoch_loss, epoch)
            writer.add_scalar("Train accuracy", epoch_accuracy, epoch)

        train_loss_history.append(epoch_loss)
        train_accuracy_history.append(epoch_accuracy)

        if val_dataloader:
            epoch_loss, epoch_accuracy = test_model(val_dataloader, model, verbose=False, binary=binary)

            if verbose:
                print(f"Validation loss:  {epoch_loss:>7f}, accuracy: {100*epoch_accuracy:>0.2f}%")

            if writer:
                writer.add_scalar("Validation loss", epoch_loss, epoch)
                writer.add_scalar("Validation accuracy", epoch_accuracy, epoch)

            test_loss_history.append(epoch_loss)
            test_accuracy_history.append(epoch_accuracy)
        
        if verbose:
            print("-----------------------------")
        

    history = {'train_loss': train_loss_history, 'train_accuracy': train_accuracy_history}

    if val_dataloader:
        history['val_loss'] = test_loss_history
        history['val_accuracy'] = test_accuracy_history

    return history



def train_one_epoch(dataloader, model, optimizer, binary=False):
    model.train()
    epoch_loss = 0
    correct = 0

    if binary:
        loss_fn = nn.functional.binary_cross_entropy_with_logits
        pred_fn = lambda y_hat: y_hat > 0
    else:
        loss_fn = nn.functional.cross_entropy
        pred_fn = lambda y_hat: torch.argmax(y_hat, dim = 1)

    for (X, y) in dataloader:
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)
        out = model(X)
        loss = loss_fn(out, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = pred_fn(out)
        correct += (pred == y).sum().item()
        epoch_loss += loss.item() * y.shape[0]
        

    size = len(dataloader.dataset)

    return epoch_loss/size, correct/size



def test_model(dataloader, model, verbose = True, binary=False):
    size = len(dataloader.dataset)
    model.eval()

    if binary:
        loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
        pred_fn = lambda y_hat: y_hat > 0
    else:
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        pred_fn = lambda y_hat: torch.argmax(y_hat, dim = 1)

    test_loss, correct = 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            out = model(X)
            test_loss += loss_fn(out, y).item()
            pred = pred_fn(out)
            correct += (pred == y).sum().item()

        if verbose:
            print(f"Test Error: Accuracy: {(100*correct/size):>0.2f}%, Avg loss: {test_loss/size:>8f}")

    if not verbose:
        return test_loss/size, correct/size
    
