import torch


def accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_acc = 0

    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(input_ids)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy(logits, labels)

    return total_loss / len(loader), total_acc / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            total_acc += accuracy(logits, labels)

    return total_loss / len(loader), total_acc / len(loader)