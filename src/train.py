from sklearn.metrics import accuracy_score

import torch
from tqdm import tqdm


def train_one_epoch(model, train_dataloader, criterion, optimizer, device="cuda:0"):
    model.to(device).train()
    with tqdm(total=len(train_dataloader)) as pbar:
        for batch in train_dataloader:
            input_ids1 = batch['input_ids1'].to(device)
            attention_mask1 = batch['attention_mask1'].to(device)
            input_ids2 = batch['input_ids2'].to(device)
            attention_mask2 = batch['attention_mask2'].to(device)
            labels = batch['labels'].view(-1).to(device)

            optimizer.zero_grad()
            output = model.forward(x_input_ids=input_ids1, x_segment_ids=None, x_input_mask=attention_mask1,
                                   y_input_ids=input_ids2, y_segment_ids=None, y_input_mask=attention_mask2,
                                   labels=labels)

            _, predicted = torch.max(output, 1)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(output.detach(), 1)
            accuracy = accuracy_score(predicted.cpu().numpy(), labels.cpu().numpy())
            pbar.set_description('Loss: {:.4f}; Accuracy: {:.4f}'.format(loss.item(), accuracy))
            pbar.update(1)


def predict(model, val_dataloader, criterion, device="cuda:0"):
    model.to(device).eval()
    losses = []
    predicted_classes = []
    true_classes = []
    with tqdm(total=len(val_dataloader)) as pbar:
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids1 = batch['input_ids1'].to(device)
                attention_mask1 = batch['attention_mask1'].to(device)
                input_ids2 = batch['input_ids2'].to(device)
                attention_mask2 = batch['attention_mask2'].to(device)
                labels = batch['labels'].view(-1).to(device)

                output = model.forward(x_input_ids=input_ids1, x_segment_ids=None, x_input_mask=attention_mask1,
                                       y_input_ids=input_ids2, y_segment_ids=None, y_input_mask=attention_mask2,
                                       labels=labels)
                _, predicted = torch.max(output, 1)

                loss = criterion(output, labels)
                losses.append(loss.item())
                _, predicted = torch.max(output.detach(), 1)
                predicted_classes.append(predicted)
                true_classes.append(labels)

                accuracy_mae = accuracy_score(predicted.cpu().numpy(), labels.cpu().numpy())
                pbar.set_description('Loss: {:.4f}; Accuracy_MAE: {:.4f}'.format(loss.item(), accuracy_mae))
                pbar.update(1)

    predicted_classes = torch.cat(predicted_classes).detach().to('cpu').numpy()
    true_classes = torch.cat(true_classes).detach().to('cpu').numpy()
    return losses, predicted_classes, true_classes


def train(model, train_dataloader, val_dataloader, criterion, optimizer, device="cuda:0", n_epochs=10, scheduler=None):
    model.to(device)
    for epoch in range(n_epochs):
        print('Learning rate: ', optimizer.param_groups[0]['lr'])
        print('Epoc:', epoch)
        train_one_epoch(model, train_dataloader, criterion, optimizer)
        print('Validation')
        losses, predicted_classes, true_classes = predict(model, val_dataloader, criterion)
        print('Accuracy_MAE: ', accuracy_score(true_classes, predicted_classes))
        scheduler.step()
