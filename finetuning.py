import numpy as np
import torch
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import torch.optim as optim


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(model, train_loader, loss_fct, optimizer):
    model.train()
    all_loss = []
    for batch_idx, data in enumerate(train_loader):  # Iterate in batches over the training dataset.
        data.to(DEVICE)
        out = model(data.x_dict, data.edge_index_dict, data.batch_dict)  # Perform a single forward pass.
        loss = loss_fct(out, data['article'].y)  # Compute the loss.
        all_loss.append(loss.item())
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
    return model, sum(all_loss)/len(all_loss)


def eval_model(model, test_loader, print_classification_report=False):
    model.eval()
    correct = 0
    true_y = []
    pred_y = []
    for data in test_loader:  # Iterate in batches over the training/test dataset.
        data.to(DEVICE)
        out = model(data.x_dict, data.edge_index_dict, data.batch_dict)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        pred_y.append(pred.cpu().detach().numpy())
        correct += int((pred == data['article'].y).sum())  # Check against ground-truth labels.
        true_y.append(data['article'].y.cpu().detach().numpy())
    if print_classification_report:
        print(classification_report(np.concatenate(true_y), np.concatenate(pred_y), digits=5))
    return accuracy_score(np.concatenate(true_y), np.concatenate(pred_y)), precision_score(np.concatenate(true_y), np.concatenate(pred_y), average='macro'), recall_score(np.concatenate(true_y), np.concatenate(pred_y), average='macro'), f1_score(np.concatenate(true_y), np.concatenate(pred_y), average='macro')


def train_eval_model(model, train_loader, test_loader, loss_fct, optimizer, num_epochs=1, verbose=1, use_lr_scheduler=True):
    model.to(DEVICE)
    if use_lr_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    for epoch in range(1, num_epochs+1):
        model, loss = train_model(model=model, train_loader=train_loader, loss_fct=loss_fct, optimizer=optimizer)
        if use_lr_scheduler:
            scheduler.step()
        train_acc, train_p, train_r, train_f1 = eval_model(model, train_loader)
        if epoch == num_epochs:
            test_acc, test_p, test_r, test_f1 = eval_model(model, test_loader, print_classification_report=True)
            return model, test_acc, test_p, test_r, test_f1
        else:
            test_acc, test_p, test_r, test_f1 = eval_model(model, test_loader)
            if verbose == 1:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}')
