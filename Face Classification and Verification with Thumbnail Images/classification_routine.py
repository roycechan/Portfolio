import time
import torch.nn as nn
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import config, utils, data, paths

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion_label = nn.CrossEntropyLoss()

def train(model, data_loader, test_loader, last_checkpoint=0):
    max_epoch = config.max_epoch
    optimizer_label = torch.optim.Adam(model.parameters(), lr=config.lr1, weight_decay=config.weight_decay,
                                       amsgrad=True)
    start_epoch = last_checkpoint

    model.train()
    model.to(device)
    model = nn.DataParallel(model)

    print("Starting training from Epoch", start_epoch + 1)

    for epoch in range(max_epoch):
        current_epoch = start_epoch + epoch + 1
        start_time = time.time()

        avg_loss = 0.0
        for batch_num, (feats, labels) in enumerate(data_loader):
            feats, labels = feats.to(device), labels.to(device)

            outputs = model(feats)

            loss = criterion_label(outputs, labels.long())
            optimizer_label.zero_grad()
            loss.backward()

            optimizer_label.step()
            avg_loss += loss.item()

            if batch_num % 500 == 499:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(current_epoch, batch_num + 1, avg_loss / 500))
                avg_loss = 0.0

            torch.cuda.empty_cache()
            del feats
            del labels
            del loss

        val_loss, val_acc = test_classify(model, test_loader)
        train_loss, train_acc = test_classify(model, data_loader)
        end_time = time.time()

        print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.
              format(train_loss, train_acc, val_loss, val_acc))
        print('Time taken:', end_time - start_time, "s")

        utils.save_checkpoint(model, current_epoch)


def test_classify(model, test_loader):
    with torch.no_grad():
        model.eval()
        model.to(device)
        model = torch.nn.DataParallel(model)

        test_loss = []
        accuracy = 0
        total = 0

        for batch_num, (feats, labels) in enumerate(test_loader):
            feats, labels = feats.to(device), labels.to(device)
            outputs = model(feats)

            _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
            pred_labels = pred_labels.view(-1)

            loss = criterion_label(outputs, labels.long())

            accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            test_loss.extend([loss.item()] * feats.size()[0])
            del feats
            del labels

        model.train()
        return np.mean(test_loss), accuracy / total


def predict(model, test_dataset):
    '''
    :param model: current model
    :param test_dataset: test data containing test images transformed to tensors
    :return: pandas dataframe containing ids and predicted labels
    '''

    with torch.no_grad():
        model.eval()
        model.to(device)
        model = torch.nn.DataParallel(model)

        pred_labels_list = []  # store predictions

        for feats in test_dataset:
            feats = feats.to(device)
            outputs = model(feats)

            _, pred_class = torch.max(F.softmax(outputs, dim=1), 1)
            pred_class = pred_class.view(-1).cpu().numpy()
            # convert ImageFolder classes back to correct labels
            # Ref: https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
            # get key from value in train_dataset.class_to_idx
            pred_labels = list(data.train_dataset.class_to_idx.keys())[
                list(data.train_dataset.class_to_idx.values()).index(pred_class)]
            pred_labels_list.append(pred_labels)

        # df for print to CSV
        df = pd.DataFrame(data=pred_labels_list)
        df.columns = ['label']
        df.to_csv(paths.output_classification, index_label='id')
        return df
