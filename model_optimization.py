import torch
import support_based as spb
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

def train(args, DA, model, optim, lr_scheduler):
    """Trains a Masked Autoencoder (MAE) using ECG data."""

    model.train()
    losses = []

    assert len(DA) > 0, "Error: Training dataset is empty!"

    for sample, label in tqdm(iter(DA)):
        optim.zero_grad()

        sample = sample.to(args.device)
        with torch.no_grad():  # Avoid unnecessary gradient calculations
            predicted_sample, mask = model(sample)

        loss = torch.mean((predicted_sample - sample) ** 2 * mask) / args.mask_ratio
        loss.backward()
        optim.step()

        losses.append(loss.item())

    lr_scheduler.step()
    avg_loss = sum(losses) / len(losses)
    return avg_loss

def finetune_train(args, loader, encoder, classifier, criterion, optimizer):
    """Fine-tunes the model for ECG classification."""

    loss_epoch = []
    met_epoch = []

    assert len(loader) > 0, "Error: Training dataset is empty!"

    encoder.train()
    classifier.train()
    for _, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.long().to(args.device)  # Use .long() instead of .type(torch.LongTensor)

        if args.downstream_tr_type == "pretrain_only_classifier":
            with torch.no_grad():
                h = encoder(x)
        else:
            h = encoder(x)

        output = classifier(h)
        loss = criterion(output, y)
        met = cal_me_tensor(output, y, args)

        loss.backward()
        optimizer.step()

        loss_epoch.append(loss.item())
        met_epoch.append(met)

    met_epoch = np.stack(met_epoch, axis=0)
    mean_met = np.mean(met_epoch, axis=0)
    mean_loss = sum(loss_epoch) / len(loss_epoch)

    return mean_loss, mean_met

def e_test(args, loader, encoder, classifier, criterion):
    """Evaluates the fine-tuned model on ECG classification task."""

    loss_epoch = []
    met_epoch = []

    assert len(loader) > 0, "Error: Test dataset is empty!"

    encoder.eval()
    classifier.eval()
    with torch.no_grad():  # Ensure no gradients are computed during evaluation
        for _, (x, y) in enumerate(loader):
            x = x.to(args.device)
            y = y.long().to(args.device)

            h = encoder(x)
            output = classifier(h)
            loss = criterion(output, y)
            met = cal_me_tensor(output, y, args)

            loss_epoch.append(loss.item())
            met_epoch.append(met)

    met_epoch = np.stack(met_epoch, axis=0)
    mean_met = np.mean(met_epoch, axis=0)
    mean_loss = sum(loss_epoch) / len(loss_epoch)

    return mean_loss, mean_met

def cal_me_tensor(output, y, args):
    """Computes evaluation metrics for classification."""

    output = output.detach().cpu().numpy()
    y = y.cpu().numpy()
    return spb.cal_met(output, y, args)
