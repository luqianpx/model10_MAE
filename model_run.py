import os
import torch
import math
import support_based as spb
from support_based import com_mul_str
from support_dataset import ECG_dataset
from support_model import MAE_ViT, ViT_Classifier, save_model, load_model, MLP_Classifier
from model_optimization import train, finetune_train, e_test

# Pretrain the Foundation Model
def pretrain(args):
    """Pretrains the ECG foundation model using MAE-ViT."""

    # Device Configuration
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()

    # Load Dataset
    DA = ECG_dataset(args)
    assert len(DA) > 0, "Error: Dataset is empty!"

    # Save Directory
    sa_fo = os.path.join('./save', com_mul_str(args))
    os.makedirs(sa_fo, exist_ok=True)

    # Initialize Model
    model = MAE_ViT(args).to(args.device)

    # Optimizer & Learning Rate Scheduler
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.epochs * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    # Training Loop
    min_loss = float('inf')
    loss_li = []
    for e in range(args.epochs):
        av_loss = train(args, DA, model, optim, lr_scheduler)
        loss_li.append(av_loss)
        print(f"Epoch {e}: Loss = {av_loss}")

        # Save best model
        if av_loss < min_loss:
            min_loss = av_loss
            save_model(sa_fo, 'pretrain-main', model)

        spb.save_res(sa_fo, 'pretrain-res', [loss_li, args])

def finetune(args):
    """Fine-tunes the pretrained model for ECG classification."""

    # Device Configuration
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()

    # Save Directory
    sa_fo = os.path.join('./save', args.sa_folder)

    # Load Dataset
    args.dataset_type = 'finetune'
    tr_DA = ECG_dataset(args)
    args.dataset_type = 'test'
    te_DA = ECG_dataset(args)

    # Load Model
    mae_model = MAE_ViT(args).to(args.device)
    pretrain_path = os.path.join(sa_fo, 'pretrain-main.pth')
    if not args.downstream_tr_type == 'nopretrain' and os.path.exists(pretrain_path):
        load_model(sa_fo, 'pretrain-main', mae_model)
    else:
        print("Warning: No pretrained model found, training from scratch.")

    # Define Classifier
    model = ViT_Classifier(mae_model.encoder, args.n_class).to(args.device)
    classifier = MLP_Classifier(args.emb_dim, args.n_class).to(args.device)

    # Optimizer Selection
    if args.downstream_tr_type == "pretrain_only_classifier":
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=3e-4)
    else:
        optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params': classifier.parameters()}], lr=3e-4)

    criterion = torch.nn.CrossEntropyLoss()

    # Training Loop
    highest_F1 = 0.1
    los_met_li = [[], [], args]
    for epoch in range(args.logistic_epochs):
        mean_loss_train, mean_met_train = finetune_train(args, tr_DA, model, classifier, criterion, optimizer)
        print(f"Epoch {epoch} - Train Loss: {mean_loss_train}, Train Metrics: {mean_met_train}")

        # Validation
        mean_loss_val, mean_met_val = e_test(args, te_DA, model, classifier, criterion)
        print(f"Epoch {epoch} - Validation Loss: {mean_loss_val}, Validation Metrics: {mean_met_val}")

        # Save best model
        if mean_met_val[3] > highest_F1:
            save_model(sa_fo, f'finetune-{args.downstream_tr_type}-main', model)
            save_model(sa_fo, f'finetune-{args.downstream_tr_type}-classifier', classifier)
            highest_F1 = mean_met_val[3]

        spb.save_res(sa_fo, f'finetune-{args.downstream_tr_type}-res', los_met_li)
