import argparse
import os
import support_based as spb

def parse(args=None, test=None):
    """Parses command-line arguments for dataset, model, training, and finetuning options."""
    parser = argparse.ArgumentParser()

    # Dataset options
    parser.add_argument('--n_class', type=int, default=6, help="Number of output classes.")
    parser.add_argument('--n_channel', type=int, default=12, help="Number of input ECG channels.")
    parser.add_argument('--n_length', type=int, default=5000, help="Length of input sequence.")
    parser.add_argument('--whe_mix_lead', type=str, default='nomix', choices=['mix', 'nomix'], help="Whether to mix leads.")

    # Model options
    parser.add_argument('--patch_size', type=int, default=25, help="Patch size for embedding.")
    parser.add_argument('--emb_dim', type=int, default=128, help="Embedding dimension.")
    parser.add_argument('--encoder_layer', type=int, default=12, help="Number of encoder layers.")
    parser.add_argument('--encoder_head', type=int, default=4, help="Number of encoder heads.")
    parser.add_argument('--decoder_layer', type=int, default=4, help="Number of decoder layers.")
    parser.add_argument('--decoder_head', type=int, default=4, help="Number of decoder heads.")
    parser.add_argument('--mask_ratio', type=float, default=0.75, help="Masking ratio for input sequences.")

    # Training options
    parser.add_argument('--lr', type=float, default=1.5e-5, help="Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=0.05, help="Weight decay factor.")
    parser.add_argument('--warmup_epoch', type=int, default=5, help="Number of warmup epochs.")

    # Finetuning save model folder
    parser.add_argument('--sa_folder', type=str, default="nomix_25_5_2500_c3wD", help="Folder to save finetuned model.")

    # Finetuning options
    parser.add_argument('--dataset_type', type=str, default='pretrain', choices=['pretrain', 'finetune', 'test'], help="Type of dataset.")
    parser.add_argument('--downstream_tr_type', type=str, default='pretrain_only_classifier', choices=['nopretrain', 'pretrain_full', 'pretrain_only_classifier'], help="Downstream training type.")
    parser.add_argument('--labelled_ratio', type=float, default=1.0, help="Percentage of labelled data used for finetuning.")

    # Adjust settings based on environment (HPC or local)
    if os.path.isdir('/share/home/hulianting/Project/Project20_ECG_foundation_model/') or test == 'ser':
        parser.add_argument('--batch_size', type=int, default=2048, help="Batch size for training.")
        parser.add_argument('--start_epoch', type=int, default=0, help="Starting epoch.")
        parser.add_argument('--epochs', type=int, default=1500, help="Total number of epochs.")
        parser.add_argument('--running_env', type=str, default='HPC', help="Running environment.")

        # Logistic regression options
        parser.add_argument('--logistic_batch_size', type=int, default=256, help="Batch size for logistic regression training.")
        parser.add_argument('--logistic_epochs', type=int, default=100, help="Number of epochs for logistic regression.")
    else:
        parser.add_argument('--batch_size', type=int, default=5, help="Batch size for training (local mode).")
        parser.add_argument('--start_epoch', type=int, default=0, help="Starting epoch.")
        parser.add_argument('--epochs', type=int, default=25, help="Total number of epochs (local mode).")
        parser.add_argument('--running_env', type=str, default='local', help="Running environment.")

        # Logistic regression options
        parser.add_argument('--logistic_batch_size', type=int, default=5, help="Batch size for logistic regression training.")
        parser.add_argument('--logistic_epochs', type=int, default=20, help="Number of epochs for logistic regression.")

    return parser.parse_args(args)

def fin_parse():
    """Loads pretrained model configurations and updates finetuning parameters."""
    args = parse()

    res_path = os.path.join('./save/', args.sa_folder, 'prettain-res')
    if not os.path.exists(res_path):
        raise FileNotFoundError(f"Pretrained results file not found at {res_path}")

    args1 = spb.read_res(res_path)[1]  # Load pretrained results
    args1.sa_folder = args.sa_folder
    args1.logistic_batch_size = args.logistic_batch_size
    args1.labelled_ratio = args.labelled_ratio

    return args1
