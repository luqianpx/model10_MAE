import os
import model_run
from support_args import parse, fin_parse

# Load pretrained model configurations
try:
    args = fin_parse()
except FileNotFoundError:
    print("Error: Pretrained model configuration not found.")
    exit(1)

fine_tune_modes = ["nopretrain", "pretrain_full", "pretrain_only_classifier"]

for mode in fine_tune_modes:
    try:
        print(f"\nStarting Fine-Tuning with mode: {mode}")
        args.downstream_tr_type = mode
        model_run.finetune(args)
    except Exception as e:
        print(f"Error during fine-tuning ({mode}): {e}")
