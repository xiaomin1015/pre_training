import os
import numpy as np
import torch
from src.model import Adamw, LMModel
from training.train import estimate_loss, get_device, load_config, set_seed, train
from pathlib import Path

train_BIN_PATH = "./tokenizer/TinyStories-train.bin"
valid_BIN_PATH = "./tokenizer/TinyStories-valid.bin"
train_data = np.memmap(train_BIN_PATH,dtype=np.uint16, mode="r")
valid_data = np.memmap(valid_BIN_PATH,dtype=np.uint16, mode="r")


device = get_device()
print(device)

def get_model_from_checkPoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path,map_location=device,)
    config = load_config("configs/baseline.yaml")
    # model parameter
    vocab_size = config["model"]["vocab_size"]
    d_model = config["model"]["d_model"]
    num_layers = config["model"]["num_layers"]
    num_heads = config["model"]["num_heads"]
    context_length = config["model"]["context_length"]
    rope_theta = config["model"]["rope_theta"]
    d_ff = config["model"]["d_ff"]

    model = LMModel(vocab_size, d_model, num_layers, num_heads,context_length, rope_theta, d_ff)
    model.to(device)
    result = model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )
    # <All keys matched successfully>
    print(result)

    for name, param in model.named_parameters():
        if param.device != device:
            print("WRONG DEVICE:", name, param.device, device)


    print(
        "Total parameters:",
        sum(p.numel() for p in model.parameters())
    )

    return model

def get_OPT_from_checkPoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path,map_location=device,)
    config = load_config("configs/baseline.yaml")
    # training
    weight_decay = config["training"]["weight_decay"]
    eps = config["training"]["eps"]
    betas = config["training"]["betas"]
    learning_rate = config["training"]["learning_rate"]

    opt = Adamw(model.parameters(), learning_rate, betas, eps, weight_decay)
    opt.load_state_dict(checkpoint["optimizer"])
    #print("optimizer states:", len(opt.state))

    return opt

def verify_model_fromcheckPoint(model):
    config = load_config("configs/baseline.yaml")
    context_length = config["model"]["context_length"]
    # eval
    eval_iters = config["eval"]["eval_iters"]
    # training
    batch_size = config["training"]["batch_size"]

    model.eval()

    train_loss = estimate_loss(
        model,
        train_data,
        batch_size,
        context_length,
        eval_iters,
        device,
    )

    val_loss = estimate_loss(
        model,
        valid_data,
        batch_size,
        context_length,
        eval_iters,
        device,
    )

    print("restored train loss:", train_loss)
    print("restored val loss:", val_loss)



pre_iters = 4700
cosine_cycle_iters = 5001
max_iters = 9001

#checkpoint path
#!!make sure it has correct checkpoint_dir!!
checkpoint_dir="checkpoints/rmsnorm_weight/42/"
checkpoint_path = checkpoint_dir+"checkpoint_5000.pt"
if __name__ == "__main__":
    config = load_config("configs/baseline.yaml")
    context_length = config["model"]["context_length"]
    batch_size = config["training"]["batch_size"]
    warmup_iters = config["training"]["warmup_iters"]
    max_grad_norm = config["training"]["max_grad_norm"]
    # !!!!!!special parameter for continue training from checkingpoint
    start_iters = torch.load(checkpoint_path,map_location=device,)["iteration"] + 1
    cosine_cycle_iters = config["training"]["max_iters"]
    max_iters = 9001
    # eval
    eval_iters = config["eval"]["eval_iters"]
    eval_interval = config["eval"]["eval_interval"]
    checkingpoint_interval = config["eval"]["checkingpoint_interval"]

    # learning rate hyperparameter
    config = load_config("configs/hyperparamersweep.yaml")
    max_learning_rates = config["max_learning_rates"]
    max_learning_rate=max_learning_rates[0]
    min_learning_rate = 0.1*max_learning_rate
    # learning_rate would be overwrite with group["lr"] = lr
    model = get_model_from_checkPoint(checkpoint_path)
    verify_model_fromcheckPoint(model)
    opt = get_OPT_from_checkPoint(checkpoint_path)
    set_seed(12345)
    train(model, opt, train_data, valid_data, batch_size, context_length,
        max_iters, eval_interval, eval_iters,
        max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters,
        device, checkpoint_dir, checkingpoint_interval, max_grad_norm, start_iters)
   
