import torch

from src.model import data_loading
from training.load_checkingPoint import get_model_from_checkPoint, verify_model_fromcheckPoint
from training.train import get_device, load_config, set_seed
import numpy as np
device = get_device()
print(device)

def Load_checkPoint():
    checkpoint_path = "checkpoints/rmsnorm_no_weight/42/checkpoint_6000.pt"
    print(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # dict_keys(['model', 'optimizer', 'iteration'])
    # print(checkpoint.get('iteration'))
    
    # d_model = 512
    # d_ff = 1344
    # mha W_Q W_K W_V W_O
    # sqrt(2/(d_model+d_model)) = 0.04419417382
    # SwiGLU
    # sqrt(2/(d_model+d_ff)) = 0.03282660821
    model_state_dict = checkpoint.get('model')
    for name, param in model_state_dict.items():
        if param.is_floating_point():
            print(
                f"{name:25s} "
                f"mean= {param.mean():.6f} "
                f"std= {param.std():.3f} "
                f"min= {param.min():.2f} "
                f"max= {param.max():.2f} "
                f"norm= {param.norm():.0f}"
            )

def load_compare():
    baseline = torch.load("checkpoints_archived/checkpoint_0.pt", map_location=device)["model"]
    trained = torch.load("checkpoints_archived/checkpoint_9000.pt", map_location=device)["model"]

    for name in baseline:
        if name in trained and baseline[name].is_floating_point():
            w1 = baseline[name]
            w2 = trained[name]

            relative_meanchange = (w2 - w1).mean() / w1.mean()
            relative_normchange = (w2 - w1).norm() / w1.norm()
            relative_stdchange = (w2 - w1).std() / w1.std()

            print(
                f"{name:25s} "
                f"relative mean: {relative_meanchange:.3f}      norm: {relative_normchange:.3f}    std: {relative_stdchange:.3f}"
            )

def getAvtivation(checkpoint_path):

    model = get_model_from_checkPoint(checkpoint_path)
    #verify_model_fromcheckPoint(model)
    model.eval()
    # get statistics
    for layer in model.layers:
        layer.ln1.collect_stats = True
        layer.ln2.collect_stats = True

    model.ln_final.collect_stats = True

    with torch.no_grad():
        logits = model(x)

    for i, layer in enumerate(model.layers):
        
        print(
            f"Layer {i} LN1: "
            f"before={layer.ln1.rms_before:.4f}, "
            f"after={layer.ln1.rms_after:.4f}"
            f"Layer {i} LN2: "
            f"before={layer.ln2.rms_before:.4f}, "
            f"after={layer.ln2.rms_after:.4f}"
        )

if __name__ == "__main__":

    set_seed(456)
    config = load_config("configs/baseline.yaml")
    context_length = config["model"]["context_length"]
    batch_size = config["training"]["batch_size"]
    train_BIN_PATH = "./tokenizer/TinyStories-train.bin"
    train_data = np.memmap(train_BIN_PATH,dtype=np.uint16, mode="r")
    x, y = data_loading(
        train_data,
        batch_size,
        context_length,
        device,
    )
  
    checkpoint_dir="checkpoints/rmsnorm_no_weight/42/"
    checkpoint_path1 = checkpoint_dir+"checkpoint_6000.pt"
    getAvtivation(checkpoint_path1)
    
    #Load_checkPoint()