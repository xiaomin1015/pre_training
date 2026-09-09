import os
import random
import numpy as np
import torch
from torch import nn
import time
import csv
from src.model import Adamw, LMModel, cross_entropy, data_loading, gradient_clipping, learning_rate_schedule, save_checkpoint, save_model_checkpoint
import math
import yaml
from pathlib import Path

def load_config(path):
	with open(path, "r") as f:
		return yaml.safe_load(f)

def train(
		model: LMModel,
		optimizer: Adamw,
		train_data: np.memmap,
		val_data: np.memmap,
		batch_size: int,
		context_length: int,
		max_iters: int,
		eval_interval: int,
		eval_iters: int,
		max_learning_rate: float,
		min_learning_rate: float,
		warmup_iters: int,
		cosine_cycle_iters: int,
		device: torch.device,
		checkpoint_dir: str | os.PathLike | None = None,
		checkingpoint_interval: int| None = None,
		max_grad_norm: float | None = None,
		start_iters = 0,
	):
	tracker = Tracker("experiment.csv")
	#for name, param in model.named_parameters():
		#print(name, param.device)

	# Put this model into training mode
	model.train()

	for iteration in range(start_iters, max_iters):
		# 1. Get a training batch
		x, y = data_loading(
			train_data,
			batch_size,
			context_length,
			device,
		)

		# 2. Forward pass, call model's forward() method.
		logits = model(x)

		# logits:
		# (batch_size, context_length, vocab_size)
		# y:
		# (batch_size, context_length)
		loss = cross_entropy(
			logits.reshape(-1, logits.shape[-1]),
			y.reshape(-1),
		)

		lr = learning_rate_schedule(iteration, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)

		for group in optimizer.param_groups:
			group["lr"] = lr

		# 3. Backward pass
		optimizer.zero_grad()
		loss.backward()

		# 4. Gradient clipping
		if max_grad_norm is not None:
			gradient_clipping(model.parameters(),max_grad_norm)

		# 5. Update parameters
		optimizer.step()

		# 6. Logging / evaluation
		if iteration % eval_interval == 0:
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
				val_data,
				batch_size,
				context_length,
				eval_iters,
				device,
			)
			tracker.log(iteration,train_loss,val_loss,)
			if (
				train_loss is None
				or val_loss is None
				or not math.isfinite(train_loss)
				or not math.isfinite(val_loss)
				or train_loss > 50.0
			):
				print(f"Unstable at max_lr={max_learning_rate}. " "Stopping LR sweep.")
				break

		# 7. Save checkpoint
		if iteration == max_iters-1 or iteration % checkingpoint_interval == 0:
			if checkpoint_dir is not None:
				checkpoint_path = f"{checkpoint_dir}/checkpoint_{iteration}.pt"
				if iteration == max_iters-1:
					save_checkpoint(model, optimizer, iteration, checkpoint_path)
				else:
					save_model_checkpoint(model,iteration,checkpoint_path,)


def estimate_loss(
	model: nn.Module,
	dataset: np.ndarray,
	batch_size: int,
	context_length: int,
	eval_iters: int,
	device: torch.device,
):
	model.eval()
	total_loss = 0.0

	with torch.no_grad():
		for _ in range(eval_iters):
			x, y = data_loading(dataset,batch_size,context_length,device)
			logits = model(x)
			loss = cross_entropy(logits.reshape(-1, logits.shape[-1]),y.reshape(-1))
			total_loss += loss.item()

	model.train()
	return total_loss / eval_iters


class Tracker:
	def __init__(self, log_path: str):
		self.log_path = log_path
		self.start_time = time.perf_counter()
		with open(self.log_path, "w", newline="") as f:
			writer = csv.writer(f)
			writer.writerow(["step","wall_time","train_loss","val_loss",])

	def log(self,step: int,train_loss: float,val_loss: float):
		wall_time = time.perf_counter() - self.start_time
		with open(self.log_path, "a", newline="") as f:
			writer = csv.writer(f)
			writer.writerow([
				step,
				wall_time,
				train_loss,
				val_loss,
			])
		print(
			f"step {step:6d} | "
			f"time {wall_time:8.2f}s | "
			f"train loss {train_loss:.4f} | "
			f"val loss {val_loss:.4f}"
		)



train_BIN_PATH = "./tokenizer/TinyStories-train.bin"
valid_BIN_PATH = "./tokenizer/TinyStories-valid.bin"
train_data = np.memmap(train_BIN_PATH,dtype=np.uint16, mode="r")
valid_data = np.memmap(valid_BIN_PATH,dtype=np.uint16, mode="r")


os.makedirs("checkpoints", exist_ok=True)
checkpoint_dir = "checkpoints/"

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def get_device():
	if torch.cuda.is_available():
		return torch.device("cuda:0")
	elif torch.backends.mps.is_available():
		return torch.device("mps:0")
	else:
		return torch.device("cpu")
	
if __name__ == "__main__":
	device = get_device()
	print(device)
	config = load_config("configs/baseline.yaml")
	# model parameter
	# paras: 23m, 0.085GB
	vocab_size = config["model"]["vocab_size"]
	d_model = config["model"]["d_model"]
	num_layers = config["model"]["num_layers"]
	num_heads = config["model"]["num_heads"]
	context_length = config["model"]["context_length"]
	rope_theta = config["model"]["rope_theta"]
	d_ff = config["model"]["d_ff"]
	# training
	batch_size = config["training"]["batch_size"]
	weight_decay = config["training"]["weight_decay"]
	eps = config["training"]["eps"]
	warmup_iters = config["training"]["warmup_iters"]
	max_grad_norm = config["training"]["max_grad_norm"]
	max_iters = config["training"]["max_iters"]
	betas = config["training"]["betas"]
	learning_rate = config["training"]["learning_rate"]
	cosine_cycle_iters= max_iters

	# eval
	eval_interval = config["eval"]["eval_interval"]
	eval_iters = config["eval"]["eval_iters"]
	checkingpoint_interval = config["eval"]["checkingpoint_interval"]
	# learning rate hyperparameter
	config = load_config("configs/hyperparamersweep.yaml")
	max_learning_rates = config["max_learning_rates"]

	#checkpoint path
	experiment_name = "rmsnorm_no_weight/"
	checkpoint_sub_dir = Path(checkpoint_dir+experiment_name)
	checkpoint_sub_dir.mkdir(parents=True, exist_ok=True)
	max_learning_rate = max_learning_rates[0]
	print(f"max learning rate: {max_learning_rate}")

	seeds = [456]
	for seed in seeds:
		set_seed(seed)
		print(f"seed: {seed}")
		model = LMModel(vocab_size, d_model, num_layers, num_heads, context_length, rope_theta, d_ff)
		model.to(device)
		
		min_learning_rate = 0.1*max_learning_rate
		# learning_rate would be overwrite with group["lr"] = lr
		# checking point 
		checkpoint_sub_lr_dir = checkpoint_sub_dir/str(seed)
		checkpoint_sub_lr_dir.mkdir(parents=True, exist_ok=True)
		opt = Adamw(model.parameters(), learning_rate, betas, eps, weight_decay)

		train(model, opt, train_data, valid_data, batch_size, context_length,
			max_iters, eval_interval, eval_iters,
			max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters,
			device, checkpoint_sub_lr_dir, checkingpoint_interval, max_grad_norm)
