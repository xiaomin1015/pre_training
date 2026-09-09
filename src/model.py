from collections.abc import Sequence
from typing import Callable, Iterable, Optional
import typing

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, einsum
import math
from jaxtyping import Bool, Float, Int
import os

def initialize_weight(shape: Sequence[int | torch.SymInt], mean: float, std: float, a: float, b: float, device, dtype) -> nn.Parameter:
	weight = nn.Parameter(torch.zeros(shape, dtype=dtype, device=device))
	torch.nn.init.trunc_normal_(weight, mean=mean, std=std, a = a, b = b)
	return weight

"""
	Args: 
		in_features: int, out_features: int
"""
class Linear(nn.Module):
	def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		std = math.sqrt(2/(in_features + out_features))
		# a Xavier/Glorot-style initialization
		self.weight = initialize_weight([out_features, in_features], 0.0, std, -3*std, 3*std, device, dtype)

	def assign_weight(self, weight: torch.Tensor):
		assert weight.shape == (self.out_features, self.in_features), f"expected shape ({self.out_features, self.in_features}), got {tuple(weight.shape)}"
		assert not torch.isnan(weight).any(), "weight contains NaNs"
		self.weight = nn.Parameter(weight)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		assert x.shape[-1] == self.in_features, f"expected last dim {self.in_features}, got {x.shape[-1]}"
		return x.matmul(self.weight.T)


"""
	Args:
		num_embeddings: int, embedding_dim: int
"""
class Embedding(nn.Module):
	def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
		super().__init__()
		self.num_embeddings = num_embeddings
		self.embedding_dim = embedding_dim
		self.embedding = initialize_weight([self.num_embeddings, self.embedding_dim], 0.0, 1.0, -3, 3, device, dtype)


	def assign_weight(self, weight: torch.Tensor):
		assert weight.shape == (self.num_embeddings, self.embedding_dim), f"expected shape ({self.num_embeddings, self.embedding_dim}), got {tuple(weight.shape)}"
		assert not torch.isnan(weight).any(), "weight contains NaNs"
		self.embedding = nn.Parameter(weight)
		
	def forward(self, in_indices: Int[torch.Tensor, " batch_size sequence_length"]) -> torch.Tensor:
		return self.embedding[in_indices]


""" Root Mean Square Layer Normalization
	Args:
		d_model: int, eps: float=1e-5
"""
class RMSNorm(nn.Module):
	def __init__(self, d_model: int, eps: float=1e-5, device=None, dtype=None):
		super().__init__()
		self.d_model = d_model
		self.eps = eps
		#self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
		self.collect_stats = False
		self.rms_before = None
		self.rms_after = None
	
	def assign_weight(self, weight: torch.Tensor):
		assert weight.shape[0] == self.d_model, f"expected last dim {self.d_model}, got {weight.shape[-1]}"
		assert not torch.isnan(weight).any(), "weight contains NaNs"
		self.weight = nn.Parameter(weight)

	# x: input tensor of shape (batch_size, sequence_length, d_model)
	# Apple's MPS backend does not support float64 (torch.float64) tensors
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		assert x.shape[-1] == self.d_model, f"expected last dim {self.d_model}, got {x.shape[-1]}"
		original_dtype = x.dtype 
		x = x.to(torch.float32)
		#norm = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True)/self.d_model + self.eps)
		#return (x/norm*self.weight).to(original_dtype) 
		# return (x/norm).to(original_dtype)
		
		# Activation RMS before RMSNorm
		if self.collect_stats:
			self.rms_before = torch.mean(x ** 2, dim=-1).mean().item() ** 0.5 
		norm = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True)/self.d_model + self.eps)
		output = x / norm
		#output = output * self.weight
		# Activation RMS after RMSNorm
		if self.collect_stats:
			self.rms_after = torch.mean(output ** 2, dim=-1).mean().item() ** 0.5  # sqrt applied once, at the end
		return output.to(original_dtype)

"""	Activation: linear + non_linear + linear
	W1, W2, W3
	Args:
		d_model: int, d_ff: int(optinal)
"""
class SwiGLU(nn.Module):
	def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
		super().__init__()
		self.d_model = d_model
		if d_ff is None:
			self.d_ff = round((8/3 * self.d_model / 64) * 64)
		else:
			self.d_ff = d_ff
		# in=d_model, out=d_ff for w1, w3
		# in=d_ff, out=d_model for w2
		# !!!!!!!!
		# std = math.sqrt(2/(d_model + d_model))
		std = math.sqrt(2/(d_model + d_ff))
		self.w1 = initialize_weight([self.d_ff, self.d_model], 0.0, std, -3*std, 3*std, device, dtype)
		self.w2 = initialize_weight([self.d_model, self.d_ff], 0.0, std, -3*std, 3*std, device, dtype)
		self.w3 = initialize_weight([self.d_ff, self.d_model], 0.0, std, -3*std, 3*std, device, dtype)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		assert x.shape[-1] == self.d_model, f"expected last dim {self.d_model}, got {x.shape[-1]}"
		linear_out = x.matmul(self.w1.T)
		silu_out = torch.sigmoid(linear_out) * linear_out
		return (x.matmul(self.w3.T) * silu_out).matmul(self.w2.T)

	def assign_weight(self, w1: torch.Tensor, w2:torch.Tensor, w3:torch.Tensor):
		assert w1.shape == (self.d_ff, self.d_model), f"expected shape ({self.d_ff, self.d_model}), got {tuple(w1.shape)}"
		assert w2.shape == (self.d_model, self.d_ff), f"expected shape ({self.d_model, self.d_ff}), got {tuple(w2.shape)}"
		assert w3.shape == (self.d_ff, self.d_model), f"expected shape ({self.d_ff, self.d_model}), got {tuple(w3.shape)}"
		assert not torch.isnan(w1).any(), "weight contains NaNs"
		assert not torch.isnan(w2).any(), "weight contains NaNs"
		assert not torch.isnan(w3).any(), "weight contains NaNs"
		self.w1 = nn.Parameter(w1)
		self.w2 = nn.Parameter(w2)
		self.w3 = nn.Parameter(w3)

""" Args:
		theta: float, d_k: int, max_seq_len: int
	aligns the shapes from the rightmost dimensions:
	x: 				(batch, head, seq, d_k)
	even:			(batch, head, seq, d_k//2)
	
	token_position: (           , seq, d_k)
	cos:			(           , seq, d_k//2)		 broadcasting would Pad the all left dimensions with size 1
	token_position: (batch  ,  1, seq, d_k)          need to unsqueeze(-3)
	cos:			(batch  ,  1, seq, d_k//2)
"""
class RotaryPositionalEmbedding(nn.Module):
	def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
		super().__init__()
		self.theta = theta
		assert d_k % 2 == 0
		self.d_k = d_k
		self.max_seq_len = max_seq_len
		self.device = device

	def get_inv_freq(self):
		pair_idx = torch.arange(0, self.d_k // 2, dtype=torch.float32)
		inv_freq = 1.0 / (self.theta ** (2 * pair_idx / self.d_k))
		return inv_freq
	
	# input tensor x of shape (..., seq_len, d_k)
	""" token_positions: a tensor of shape (..., seq_len)
		example of token_positions[
									[0,1,2,3,4,5,6,7],
									[0,1,2,3,4,5,6,7]
								]
	"""
	def forward(self, x: torch.Tensor, token_positions: torch.Tensor|None=None) -> torch.Tensor:
		*leading_dims, seq_len, d_k = x.shape
		assert d_k == self.d_k
		#inv_freq = self.get_inv_freq()
		pair_idx = torch.arange(0, self.d_k // 2, dtype=torch.float32, device=x.device)
		inv_freq = 1.0 / (self.theta ** (2 * pair_idx / self.d_k))
		# construct token_positions is not specified
		if token_positions is None:
			token_positions = torch.arange(seq_len, device=x.device)
		# inserts a new dimension of size 1.
		# then broadcast the last dimension to size d_k//2
		# shape become: (..., seq_len, d_k//2)
		# to facilitate the later rotate in element wise
		rotary = token_positions[..., :, None] * inv_freq
		cos = rotary.cos()
		sin = rotary.sin()

		#  split into pairs
		x = x.reshape(*leading_dims, seq_len, d_k // 2, 2)
		# Integer indexing removes that dimension
		even = x[..., 0]
		odd = x[..., 1]

		#  rotate, element wise multiply 
		# even shape: (*leading_dims, seq_len, d_k//2)
		# cos shape:  (*leading_dims, seq_len, d_k//2)
		even_rot = even * cos - odd * sin
		odd_rot = even * sin + odd * cos

		#  restore shape
		out = torch.stack((even_rot, odd_rot), dim=-1)
		return out.reshape(*leading_dims, seq_len, self.d_k)


# apply softmax to the 𝑖-th dimension of the input tensor.
# Args: i: int
def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
	x_max = torch.max(x, dim=i, keepdim=True).values
	x_exp = torch.exp(x - x_max)
	x_sum = torch.sum(x_exp, dim=i, keepdim=True)
	return x_exp / x_sum

# apply mask to QK, return softmax(scores) @ V
# softmax: arbitrary scores => a probability distribution.
def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Bool[torch.Tensor, "... queries keys"]| None=None ) -> torch.Tensor:
	assert Q.shape[:-2] == K.shape[:-2] == V.shape[:-2], \
		"Batch dimensions of Q, K, and V must match."
	assert K.shape[-2] == V.shape[-2], \
		"K and V must have the same number of keys."
	assert Q.shape[-1] == K.shape[-1], \
		"Q and K must have the same embedding dimension."
	d_k = Q.shape[-1]
	queries = Q.shape[-2]
	keys = K.shape[-2]
	scores = Q.matmul(rearrange(K,"... seq_len d_k -> ... d_k seq_len"))
	scores = scores/math.sqrt(d_k)
	if mask is None:
		# True means "mask this position, replace with value"
		mask = torch.triu(torch.ones(queries, keys, dtype=torch.bool, device=Q.device),diagonal=1)
		scores = scores.masked_fill(mask, float("-inf"))
	else:
		assert mask.shape[-2:] == (queries, keys), f"last 2 dimensions of mask must be [{queries}, {keys}]."
		scores = scores.masked_fill(~mask, float("-inf"))

	attention_weights = torch.softmax(scores, dim=-1)
	return attention_weights @ V

""" Args:
		d_model: int, num_heads: int, rope: RotaryPositionalEmbedding(optional)
"""
class Multihead_self_attension(nn.Module):
	def __init__(self, d_model: int, num_heads: int, rope: RotaryPositionalEmbedding| None=None, device=None, dtype=None):
		super().__init__()
		assert d_model % num_heads == 0
		self.d_model = d_model
		self.num_heads = num_heads
		self.head_dim = d_model // num_heads
		# batch, seq, d_model
		std = math.sqrt(2/(d_model + d_model))
		self.W_Q = initialize_weight([self.d_model, self.d_model], 0.0, std, -3*std, 3*std, device, dtype)
		self.W_K = initialize_weight([self.d_model, self.d_model], 0.0, std, -3*std, 3*std, device, dtype)
		self.W_V = initialize_weight([self.d_model, self.d_model], 0.0, std, -3*std, 3*std, device, dtype)
		self.W_O = initialize_weight([self.d_model, self.d_model], 0.0, std, -3*std, 3*std, device, dtype)

		# Don't apply RoPE to in_features before passing them to Multi-Head Attention
		# since RoPE is applied to Q and K, not to the input embeddings x
		if rope is not None:
			self.rope = rope

	def forward(self, x: torch.Tensor, token_positions: torch.Tensor| None=None) -> torch.Tensor:
		assert x.shape[-1] == self.d_model
		# Project to Q, K, V, not split yet
		Q = x.matmul(self.W_Q.T)   
		K = x.matmul(self.W_K.T)
		V = x.matmul(self.W_V.T)
		# Split into heads and Permutation  1）(..., d_model) -> (..., num_heads, head_dim)
		Q = rearrange(Q, "... (heads dim) -> ... heads dim", heads=self.num_heads)
		K = rearrange(K, "... (heads dim) -> ... heads dim", heads=self.num_heads)
		V = rearrange(V, "... (heads dim) -> ... heads dim", heads=self.num_heads)
		# 2） (..., seq_len, num_heads, head_dim) -> (..., num_heads, seq_len, head_dim)
		Q = Q.transpose(-2,-3)
		K = K.transpose(-2,-3)
		V = V.transpose(-2,-3)

		assert self.rope is not None
		if token_positions is not None:
			assert Q.shape[-2] == token_positions.shape[-1]
		Q = self.rope.forward(Q, token_positions)
		K = self.rope.forward(K, token_positions)

		# Attention
		out = scaled_dot_product_attention(Q, K, V)

		# Concatenate heads
		out = rearrange(out, "... num_heads seq_len head_dim -> ... seq_len (num_heads head_dim) ")
		return out.matmul(self.W_O.T)

	def assign_weight(self, W_Q: torch.Tensor, W_K:torch.Tensor, W_V:torch.Tensor, W_O:torch.Tensor):
		assert W_Q.shape == (self.d_model, self.d_model), f"expected shape ({self.d_model, self.d_model}), got {tuple(W_Q.shape)}"
		assert W_K.shape == (self.d_model, self.d_model), f"expected shape ({self.d_model, self.d_model}), got {tuple(W_K.shape)}"
		assert W_V.shape == (self.d_model, self.d_model), f"expected shape ({self.d_model, self.d_model}), got {tuple(W_V.shape)}"
		assert W_O.shape == (self.d_model, self.d_model), f"expected shape ({self.d_model, self.d_model}), got {tuple(W_O.shape)}"
		assert not torch.isnan(W_Q).any(), "weight contains NaNs"
		assert not torch.isnan(W_K).any(), "weight contains NaNs"
		assert not torch.isnan(W_V).any(), "weight contains NaNs"
		assert not torch.isnan(W_O).any(), "weight contains NaNs"
		self.W_Q = nn.Parameter(W_Q)
		self.W_K = nn.Parameter(W_K)
		self.W_V = nn.Parameter(W_V)
		self.W_O = nn.Parameter(W_O)

""" Args:
		d_model: int, num_head: int,
		mha: Multihead_self_attension,
		ln1: RMSNorm, ln2: RMSNorm, swiglu: SwiGLU
"""
class Block(nn.Module):
	def __init__(self, d_model: int, num_head: int, mha: Multihead_self_attension,
			   ln1: RMSNorm, ln2: RMSNorm, swiglu: SwiGLU):
		super().__init__()
		self.d_model = d_model
		self.num_head = num_head

		self.mha = mha
		self.ln1 = ln1
		self.ln2 = ln2
		self.swiglu = swiglu

	def forward(self, in_features: torch.Tensor) -> torch.Tensor:
		# MHA
		x = self.ln1.forward(in_features)
		residual = in_features + self.mha.forward(x) # add current residual stream, which is in_features
		# swiglu
		x = self.ln2.forward(residual)
		x = residual + self.swiglu(x)
		return x


class LMModel(nn.Module):
	def __init__(self, vocab_size:int, d_model:int, num_layers:int, num_heads:int, 
			  context_length: int, rope_theta:float, d_ff: int, 
			  weights: dict[str, torch.Tensor] | None=None,
			  ):
		super().__init__()
		self.vocab_size=vocab_size
		self.d_model=d_model
		self.num_layers=num_layers
		assert d_model % num_heads==0
		self.rope = RotaryPositionalEmbedding(rope_theta, d_model//num_heads, context_length)
		self.embedding = Embedding(vocab_size, d_model)
		if weights is not None:
			self.embedding.assign_weight(weights.get("token_embeddings.weight"))

		self.layers = nn.ModuleList()
		for layer in range(self.num_layers):
			ln1 = RMSNorm(d_model)
			ln2 = RMSNorm(d_model)
			mha = Multihead_self_attension(d_model, num_heads, self.rope)
			swiglu = SwiGLU(d_model, d_ff)
			self.layers.append(Block(d_model, num_heads, mha, ln1, ln2, swiglu))
			if weights is not None:
				ln1.assign_weight(weights.get(f"layers.{layer}.ln1.weight"))
				ln2.assign_weight(weights.get(f"layers.{layer}.ln2.weight"))
				mha.assign_weight(weights.get(f"layers.{layer}.attn.q_proj.weight"), weights.get(f"layers.{layer}.attn.k_proj.weight"),weights.get(f"layers.{layer}.attn.v_proj.weight"),weights.get(f"layers.{layer}.attn.output_proj.weight"))
				swiglu.assign_weight(weights.get(f"layers.{layer}.ffn.w1.weight"), weights.get(f"layers.{layer}.ffn.w2.weight"),weights.get(f"layers.{layer}.ffn.w3.weight"))

		self.ln_final = RMSNorm(d_model)
		self.LMHead = Linear(d_model, vocab_size)
		if weights is not None:
			self.ln_final.assign_weight(weights.get("ln_final.weight"))
			self.LMHead.assign_weight(weights.get("lm_head.weight"))


	def forward(self, in_indices: Int[torch.Tensor, " batch_size sequence_length"]) -> torch.Tensor:
		in_features = self.embedding.forward(in_indices)
		for block in self.layers:
			in_features = block.forward(in_features)
		output = self.ln_final.forward(in_features) # (batch_size, sequence_length, d_model)
		return self.LMHead.forward(output)

# to avoid nearly zero value
def log_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
	max = x.max(dim=dim, keepdim=True).values
	shifted = x-max
	log_sum_exp = torch.log(torch.exp(shifted).sum(dim=dim, keepdim=True))
	return shifted - log_sum_exp


def cross_entropy(inputs: Float[torch.Tensor, " batch_size vocab_size"], targets: Int[torch.Tensor, " batch_size"]) -> Float[torch.Tensor, ""]:
	# !!wrong!! indexes the first dimension, not the last.
	# prob = softmax(inputs, -1)[targets]
	log_prob_distribution = log_softmax(inputs, -1)
	# manuelly create the batch indices, e.g. [0, 1, 2, 3, 4, 5, 6, 7]
	batch = torch.arange(inputs.shape[0], device=inputs.device,)
	# pairs them element-wise: (b0, t0),..., (bi, ti)
	log_probs = log_prob_distribution[batch, targets]
	# result is a scalar
	res= -log_probs.mean()
	return res

class Adamw(torch.optim.Optimizer):
	def __init__(self, params, lr: float, betas =(0.9, 0.999), eps=1e-8,weight_decay=0.01,):
		defaults = dict(
			lr=lr,
			betas=betas,
			eps=eps,
			weight_decay=weight_decay,
		)
		super().__init__(params, defaults)

	@torch.no_grad()
	def step(self):
		for group in self.param_groups:
			lr = group["lr"]
			beta1, beta2 = group["betas"]
			eps = group["eps"]
			weight_decay = group["weight_decay"]
			for p in group["params"]:
				if p.grad is None:
					continue
				# Get state associated with p
				state = self.state[p] 
				if len(state) == 0:
					state["step"] = 0
					state["exp_avg"] = torch.zeros_like(p)
					state["exp_avg_sq"] = torch.zeros_like(p)

				exp_avg = state["exp_avg"]
				exp_avg_sq = state["exp_avg_sq"]
				state["step"] += 1
				t = state["step"]

				grad = p.grad
				# 1. Adjusted learning rate
				step_size = lr * math.sqrt(1-beta2**t) / (1-beta1**t)
				
				# 2. Weight decay
				p.mul_(1-lr*weight_decay)
				
				# 3. Update first moment, mul_(), add_() update the tensor in place
				exp_avg.mul_(beta1)
				exp_avg.add_(grad, alpha=1 - beta1)
				# 4. Update second moment
				exp_avg_sq.mul_(beta2)
				exp_avg_sq.addcmul_(grad, grad, value=1 - beta2) # addcmul_(): self = self + value * tensor1 * tensor2

				# 5. Adam update
				denom = exp_avg_sq.sqrt().add_(eps)
				p.addcdiv_(exp_avg, denom, value=-step_size) # addcdiv_() in place: self = self + value × (tensor1 / tensor2)


def learning_rate_schedule(it: int, max_learning_rate: float, min_learning_rate: float, 
						   warmup_iters: int, cosine_cycle_iters: int):
	if it < warmup_iters:
		return it*max_learning_rate/warmup_iters
	elif it > cosine_cycle_iters:
		return min_learning_rate
	else:
		t_para = (it-warmup_iters)/(cosine_cycle_iters-warmup_iters)*math.pi
		return min_learning_rate+(max_learning_rate-min_learning_rate)/2*(1+math.cos(t_para))

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
	parameters = list(parameters) # list makes the parameters reusable
	total_squared_norm = 0.0
	for param in parameters:
		if param.grad is not None:
			total_squared_norm += param.grad.detach().pow(2).sum() # detach() separates it from autograd computation graph

	total_norm = torch.sqrt(total_squared_norm)
	if total_norm <= max_l2_norm:
		return
	
	scale = max_l2_norm / (total_norm + 1e-6)
	for param in parameters:
		if param.grad is not None:
			param.grad.mul_(scale)


def data_loading(dataset: np.array, batch_size: int, context_length: int, device):
	# Randomly choose the starting positions of each sequence
	start_indices = np.random.randint(
		0,
		len(dataset) - context_length,
		size=batch_size
	)
	x = np.stack([
		dataset[i : i+context_length] for i in start_indices
	])
	y = np.stack([
		dataset[i+1 : i+context_length+1] for i in start_indices
	])

	x = torch.tensor(x, dtype=torch.long, device=device)
	y = torch.tensor(y, dtype=torch.long, device=device)
	x, y = x.to(device), y.to(device)
	return x,y

# TODO: saving the config alongside every checkpoint so it becomes much more self-contained
def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out:str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
	checkpoint = {
		"model": model.state_dict(),
		"optimizer": optimizer.state_dict(),
		"iteration": iteration,
	}
	#print(f"save checkpoint to {out} with {iteration} iterations")
	torch.save(checkpoint, out)

	
def save_model_checkpoint(model: nn.Module, iteration: int, out:str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
	checkpoint = {
		"model": model.state_dict(),
		"iteration": iteration,
	}
	#print(f"save model status checkpoint to {out} with {iteration} iterations")
	torch.save(checkpoint, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: nn.Module, optimizer: torch.optim.Optimizer):
	checkpoint = torch.load(src)
	model.load_state_dict(checkpoint["model"])
	optimizer.load_state_dict(checkpoint["optimizer"])

	return checkpoint["iteration"]

@torch.no_grad()
def decode(
	model: nn.Module,
	prompt: torch.Tensor,
	max_new_tokens: int,
	temperature: float = 1.0,
	top_p: float = 1.0,
	eos_token_id: int | None = None,
) -> torch.Tensor:
	if temperature <= 0:
		raise ValueError("temperature must be greater than 0")

	if not 0 < top_p <= 1:
		raise ValueError("top_p must be in the range (0, 1]")

	model.eval()
	for _ in range(max_new_tokens):
		logits = model(prompt) 
		# logits:(batch_size, context_length, vocab_size)
		# only need the last predict token
		
		next_token_logits = logits[:,-1,:] # (batch_size, vocab_size)
		# Temperature
		next_token_logits = next_token_logits/temperature
		probs = softmax(next_token_logits, -1, ) # [batch_size, vocab_size]
		# nucleus
		sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
		cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
		# top p
		remove_mask = cumulative_probs > top_p
		remove_mask[..., 1:] = (
			remove_mask[..., :-1].clone()
		)
		remove_mask[..., 0] = False
		selected_probs = sorted_probs.masked_fill(
			remove_mask,
			0.0,
		)

		# renormalize
		selected_probs = selected_probs/selected_probs.sum(dim=-1, keepdim=True,)
		sampled_idx = torch.multinomial(
			selected_probs,
			num_samples=1,
		)

		next_token = sorted_indices.gather(
			dim=-1,
			index=sampled_idx,
		)


		prompt = torch.cat([prompt, next_token], dim=1)
		# Stop at <|endoftext|>
		if eos_token_id is not None:
			if torch.all(next_token == eos_token_id):
				break

	return prompt