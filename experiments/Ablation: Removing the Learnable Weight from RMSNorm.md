# Ablation Study: Removing the Learnable Weight from RMSNorm
 
## 1. Motivation and Research Question

When I studied transformer weights, I read an explanation that learnable weights control how information flows through matrix multiplication. An elementwise weight is different. It only rescales each element of the input. It does not mix information across dimensions the way a matrix multiply does. So the elementwise weight feels less tightly coupled to the transformer architecture than the matrix-multiply weights are.

I plan to explore: what happens if the learnable scale weight is removed from RMSNorm? My plan is to remove the learnable scale weight from RMSNorm and see what changes.

I chose RMSNorm because it appears in every block of my model, plus in the final norm layer before the output. 



### Research Question

The goal is to measure how the learnable scale affects training in three ways:

**Optimization**
- train loss vs. step curve
- validation loss vs. step curve
- final train loss
- final validation loss 

**Stability**
- mean of validation loss across seeds
- standard deviation of validation loss across seeds


## 2. Experimental Setup

### 2.1 model
Inside RMSNorm, the weight does an elementwise multiply on the normalized output. This learnable weight lets the model rescale each dimension separately.

Standard RMSNorm:
```
    x
    ↓
RMS normalization
    ↓
dimension-wise learned scaling (γ)
    ↓
  output
```
 
No-weight RMSNorm / Ablation:
```
    x
    ↓
RMS normalization
    ↓
  output
```

To test my idea, I compare a standard transformer architecture to a "no-weight" architecture. In the no-weight version, I remove the elementwise weight from RMSNorm. And keep everything else identical between the baseline and the ablation. 

**Baseline (standard RMSNorm)**
 
The model has two RMSNorm layers per transformer block. I use a pre-norm architecture:
- `ln1`: before the attention sublayer
- `ln2`: before the SwiGLU sublayer
Both `ln1` and `ln2` have one learnable weight:
```python
x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
x = x * weight
```

**Ablation (no-weight RMSNorm)**
 
I remove the learnable weight from every RMSNorm layer:
```python
x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
```
 
So both `ln1` and `ln2` in each block as well as final RMSNorm layer lose their weight, and nothing else in the architecture changes.

**Training setup**

I keep the tokenizer, model architecture, learning-rate schedule, optimizer, random seed, and evaluation procedure identical between the two runs. The only architectural difference is the removal of the RMSNorm scale parameters. 

**Seeds**

I also use seed to control weight initialization and the training/validation data sampling, to remove as many sources of randomness as possible.

## 3. Training Results 

I train the baseline and the ablation model for 6,000 steps respectively, using three seeds: 42, 1337, and 456. Both versions start from the same initial loss for each seed:
 
| Seed | Baseline train loss (step 0) | No-weight train loss (step 0) |
|:----:|:-----------------------------:|:------------------------------:|
| 42   | 9.2646                        | 9.2646                         |
| 1337 | 9.2557                        | 9.2557                         |
| 456  | 9.2597                        | 9.2597                         |
 
![alt text](ValiLoss[baseline&ablation(no weight in RMSRorm)].png)
The validation loss curves for baseline and ablation are close. 
Looking more closely, the ablation's validation loss is slightly higher than the baseline's at every checkpoint.
 
### 3.1 Per-seed results
validation loss for seed 42:
| Step | Baseline val loss | No-weight val loss |   Gap  | Gap % |
|:----:|:-----------------:|:------------------:|:------:|:-----:|
| 0    | 9.2572            | 9.2572             | 0.0000 | ----- |
| 1000 | 2.0807            | 2.1304             | 0.0497 | 2.39% |
| 2000 | 1.8613            | 1.9108             | 0.0495 | 2.66% |
| 3000 | 1.7395            | 1.7788             | 0.0393 | 2.26% |
| 4000 | 1.6450            | 1.6792             | 0.0342 | 2.08% |
| 5000 | 1.6218            | 1.6449             | 0.0231 | 1.42% |
| 6000 | 1.5618            | 1.5858             | 0.0240 | 1.54% |

validation loss for seed 1337:
| Step | Baseline val loss | No-weight val loss |   Gap  | Gap % |
|:----:|:-----------------:|:------------------:|:------:|:-----:|
| 0    | 9.2576            | 9.2576             | 0.0000 | ----- |
| 1000 | 2.1177            | 2.1610             | 0.0433 | 2.05% |
| 2000 | 1.8268            | 1.8733             | 0.0465 | 2.55% |
| 3000 | 1.7456            | 1.7856             | 0.0400 | 2.29% |
| 4000 | 1.6541            | 1.6865             | 0.0324 | 1.96% |
| 5000 | 1.5824            | 1.6069             | 0.0245 | 1.55% |
| 6000 | 1.5548            | 1.5775             | 0.0227 | 1.46% |

validation loss for seed 456:
| Step | Baseline val loss | No-weight val loss |   Gap  | Gap % |
|:----:|:-----------------:|:------------------:|:------:|:-----:|
| 0    | 9.2578            | 9.2578             | 0.0000 | ----- |
| 1000 | 2.0546            | 2.1034             | 0.0488 | 2.38% |
| 2000 | 1.8559            | 1.9099             | 0.0540 | 2.91% |
| 3000 | 1.7695            | 1.8136             | 0.0441 | 2.49% |
| 4000 | 1.6566            | 1.6897             | 0.0331 | 2.00% |
| 5000 | 1.6014            | 1.6277             | 0.0263 | 1.64% |
| 6000 | 1.5975            | 1.6209             | 0.0234 | 1.47% |

These results suggest the learnable RMSNorm scale slightly speeds up training. But the gap percentage shrinks as training goes on. Also, the loss-vs-step curve shows the no-weight run is just as stable as the baseline.
 
**To do:** continue training from these checkpoints and check whether this trend holds.

### 3.2 Cross-seed statistics

Next I calculated the mean and standard deviation of val loss from both models, across three of seeds in my experiment: 42, 1337, 456.

| Step | Baseline Mean | Baseline Std | Ablation Mean | Ablation Std | Mean gap | Std of gap |
|:---:|:---:|:---:|---|---|---|---|
| 1000 | 2.0843 | 0.0317 | 2.1316 | 0.0288 | 0.0473 | 0.0035 |
| 2000 | 1.8480 | 0.0186 | 1.8980 | 0.0214 | 0.0500 | 0.0038 |
| 3000 | 1.7515 | 0.0159 | 1.7927 | 0.0184 | 0.0411 | 0.0026 |
| 4000 | 1.6519 | 0.0061 | 1.6851 | 0.0054 | 0.0332 | 0.0009 |
| 5000 | 1.6019 | 0.0197 | 1.6265 | 0.0190 | 0.0246 | 0.0016 |
| 6000 | 1.5714 | 0.0229 | 1.5947 | 0.0230 | 0.0234 | 0.0007 | 

Notice the individual standard deviation of val loss has fairly large value(column "Baseline Std", "Ablation Std"), from 0.0054 to 0.0317, but the standard deviation of gap val losses across three seeds(column "Std of gap") has relative small value, from 0.0007 to 0.0038. 

The gap varies less across seeds than the individual validation losses, suggesting that some seed-dependent variation is shared between the two conditions and therefore partially cancels when comparing them directly.


## 4. What Does the Learnable RMSNorm Scale Do?
Because the training losses looked close, I wanted to look deeper into the RMSNorm weight statistics. I loaded checkpoints and inspected the learned RMSNorm weights respectively.

### 4.1 γ statistics

At steps 3000 and 6000, I recorded the mean of each layer's RMSNorm weight(I refer to γ as the learnable scale parameter in RMSNorm) from seed-42 model:
|   Layer  | Step 3000 – ln1 mean | Step 3000 – ln2 mean | Step 6000 – ln1 mean | Step 6000 – ln2 mean |
|:--------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|
| layer0   | 0.460742             | 0.564059             | 0.411957             | 0.524661             |
| layer1   | 0.497952             | 0.562219             | 0.460789             | 0.530776             |
| layer2   | 0.595453             | 0.631122             | 0.557453             | 0.598735             |
| layer3   | 0.678471             | 0.703492             | 0.642788             | 0.664543             |


After normalizing, the model does not scale its output back up to full magnitude. Instead it roughly settles on 40-70% of full magnitude.

The mean weight also grows with depth: later layers use a larger RMSNorm scale than earlier layers. This suggests the model does not use RMSNorm's scale the same way at every depth. Different depths seem to prefer different signal magnitudes.
 
However, this comes from a single training run, so I do not treat it as a firm conclusion yet.

### 4.2 Activation RMS
The next question is: without the learnable scale, how does the model control the magnitude of the data flowing through the network?
 
Section 4.1 only examines the static mean of the RMSNorm weights, so it does not show what happens to the activations as they flow through the network. So I need to check on the activation statistics. 
Because I'm interested in activation magnitude, so I use RMS as the primary measure.
$RMS(x) = \sqrt{\mathbb{E}_i[x_i^2]}$

I loaded the seed-42 checkpoint at step 6000 for both baseline and ablation, and put both models in eval mode. I compared the pre- and post-RMSNorm activation statistics between the two models, using the same data sampling seed for both.
(I also control the seed for data sampling when comparing these two models' activation)

|  Layer / LN |     Row    | Baseline | No-weight |
|:-----------:|:----------:|:--------:|:---------:|
| Layer 0 LN1 | Input RMS  | 0.6684   | 0.6788    |
|             | Output RMS | 0.4107   | 1.0000    |
| Layer 0 LN2 | Input RMS  | 0.7020   | 0.7677    |
|             | Output RMS | 0.5212   | 1.0000    |
| Layer 1 LN1 | Input RMS  | 1.0441   | 2.4327    |
|             | Output RMS | 0.4585   | 1.0000    |
| Layer 1 LN2 | Input RMS  | 1.1745   | 2.7204    |
|             | Output RMS | 0.5173   | 1.0000    |
| Layer 2 LN1 | Input RMS  | 1.6812   | 4.1004    |
|             | Output RMS | 0.5512   | 1.0000    |
| Layer 2 LN2 | Input RMS  | 2.0289   | 4.8473    |
|             | Output RMS | 0.5886   | 1.0000    |
| Layer 3 LN1 | Input RMS  | 2.5237   | 5.9230    |
|             | Output RMS | 0.6354   | 1.0000    |
| Layer 3 LN2 | Input RMS  | 3.1055   | 7.1579    |
|             | Output RMS | 0.6599   | 1.0000    |


First compare the input RMS between baseline and no-weight for ln1: 

    Layer 0: 0.6684 vs 0.6788 
    Layer 1: 1.0441 vs 2.4327 
    Layer 2: 1.6812 vs 4.1004 
    Layer 3: 2.5237 vs 5.9230 
The two models develop very different activation scales before RMSNorm, and the gap widens with depth: the ablation model's input RMS grows faster with depth than the baseline's. That also apply to ln2.

Then let's look at the output activation:
The no-weight model's RMSNorm output RMS is 1.0 for all RMSNorm layer, This is as expected: 
Normalization guarantees by construction(ignoring $\epsilon$, 1e-5 in my model setting):
$$y_i = x_i/norm = x_i/\sqrt{\mathbb{E}_i[x_i^2]}$$
$$\text{RMS}(\hat{y_i}) = \sqrt{\mathbb{E}_i[\hat{y}_i^2]} 
 =\sqrt{\mathbb{E}_i[x_i^2/\mathbb{E}_i[x_i^2]]} = \sqrt{\mathbb{E}_i[x_i^2]/\mathbb{E}_i[x_i^2]} = 1 
\quad\Rightarrow\quad \mathbb{E}_i[\hat{y}_i^2] = 1
$$
So come back to my previous question: does the ablation model lost control of magnitude of data flow at norm layer? It seems so. Its output scale is fixed at exactly 1.0 by construction, while the baseline model's output activation at RMSNorm consistently scale below 1, range from 0.41 to 0.66.
Then does the ablation model find other way to compensate the magnitude control somewhere else?

I compare these two models' attention weights and the SwiGLU weights
here is checkpoint statistic of two models using seed 42 at step 6000

**Q/K/V/O statistics**

| Layer | Weight | Baseline std | No-weight std |  Δ %    |
|:-----:|:------:|:------------:|:--------------:|:------:|
| 0     | W_Q    | 0.048        | 0.036          |  -25.0% |
|      | W_K    | 0.047        | 0.038          |  -19.1% |
|     | W_V    | 0.036        | 0.032          |  -11.1% |
|      | W_O    | 0.039        | 0.032          |  -17.9% |
| 1     | W_Q    | 0.050        | 0.039          |  -22.0% |
|      | W_K    | 0.050        | 0.040          |  -20.0% |
|      | W_V    | 0.046        | 0.043          |   -6.5% |
|      | W_O    | 0.048        | 0.046          |   -4.2% |
| 2     | W_Q    | 0.049        | 0.040          |  -18.4% |
|      | W_K    | 0.049        | 0.041          |  -16.3% |
|      | W_V    | 0.055        | 0.056          |   +1.8% |
|      | W_O    | 0.057        | 0.058          |   +1.8% |
| 3     | W_Q    | 0.048        | 0.041          |  -14.6% |
|      | W_K    | 0.048        | 0.042          |  -12.5% |
|      | W_V    | 0.060        | 0.066          |  +10.0% |
|      | W_O    | 0.062        | 0.067          |   +8.1% |

The Q, K, V, and O weight matrices have different magnitudes in the two models.
The ablation model appears to compensate for the missing RMSNorm scale through changes in other learned parameters. The most noticeable differences occur in the attention Q/K weights, suggesting that attention may play an important role in adapting activation magnitude. However, this experiment does not establish that Q/K weights are the primary mechanism.

**SwiGLU statistics**

| Layer | Weight | Baseline std | No-weight std |  Δ %   |
|:-----:|:------:|:------------:|:--------------:|:-----:|
| 0     | w1     | 0.047        | 0.045          |  -4.3% |
|      | w2     | 0.047        | 0.046          |  -2.1% |
|      | w3     | 0.047        | 0.045          |  -4.3% |
| 1     | w1     | 0.048        | 0.044          |  -8.3% |
|     | w2     | 0.048        | 0.045          |  -6.3% |
|     | w3     | 0.047        | 0.042          |  -10.6%|
| 2     | w1     | 0.051        | 0.049          |  -3.9% |
|      | w2     | 0.050        | 0.049          |  -2.0% |
|      | w3     | 0.050        | 0.049          | - -2.0% |
| 3     | w1     | 0.053        | 0.052          |  -1.9% |
|      | w2     | 0.052        | 0.053          |  +1.9% |
|      | w3     | 0.053        | 0.054          |  +1.9% |

The SwiGLU weights have similar magnitudes in the two models. So the no_weight model 
control data flow's magnitude more likely happend in attention scorce calculation, by adaptive
Q, K weight. 
### 4.3 Mathematical relationship between γ and output RMS
I found an interesting relationship when comparing the output RMS vs. mean weight of baseline's RNSNorm layers:
| Layer / LN  | Weight mean (baseline) | Output RMS (baseline) | Gap     | Gap %  |
|:-----------:|:-----------------------:|:-----------------------:|:-------:|:------:|
| Layer 0 LN1 | 0.411957                | 0.4107                  | -0.0013 | -0.31% |
| Layer 1 LN1 | 0.460789                | 0.4585                  | -0.0023 | -0.50% |
| Layer 2 LN1 | 0.557453                | 0.5512                  | -0.0062 | -1.12% |
| Layer 3 LN1 | 0.642788                | 0.6354                  | -0.0074 | -1.15% |
| Layer 0 LN2 | 0.524661                | 0.5212                  | -0.0035 | -0.66% |
| Layer 1 LN2 | 0.530776                | 0.5173                  | -0.0135 | -2.54% |
| Layer 2 LN2 | 0.598735                | 0.5886                  | -0.0101 | -1.69% |
| Layer 3 LN2 | 0.664543                | 0.6599                  | -0.0046 | -0.70% |

The mean RMSNorm weight and output RMS are close across all layers, with less than 3% gap.

Below is simple derivation of mean of weight vs. output RMS relationship:


1) **What the weight multiply does**
 
$$\text{output}_i = \gamma_i \cdot \hat{y}_i$$
 
$$\text({RMS(output)})^2 = \mathbb{E}_i\left[\gamma_i^2 \hat{y}_i^2\right]$$
 
In general, an average of **product** of $\gamma_i^2$ and $\hat{x}_i^2$ is not the average of a product is **not** the product of the averages.
 
2) **Split the product using the covariance identity**
 
For any two channel-indexed quantities $A_i, B_i$:
$$\mathbb{E}_i[A_i B_i] = \mathbb{E}_i[A_i]\,\mathbb{E}_i[B_i] + \text{Cov}_i(A_i, B_i)$$
 
Let $A_i = \gamma_i^2$ and $B_i = \hat{y}_i^2$. Using $\mathbb{E}_i[\hat{y}_i^2] = 1$ from Drivation of no weight ablation:
 
$$\text({RMS(output)})^2 = \mathbb{E}_i[\gamma_i^2] \cdot 1 + \text{Cov}_i(\gamma_i^2, \hat{y}_i^2) = \mathbb{E}_i[\gamma_i^2] + \text{Cov}_i(\gamma_i^2, \hat{y}_i^2)$$
 

3) **Relate mean(γ²) to mean(γ)**
 
By the definition of variance:
$$\text{Var}_i(\gamma) = \mathbb{E}_i[\gamma_i^2] - \left(\mathbb{E}_i[\gamma_i]\right)^2 \quad\Rightarrow\quad \mathbb{E}_i[\gamma_i^2] = \left(\mathbb{E}_i[\gamma_i]\right)^2 + \text{Var}_i(\gamma)$$
 
Substituting into the result from Step 2:
 
$$\boxed{ ({RMS(output)})^2  = {mean}(\gamma))^2 + \text{Var}_i(\gamma) +\text{Cov}_i(\gamma_i^2, \hat{y}_i^2) }$$
The data shows:
$$\boxed{    \text{RMS(output)} \approx \text{mean}(\gamma) \quad }$$
Two possible hypothesis:
1. **Cancellation of Var(γ) and Cov(γ²,y²) hypothesis**: variance of $\gamma$ is either zero or positive while covariance of (γ²,y²) could be negative
2. **Low Var(γ) + Low Cov(γ²,y²) hypothesis**: 
a) Low weight variance across channels $\text{Var}_i(\gamma) \ll \text{mean}(\gamma)^2$,i.e. $\gamma$ doesn't vary much, nearly uniform across channels, low spread.
b) Low covariance between weight and signal $\text{Cov}_i(\gamma_i^2, \hat{x}_i^2) \approx 0$, i.e. large-weight channels aren't systematically the same channels that carry large normalized activations. 
I calculate the Cov(γ²,y²) value from the equation of previous derivation.


| Layer / LN |  mean (γ) |  RMS (Output) | Var(γ) | Cov(γ²,y²) | \|Cov(γ²,y²)/ mean²\| | Var(γ)/ mean² |
|---|---|---|---|---|---|---|
| Layer 0 LN1 | 0.411957 | 0.4107 | 0.0016 | -0.00263408 | 1.6% | 0.9% |
| Layer 1 LN1 | 0.460789 | 0.4585 | 0.001681 | -0.003785253 | 1.8% | 0.8% |
| Layer 2 LN1 | 0.557453 | 0.5512 | 0.001296 | -0.008228407 | 2.6% | 0.4% |
| Layer 3 LN1 | 0.642788 | 0.6354 | 0.001369 | -0.010812253 | 2.6% | 0.3% |
| Layer 0 LN2 | 0.524661 | 0.5212 | 0.001089 | -0.004708725 | 1.7% | 0.4% |
| Layer 1 LN2 | 0.530776 | 0.5173 | 0.002025 | -0.016148872 | 5.7% | 0.7% |
| Layer 2 LN2 | 0.598735 | 0.5886 | 0.002304 | -0.01433764 | 4.0% | 0.6% |
| Layer 3 LN2 | 0.664543 | 0.6599 | 0.001521 | -0.007670389 | 1.7% | 0.3% |

Both the covariance and variance terms are relatively small compared with (mean (γ))^2. 
The variance of γ is small, while the negative covariance partially offsets the remaining difference. It suggests that the latter explanation, Low Var(γ) and Low Cov(γ²,y²) hypothesis, is more consistent with the observed behavior.

The low Var(γ) suggests RMSNorm weights vary relatively little across channels, the learned RMSNorm scale behaves more like a layer-level magnitude adjustment than a selective per-channel gating mechanism.

So removing the learnable weight of RMSNorm does not stop the model from controlling representation magnitude. both two architectures reach comparable loss. They use different mechanisms controlling
magnitude: a per-layer multiplicative gate in the baseline, while ablation model adaptively reorganize the attention weights

From the comparism of baseline model's output RMS and the weight's own mean, I infer the weight is fairly uniformly across channels rather than selectively concentrated on a few dimensions. That also support that, without weight in RMSNorm, transfermer model can also adaptive and perform well.


## 5. Continued Training
To see if the gap between baseline and ablation continue to shrink, I continue training from checkpoints.

At step 6000, I loaded the checkpoints of the baseline and no-weight models trained with seed 42 and continued training both models to step 9000. I reset both runs to the same new seed (12345) before continuation so that both models receive the same sequence of newly sampled training batches.
```
Continuation experiment:
Baseline checkpoint ─────────────────────────> 9000 steps
                    continue training with a shared new seed
Ablation checkpoint ─────────────────────────> 9000 steps
```
here is the data of 9000 steps:
| Step | Baseline val loss | No-weight val loss |   Gap  | Gap % |
|:----:|:-----------------:|:------------------:|:------:|:-----:|
| 0    | 9.2572            | 9.2572             | 0.0000 | ----- |
| 1000 | 2.0807            | 2.1304             | 0.0497 | 2.39% |
| 2000 | 1.8613            | 1.9108             | 0.0495 | 2.66% |
| 3000 | 1.7395            | 1.7788             | 0.0393 | 2.26% |
| 4000 | 1.6450            | 1.6792             | 0.0342 | 2.08% |
| 5000 | 1.6218            | 1.6449             | 0.0231 | 1.42% |
| 6000 | 1.5618            | 1.5858             | 0.0240 | 1.54% |
| 7000 | 1.5276            | 1.5451             | 0.0175 | 1.15% |
| 8000 | 1.5435            | 1.5355             | -0.008 | ----- |
| 9000 | 1.5692            | 1.5299             | -0.0114 | ----- |
| min5th_avg_val_loss | 1.5218            | 1.5325             | 0.01076 | 0.71% |
| min_val_loss_step | 1.5159            | 1.5240             | 0.0081 | 0.53% |


1) Both curves remain stable and reach validation losses in a similar range, with no abvious divergence. curves plateau around 1.50–1.60
2) The gap continuing get shrink as training goes. But the validation loss also get more noisier from the data. To avoid per-step noise, I find the steps that achieved the minimun validation loss and the least 5 validation loss for baseline model and ablation model, compare the gap of min5th_avg_val_loss and min_val_loss_step between 2 model, they are both positive, and small.
The learnable RMSNorm weight gives a real, measurable early-training speed advantage, but that advantage decays as training continues, the difference becomes so small relative to the fluctuations in the validation loss curve
to egnore.


## 7. Conclusion and Limitations
### conclution
Removing the learnable scale parameter from RMSNorm will not prevent the Transformer from training.
Without weight in RMSNorm, model's training is smooth. Even though the ablation model seems a little behide to the baseline, but the validation loss gap continued to narrow during the additional 3000 steps of training, suggesting that the no-weight model may partially recover the performance difference with further training.

I investgate the mean of weight in baseline model and check on the activation before and after RMSnorm. The learnable scale appears useful for controlling activation magnitude while No-weight RMSNorm forces every normalized output to have RMS ≈ 1.
But the no-weight model appears to compensate through other learned parameters: Q, K changes are a plausible contributor.

Although RMSNorm γ provides a per-channel scaling mechanism, the relatively small variance of γ suggests that the model primarily uses it to control overall layer magnitude rather than strongly differentiate among channels in this experiment.


### Limitation and Future Work
1. Small model and small trainig data set
2. Only three random seeds were used.
3. The attention-weight analysis is based on a single seed-42 checkpoint.
4. The Q/K weight changes are correlational and do not establish causality.
5. Larger models and datasets should be tested.
