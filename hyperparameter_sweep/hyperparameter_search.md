# Hyperparameter Search

## Learning Rate Search

I performed the learning-rate search in two rounds. Each candidate
was trained for 2,000 steps using the same model architecture,
dataset, batch size, and other training settings.

### Round 1: Coarse Search

I first evaluated four learning rates:

| Learning Rate | Final Loss |
|--------------:|-----------:|
| 0.001 | 1.8924 |
| 0.002 | 1.9137 |
| 0.004 | 2.3173 |
| 0.006 | 2.2589 |

The results suggested that the best region was between 0.0001 to 0.002.
Therefore, I performed a finer search around this value.

### Round 2: Fine Search

I evaluated:

| Learning Rate | Final Loss |
|--------------:|-----------:|
| 0.0007 | 1.9463 |
| 0.0009 | 1.8995 |
| 0.0010 | 1.8924 |
| 0.0012 | 1.8562 |
| 0.0015 | 1.8822 |

Among the tested values, 0.0012 achieved the lowest validation loss
after 2,000 steps. I therefore selected 0.0012 as the learning rate
for further training.