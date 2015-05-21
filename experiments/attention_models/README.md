# Recurrent visual attention model
(Experiment notes)

### 2015/3/19
#### REINFORCE learning for the policy of location network


In the paper, the policy is defined as multivariate gaussian pdf of position.

Then the REINFORCE algorithm learns the gradient of this log probability wrt. weight of location network

This never worked in my experimental setting, instead, it seems the following simple gradient can make REINFORCE work.

$$ \nabla_{\theta}{l_t} $$

---

#### Solve zero-gradient problem for gaussian pdf function in theano

Consider sampled position as a constant in T.grad.

Runs with theano 0.6.

```python
_tanh = nnprocessors.build_activation("tanh")

wl = theano.shared(np.array([[0.2,0.3], [0.1,0.3]], dtype="float32"))
h_t = T.constant(np.array([0.1,0.2]))
l_t = _tanh(T.dot(h_t, wl))
sampled_l_t = _sample_gaussian(l_t, T.constant(np.array([[0.1,0],[0,0.1]]), dtype="float32"))
sampled_pdf = _multi_gaussian_pdf(sampled_l_t, l_t)

g = T.grad(T.log(sampled_pdf), wl, known_grads={sampled_l_t: theano.gradient.DisconnectedType()()})

f = theano.function([], [l_t, sampled_pdf, g])
f()
```

---

### 2015/3/24
#### Variance of gaussian sampler should be decrease when the learning get stucked

---

#### REINFORCE rule does not work well with ADADELTA

---

#### May be use gaussian distribution as policy is a bad idea

If the variance is set to be a small value, then a outlier can give exploding gradient.

If the variance is large, then it's hard to make the training converge.

- stochacity will increase over time in the recurrent network
- Clip gradients can not help much

---

### 2015/3/25
Backprop with RL really is non-trivial thing.

---

It seems ADAGRAD is more stable than ADADELTA leads the validate error to 9.8

![screen shot 2015-03-25 at 12 44 42](https://cloud.githubusercontent.com/assets/1029280/6818067/ef0081e8-d2ec-11e4-9e4a-a8e08e98f3f7.png)

---

More details found available in "MULTIPLE OBJECT RECOGNITION WITH VISUAL ATTENTION", including variance and zoom ratio (15%)

### 2015/3/26

Final results using fine-tuning ADAGRAD:

- valid error: 3.64
- test error: 4.19

Differences between the model described in literal:

- tanh non-linearity vs. RELU
 - RELU gives explosion in gradients in my experiments
- Reccurent NN vs. LSTM
 - In this experiment, normal recurrent NN is used to save time

 Other insights:

 - Pre-training with random glimpses may be important
  - Use ADADELTA
  - Enable the network to link partial observation to final prediction

Attention trajectories:

![avaaaaaelftksuqmcc](https://cloud.githubusercontent.com/assets/1029280/6840429/50e5b1d4-d3bb-11e4-9444-8d6319b7de61.png)

[Notebook for plotting](http://nbviewer.ipython.org/github/zomux/wiki/blob/master/experiments/recurrent_visual_attention/Plot%20attentions.ipynb)
