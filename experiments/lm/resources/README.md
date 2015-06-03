# About Penn Treebank data


These data from taken from Mikolov's page: http://www.fit.vutbr.cz/~imikolov/rnnlm/


## Presented emprical results

| Model        | Individual PP           |
| ------------- |:-------------:|
| 5-gram, Kneser-Ney smoothing (KN5)     | 141.2 |
| 5-gram, Kneser-Ney smoothing + cache      | 125.7      |
| Log-bilinear LM | 144.5      |
| Feedforward neural network LM | 140.2      |
| Recurrent neural network LM | 131.3      |
| Dynamically evaluated RNNLM  | 124.7     |
| Combination of static RNNLMs | 102.1      |
| Combination of dynamic RNNLMs | 101.0      |