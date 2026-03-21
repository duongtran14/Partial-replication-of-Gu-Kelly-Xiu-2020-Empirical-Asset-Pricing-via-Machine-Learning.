# Partial-replication-of-Gu-Kelly-Xiu-2020-Empirical-Asset-Pricing-via-Machine-Learning.
This is my own attempt to replicate some of the key findings in the paper Empirical Asset Pricing via Machine Learning by Gu, Kelly, &amp; Xiu (2020)[^1]. Specifically, I use **neural networks with 3 hidden layers** ("NN3") to predict US stocks' excess returns and evaluate the model's performance based on 30-year out-of-sample testing. The paper, using a total of 920 baseline predictors (including 94 stock-level characteristics, 8 macro-economic predictors), has shown that machine learning models, notably trees and neural networks, are very promising for algorithmic trading strategies and can yield large economic gains to investors, demonstrated by a high out-of-sample $R^2$ and Sharpe ratio using a decile-sorted machine learning portfolio strategy (compared to existing literature and other simple regression-based methods). 

Some of my successful replicated statistics are: (i) A pooled out-of-sample $R^2$ of 0.38\% (0.40\% in original paper); (ii) An average **monthly returns of 3.23\%** (3.27\% in original paper) with annualized **Sharpe ratio of 2.5\%** (2.36\% in original paper) using long-short equally-weighted portfolios, with a low maximum drawdown of 8.88\% (17.34\% in orginal paper).

## Replication results
All of my replication results are described below. I denote "good"/ "not good" for statistics that are close/ not close to the ones in the original paper. Overall, my model did a pretty good job in predicting stocks' excess returns in the highest decile, while not so much for the lowest decile. For both equally-weighted and value-weighted portfolios, the actual returns of the short portfolios are much higher than predicted, which is more notable for value-weighted portfolios, shrinking the long-short spreads. However, because I only estimate the model with an ensemble of 5 random seeds due to limited resources while the orignal paper uses 10, I believe that the predictive accuracy can vastly improve with a larger ensemble.

- Monthly out-of-sample stock-level prediction performance using NN3 (percentage $R^2$) (notebook `2_NN3`)
  
| | Replicated  | Original | Replication quality |
|---|---|---|---|
| All stocks| 0.38| 0.40| good |
| Top 1000 by market value  |  0.42| 0.70| not good |
| Bottom 1000 by market value  | 0.72| 0.45| good |

- Performance of equally-weighted machine learning portfolios (notebook `3_MLportfolios`). Out-of-sample stocks are sorted to 10 deciles each month based on their predicted returns. "H-L" denotes a zero-net strategy where we long all stocks in the highest decile and short all stocks in the lowest decile. All statistics are monthly average, except for annualized Sharpe ratio.
  
| | | Replicated  | Original | Replication quality |
|---|---|---|---|---|
|decile 1 | Predicted returns | -1.19 | -0.31 | not good |
| | Observed returns | -0.73 | -0.92 | not good |
| | Observed std | 7.95 | 7.94 | good |
| | Sharpe ratio | -0.32 | -0.40 | good |
|decile 10 | Predicted returns | 2.70 | 2.28 | good |
| | Observed returns | 2.49 | 2.35 | good |
| | Observed std | 8.44 | 8.11 | good |
| | Sharpe ratio | 1.02 | 1.00 | good |
|H-L| Predicted returns | 3.90 | 2.58 | good |
| | Observed returns | 3.23 | 3.27 | good |
| | Observed std | 4.48  | 4.80 | good |
| | Sharpe ratio | 2.50 | 2.36 | good |

- Performance of value-weighted machine learning portfolios (notebook `3_MLportfolios`). Out-of-sample stocks are sorted to 10 deciles each month based on their predicted returns. "H-L" denotes a zero-net strategy where we long all stocks in the highest decile and short all stocks in the lowest decile. All statistics are monthly average, except for annualized Sharpe ratio.
  
| | | Replicated  | Original | Replication quality |
|---|---|---|---|---|
|decile 1 | Predicted returns | -1.04 | -0.03 | not good |
| | Observed returns | -0.11 | -0.43 | not good |
| | Observed std | 7.21 | 7.73 | good |
| | Sharpe ratio | -0.05 | -0.19 | good |
|decile 10 | Predicted returns | 1.95 | 1.83 | not good |
| | Observed returns | 1.54 | 1.69 | not good |
| | Observed std | 7.26 | 7.29 | good |
| | Sharpe ratio | 0.74 | 0.80 | good |
|H-L| Predicted returns | 2.99 | 1.86 | not good |
| | Observed returns | 1.65 | 2.12 | not good |
| | Observed std | 5.10 | 6.13 | good |
| | Sharpe ratio | 1.12 | 1.20 | good |

- Drawdown and Turnover of machine learning portfolios (notebook `3_MLportfolios`):

| | | Replicated  | Original | Replication quality |
|---|---|---|---|---|
Value weighted | Max drawdown (%) | 26.78 | 30.84 | good |
| | Max 1M loss (%) | 13.49 | 30.84 | good |
| | Turnover (%) | 128.46 | 123.50 | good |
Equally weighted | Max drawdown (%) | 8.88 | 17.34 | good |
| | Max 1M loss (%) | 8.09 | 12.50 | good |
| | Turnover (%) | 116.66 | 113.76 | good |


## Replication process

### 1. Input data
- Monthly stock-level characteristics (one-month lag): downloaded from one of the authors, [Xiu's webpage](https://dachxiu.chicagobooth.edu/download/datashare.zip). The data span from 1957 to 2016 (60 years), where the last 30 years, from 1987 to 2016, are used for out-of-sample testing.
- Monthly stock returns: downloaded from WRDS;
- Monthly macro-economic predictors data: downloaded from [Amit Goyal's webpage](https://docs.google.com/spreadsheets/d/10_nkOkJPvq4eZgNl-1ys63PzhbnM3S2y/edit?gid=1922816101#gid=1922816101).
  
### 2. Preprocessing data

Notebook `1_Preprocessing` is where I preprocess the above input data, strictly following the paper, namely:

- Monthly stock-level characteristics:
  - Fill NA values with the cross-sectional median at each month for each stock;
  - Cross-sectionally rank stocks each month and map these ranks to the [-1,1] interval.
- Monthly macro-economic predictors data: Calculate 8 monthly predictors: (i) dividend-price ratio, (ii) earnings-price ratio, (iii), book-to-market ratio, (iv) net equity expansion, (v) Treasury-bill rate, (vi) term spread, (vii) default spread, (viii) and stock variance (svar).
- Calculate interaction terms between stock characteristics and macro-economic predictors; One-hot encoded industry SIC dummies; Creating a (3743049 x 920) design matrix;
- Calculate stocks' excess returns;
- Recursively split the sample 30 times into 3 subsets: training, validation, and testing samples. The first split has training sample spanning from 1957-1974, validation sample spanning from 1975-1986, and testing sample is 1987. Each time the training sample is increased by one year while the validation sample length is always fixed (12 years).

### 3. Training and Testing
Notebook `2_NN3` is where I train neural networks with 3 hidden layers and generate 30-year out-of-sample excess returns predictions. The pooled out-of-sample $R^2$ for all stocks, for top 1000 stocks and for bottom 1000 stocks by market values can be found at the very end of the notebook.

- Create a neural networks specification with 03 hidden layers, activation function is ReLU, and Batch Normalization layers are applied after ReLU transformation, except for the last activation layer;
- Specify parameters grid:
    - l1 regularization parameter: $\lambda \in (10^{-5}, 10^{-3})$;
    - Learning rate $=$ 0.01;
    - Batch size $=$ 10000;
    - Maximum epochs $=$ 100;
    - Patience $=$ 5 (number of consecutive epochs that fails to decrease validation loss);
    - Adam parameters $=$ Default;
    - Ensemble $=$ 5 (number of random seeds).
- The model is fitted on training sample using Adam for Stochastic Gradient Descent;
- After each epoch, predictions for validation sample is generated;
- Early stopping is applied when 5 consecutive epochs fails to decrease validation loss;
- The model that gives minimum validation loss is chosen;
- Use the chosen model to generate predictions for testing sample;
- Run with different seeds and average testing predictions across seeds to get the final predictions for each year;
- Repeat the process to get the pooled out-of-sample predictions for 30 years.

### 4. Construct and evaluate machine learning portfolios

Notebook `3_MLportfolios` is where i evaluate machine-learning portfolios performance:
- Construct decile-sorted portfolios based on individual stock excess return predictions, both equally and value weighted;
- Compute long-short portfolios average monthly returns and annualized Sharpe ratio;
- Compute max drawdown, max 1M loss, and turnover of each strategy.

[^1]: The Review of Financial Studies, Volume 33, Issue 5, May 2020, Pages 2223–2273, https://doi.org/10.1093/rfs/hhaa009
