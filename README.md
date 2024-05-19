## Dynamic training for handling textual label noise

## Environment
We use python version is 3.7.
```markdown
transformers==4.18.0
torch==1.11.0
numpy==1.21.6
pandas==1.3.5
scikit-learn==1.0.2
```

## Datasets

We do experiments on four text classification benchmarks of different types, including Trec, AG-News and IMDB.

| Dataset | Class | Train | Test |
|:--------|:-----:|:-----:|:-----|
|  Trec | 6 | 5452  | 500  |
| IMDB | 2 |  25K  | 25K  |
| AG-News | 4 | 120K  | 7.6K |
| Chnsenticorp | 2 | 10430 | 1200 |


### Noise Sample Generation

We evaluate our strategy under the following two types of label noise

* Asymmetric noise (Asym): Following [Chen et al.](https://arxiv.org/pdf/1905.05040.pdf), we choose a certain proportion of samples and flip their labels to the corresponding class according to the asymmetric noise transition matrix.
* Instance-dependent Noise (IDN): Following [Algan and Ulusoy](https://arxiv.org/pdf/2003.10471.pdf), we train an LSTM classifier on a small set of the original training data and flip the origin labels to the class with the highest prediction.

You can construct noisy datsets according to [the code repo](https://github.com/noise-learning/SelfMix)

We have uploaded two sample datasets to the data directory, which can be used directly. Noisy file is named as ${noise_type}_${noise_ratio}_fig.csv

## Quick start

```markdown
python train.py config/trec_fig.json
```