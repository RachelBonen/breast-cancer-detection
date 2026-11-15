# Breast Cancer Detection

Final project for an *Introduction to Deep Learning* course, by Rachel Bonen and Efi Korenfeld.

We built a classifier that labels breast tumors as **malignant** or **benign** from their measured characteristics. Because the dataset is small and imbalanced, we first balanced it with a **variational autoencoder (VAE)** that generates synthetic malignant samples, then trained a fully-connected neural network and compared it with classical baselines.

## Dataset

[Breast Cancer Dataset (Kaggle)](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset): 569 samples, each with a diagnosis and numeric tumor features. The classes are imbalanced: 212 malignant (37%) and 357 benign (63%).

## Approach

1. **Exploration** - dropped the irrelevant `id` column, encoded the labels (M = 0, B = 1), and compared feature distributions and correlations between the two classes.
2. **Augmentation with a VAE** - an encoder/decoder of fully-connected blocks (Linear, BatchNorm1d, ReLU) trained with reconstruction loss plus a beta-weighted KL term. It generated 145 synthetic malignant samples to balance the classes. A relatively small beta worked best.
3. **Checking the synthetic data** - Mann-Whitney U and Kolmogorov-Smirnov tests plus side-by-side boxplots. All features passed both tests. A tabular GAN ([`tabgan`](https://github.com/Diyago/GAN-for-tabular-data)) did worse on the same tests.
4. **Classifier** - five fully-connected blocks (Linear, BatchNorm1d, ReLU in the first four; Linear with a Sigmoid output in the last), trained with Adam and binary cross-entropy.
5. **Evaluation** - accuracy and F1 score, compared with Random Forest and Gaussian Naive Bayes baselines on the same split.

## Results

| Model | Test accuracy | F1 score |
|---|---|---|
| Our neural network | 98.60% | 0.987 |
| Random Forest | 93.49% | 0.94 |
| Gaussian Naive Bayes | 93.95% | 0.94 |

**Caveat:** the augmented dataset (357 benign + 212 real malignant + 145 synthetic malignant) was shuffled and then split into train, validation and test sets, so the test set contains synthetic samples and is not a purely real-patient evaluation. Treat these numbers as a comparison on this split, not as clinical performance.

## Files

| File | Description |
|---|---|
| `FinalProject_clean.ipynb` | The full notebook: exploration, VAE augmentation, classifier and evaluation. It is the original notebook with the Colab widget metadata removed so GitHub can render it. |
| `FinalProject.pdf` | A PDF export of the notebook, for reading without running anything. |
| `breast-cancer.csv` | The dataset. |

## Running the notebook

The notebook was written in Google Colab. To run it elsewhere:

- Install the dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels`, `scikit-learn`, `scipy`, `torch` and `tabgan`.
- Remove the cell that mounts Google Drive (`from google.colab import drive`).
- Change `data_path` to the location of `breast-cancer.csv`. It currently points to a Google Drive path (`/content/gdrive/My Drive/Intro_to_Deep_Learning/Project/breast-cancer.csv`).

## References

1. [Breast Cancer Dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)
2. [Review of GANs for tabular data](https://towardsdatascience.com/review-of-gans-for-tabular-data-a30a2199342)
3. [GAN-for-tabular-data (`tabgan`)](https://github.com/Diyago/GAN-for-tabular-data)
