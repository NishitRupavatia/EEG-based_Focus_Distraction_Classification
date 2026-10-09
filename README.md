# EEG-Based Focus and Distraction Classification

## Overview

This project investigates the use of electroencephalography (EEG) signals and deep learning to distinguish between focused and distracted cognitive states during a cognitive task. It implements an EEG processing and classification pipeline that combines signal preprocessing, artifact reduction, temporal feature learning, and machine learning evaluation.

The primary objective is to evaluate whether EEG signals contain discriminative patterns associated with experimentally induced distraction and to compare different neural network architectures for EEG-based binary classification.

The project also emphasizes subject-independent evaluation to investigate how well models generalize beyond the participants whose data were used during training.

## Problem Statement

Distraction can affect attention, task performance, and cognitive efficiency. EEG provides a non-invasive method for recording brain electrical activity and studying changes associated with cognitive processes.

However, EEG classification presents several challenges:

* **Low signal-to-noise ratio:** EEG recordings contain neural activity mixed with physiological and environmental noise.
* **Artifacts:** Eye movements, muscle activity, and other non-neural sources can distort the recorded signals.
* **Temporal variability:** Brain activity changes over time, making consistent classification difficult.
* **Inter-subject variability:** EEG patterns differ across individuals, which can reduce model generalization.
* **Experimental confounds:** A classifier may learn unrelated signal characteristics instead of the cognitive state of interest.

This project explores signal processing and deep learning techniques to address these challenges and evaluate the feasibility of EEG-based focus and distraction classification.

## Project Objectives

* Develop a reproducible EEG preprocessing and classification pipeline.
* Study EEG recordings collected during focused and distracted task conditions.
* Apply filtering, artifact reduction, and epoch-based signal processing.
* Compare established EEG deep learning architectures.
* Evaluate classification performance using subject-independent validation.
* Investigate the role of temporal and spatial information in EEG classification.
* Analyze model performance using metrics beyond accuracy.

## Dataset and Experimental Setup

The dataset was collected from participants performing timed word-puzzle tasks under experimentally controlled conditions. Notifications were introduced to induce distraction during selected trials.

| Dataset characteristic | Description                      |
| ---------------------- | -------------------------------- |
| Participants           | 23                               |
| EEG channels           | 15                               |
| Cleaned trials         | 4,544                            |
| Task                   | Timed word puzzles               |
| Classification task    | Focused vs. distracted           |
| Signal modality        | EEG                              |
| Learning approach      | Supervised binary classification |

The cleaned dataset contains trials retained after preprocessing and quality-control steps.

The experimental labels represent the task conditions assigned during data collection. They should not be interpreted as direct or definitive measurements of a person's internal attentional state.

**Data availability:** The original EEG recordings and processed data should only be included or redistributed in accordance with the applicable consent, privacy, and dataset-sharing requirements.

## Methodology

The overall workflow is organized into the following stages:

1. EEG data loading and inspection
2. Signal filtering and preprocessing
3. Artifact reduction
4. Trial segmentation and epoch construction
5. Dataset preparation and validation splitting
6. Deep learning model training
7. Performance evaluation and comparison
8. Interpretation of classification results

### 1. EEG Signal Preprocessing

EEG signals require preprocessing before model training because raw recordings may contain noise, artifacts, and irrelevant frequency components.

The preprocessing pipeline uses the following techniques.

#### FIR Filtering

Finite Impulse Response (FIR) filtering is used to restrict the signal to a selected frequency range and reduce unwanted frequency components.

The deep learning experiments use a frequency range of approximately **1–45 Hz**, retaining commonly studied EEG rhythms while attenuating slow drift and higher-frequency noise.

The exact filtering configuration should be kept consistent with the implementation used to generate the final experimental results.

#### Notch Filtering

Notch filtering suppresses narrow-band electrical interference, such as power-line noise, when present in the recordings.

#### Independent Component Analysis

Independent Component Analysis (ICA) is used for artifact analysis and reduction. It decomposes multichannel EEG recordings into statistically independent components that may help identify non-neural activity, such as eye-related artifacts.

Artifact removal requires careful component selection. Because ICA operates on the available channel information, its effectiveness depends on recording quality, channel count, and the reliability of artifact identification.

#### Epoching

Continuous EEG recordings are divided into trial-level segments so that each training example corresponds to a defined time interval within an experimental trial.

The deep learning pipeline uses a sampling frequency of **125 Hz** and evaluates task-related temporal windows, including:

* **Control window:** 0–2 seconds.
* **Task-related window:** 2–5 seconds.

These windows are used to examine signal information associated with different phases of the experimental trial. Their interpretation depends on the experimental event timing and labeling procedure.

### 2. Dataset Preparation

After preprocessing, EEG epochs are organized into the input format required by each model.

The preparation stage includes:

* Mapping experimental labels to the binary target classes.
* Checking epoch dimensions and channel consistency.
* Preparing training and validation partitions.
* Maintaining appropriate separation between training and evaluation data.
* Monitoring class distribution and potential data leakage.

For subject-independent evaluation, the validation protocol must prevent trials from the held-out participants from appearing in the training data.

Any normalization, feature selection, or other transformation that learns parameters from the data should be fitted using the training partition only.

## Deep Learning Architectures

The project explores multiple architectures designed to learn spatial, temporal, and spectral representations from EEG signals.

### EEGNet

EEGNet is a compact convolutional neural network designed specifically for EEG-based brain-computer interface tasks.

Its main components include:

* **Temporal convolution:** Learns temporal filters from EEG time series.
* **Depthwise convolution:** Learns channel-wise spatial representations.
* **Separable convolution:** Combines learned representations while reducing parameter count.
* **Nonlinear activation and pooling:** Extract useful patterns while controlling feature dimensions.
* **Classification head:** Converts the learned representation into class predictions.

EEGNet provides a lightweight baseline for evaluating whether convolutional feature learning can distinguish the experimental conditions.

### LMDA-Net

LMDA-Net is an EEG classification architecture designed to learn discriminative features through learned spatial and temporal representations.

Its attention-oriented feature processing helps the model emphasize informative signal components and channel relationships.

In this project, LMDA-Net serves as an alternative to conventional convolution-based EEG classification, allowing performance to be compared across different feature-learning strategies.

### CTNet

CTNet is a convolutional-transformer architecture for EEG classification.

The convolutional component extracts local signal patterns, while transformer-based processing models relationships between learned representations.

This combination allows the model to investigate both local EEG structure and broader temporal dependencies.

### EEG Conformer

EEG Conformer combines convolutional feature extraction with transformer-based attention.

Its general processing stages include:

1. Convolutional extraction of local temporal and spatial features.
2. Projection of learned features into a token or embedding representation.
3. Transformer-based attention across the representation sequence.
4. Classification using the learned features.

The convolutional stage reduces the raw signal into a more compact representation. The transformer stage then learns relationships between these representations.

This architecture is particularly relevant when evaluating whether attention-based modeling can improve the classification of EEG trials.

### Model Comparison

| Model         | Main approach                            | Purpose                                            |
| ------------- | ---------------------------------------- | -------------------------------------------------- |
| EEGNet        | Compact convolutional network            | Baseline EEG classification                        |
| LMDA-Net      | Learned spatial-temporal representations | Evaluate alternative EEG-specific feature learning |
| CTNet         | CNN with transformer-based processing    | Learn local and broader temporal relationships     |
| EEG Conformer | Convolution plus self-attention          | Evaluate attention-based EEG representations       |

All architectures should be compared under consistent data splits, preprocessing, and evaluation metrics wherever their input requirements permit.

## Experimental Evaluation

### Subject-Independent Validation

Subject-independent evaluation is important because EEG recordings from the same individual can share participant-specific characteristics.

The project uses a **five-fold subject-independent cross-validation strategy** to assess model performance on participants separated from the corresponding training data.

For a valid evaluation, each fold should ensure that:

* Training and validation participants are disjoint.
* Preprocessing transformations learned from data are fitted without using held-out participants.
* Model selection does not use the final held-out evaluation data.
* Performance is reported consistently across folds.

Subject-independent results are generally more informative about cross-participant generalization than randomly splitting individual trials, which can allow subject-specific information to appear in both training and validation sets.

### Evaluation Metrics

The following metrics are useful for assessing binary classification performance:

| Metric           | Purpose                                                           |
| ---------------- | ----------------------------------------------------------------- |
| Accuracy         | Fraction of correctly classified trials                           |
| Precision        | Fraction of predicted positive trials that are positive           |
| Recall           | Fraction of positive trials correctly identified                  |
| F1-score         | Harmonic mean of precision and recall                             |
| ROC-AUC          | Ability to distinguish the two classes across decision thresholds |
| Confusion matrix | Breakdown of correct and incorrect predictions for each class     |

Accuracy should not be interpreted in isolation, particularly when class distributions are imbalanced. Recall, precision, F1-score, ROC-AUC, and class-wise error patterns provide additional context.

## Results

The initial EEGNet baseline achieved the following reported performance:

| Metric   | Result |
| -------- | -----: |
| Accuracy |  72.4% |
| ROC-AUC  |  0.794 |

These values describe the reported baseline experiment and should be accompanied by the corresponding evaluation protocol when presenting final results.

The results suggest that the EEG representations contain information that can help distinguish the experimental task conditions. However, performance alone does not establish that the model is detecting attention directly; it may also learn other characteristics associated with the experimental setup.

Results for LMDA-Net, CTNet, and EEG Conformer should be added after their final evaluation runs have been completed under the same validation protocol.

### Interpreting the Results

Several factors are important when interpreting EEG classification performance:

* **Generalization:** Performance on unseen participants is more informative than performance on trials from participants seen during training.
* **Class imbalance:** Accuracy can be misleading if one class is more common than the other.
* **Signal confounds:** Models may exploit non-attentional differences between experimental conditions.
* **Frequency-band dependence:** Strong performance in a particular band should be examined to determine whether it reflects relevant neural activity or an unrelated confound.
* **Reproducibility:** Results should be supported by documented splits, preprocessing configurations, random seeds, and evaluation scripts.

## Technology Stack

| Technology   | Purpose                                           |
| ------------ | ------------------------------------------------- |
| Python       | Core implementation                               |
| MNE-Python   | EEG data handling and signal preprocessing        |
| PyTorch      | Deep learning model implementation and training   |
| NumPy        | Numerical operations                              |
| Pandas       | Data organization and analysis                    |
| Scikit-learn | Evaluation metrics and validation utilities       |
| Matplotlib   | Visualization of signals and experimental results |

The exact package versions should be documented in the project's dependency file or environment configuration.

## Repository Structure

The repository may contain notebooks, scripts, model definitions, evaluation utilities, and generated results. A recommended organization is shown below; adjust the paths to match the actual files in the repository.

```text
EEG-based_Focus_Distraction_Classification/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── preprocessing.ipynb
│   └── model_evaluation.ipynb
├── src/
│   ├── preprocessing/
│   ├── models/
│   │   ├── eegnet.py
│   │   ├── lmda_net.py
│   │   ├── ctnet.py
│   │   └── eeg_conformer.py
│   ├── training/
│   └── evaluation/
├── results/
├── requirements.txt
├── .gitignore
└── README.md
```

This layout is a suggested structure, not a verified listing of the repository's current files.

## Installation

### Prerequisites

* Python 3.10 or a compatible version supported by the project's dependencies.
* Git.
* A virtual environment.
* Jupyter Notebook or JupyterLab if the workflow uses notebooks.
* A CUDA-compatible GPU is optional and may accelerate deep learning experiments.

### 1. Clone the Repository

```bash
git clone https://github.com/NishitRupavatia/EEG-based_Focus_Distraction_Classification.git
cd EEG-based_Focus_Distraction_Classification
```

### 2. Create a Virtual Environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

If the repository includes a `requirements.txt` file:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If a dependency file is not yet available, install the required packages according to the actual imports and model implementations in the repository. Record the tested package versions before publishing the final environment configuration.

### 4. Launch Jupyter

If the implementation is notebook-based:

```bash
jupyter lab
```

Open the relevant notebook and execute the cells in the documented order, starting with data loading and preprocessing.

## Running Experiments

The exact commands depend on the scripts and notebooks included in the repository. A typical experimental workflow is:

1. Load the available EEG recordings and their labels.
2. Inspect channel metadata, sampling rate, and recording quality.
3. Apply the documented preprocessing and artifact-reduction pipeline.
4. Construct the required epochs and labels.
5. Generate subject-independent cross-validation splits.
6. Train each model using the same evaluation protocol.
7. Evaluate the predictions using the selected classification metrics.
8. Save model checkpoints, fold-level metrics, and experiment configurations.

For reproducibility, record the random seed, model configuration, optimizer, learning rate, batch size, epoch count, and validation split for each experiment.

## Reproducibility and Data Leakage Prevention

EEG datasets can contain strong participant-specific and recording-specific patterns. The following practices are recommended:

* Split data by participant when measuring subject-independent generalization.
* Avoid fitting normalization parameters on the complete dataset before splitting.
* Keep held-out participants out of model selection and hyperparameter tuning.
* Compare models using the same folds and evaluation metrics.
* Report fold-level results as well as aggregate results.
* Document all trial-selection and artifact-removal decisions.
* Check whether class labels correlate with recording order, timing, or other experimental variables.
* Use ablation studies to assess whether performance depends on particular time windows, channels, or frequency ranges.

These checks help determine whether a model learns meaningful task-related representations or exploits unintended shortcuts in the dataset.

## Limitations

* EEG signals vary considerably across participants and recording sessions.
* Artifact removal is imperfect and may remove useful signal information or retain unwanted activity.
* Experimentally assigned task labels are proxies for cognitive states rather than direct measurements of attention.
* Small participant counts can limit the reliability of estimates of generalization to new individuals.
* High classification performance does not, by itself, establish a causal relationship between EEG activity and distraction.
* Performance may change with different recording hardware, preprocessing settings, and experimental protocols.

The system should be treated as an experimental research framework rather than a clinically validated diagnostic or real-time attention-monitoring product.

## Future Work

Potential extensions include:

* Further optimization of EEGNet, LMDA-Net, CTNet, and EEG Conformer.
* Evaluation of hybrid convolutional and recurrent architectures, such as CNN-LSTM.
* Investigation of temporal context and sequence modeling across EEG epochs.
* Ablation studies for temporal windows, frequency bands, and channel subsets.
* Improved explainability through feature attribution and model interpretation.
* Evaluation of class-wise errors and participant-level performance.
* Testing on independent datasets to measure external generalization.
* More rigorous controls for sleep-related activity, signal artifacts, and experimental confounds.
* Development of a reproducible training and evaluation pipeline.

## Ethical Considerations

EEG data are sensitive physiological data. Any collection, processing, storage, or sharing of recordings should follow the applicable participant-consent, privacy, and institutional requirements.

The model outputs represent predictions from experimental data and should not be treated as definitive judgments about an individual's attention, mental health, or cognitive ability.

## Author

**Nishit Rupavatia**

GitHub: [NishitRupavatia](https://github.com/NishitRupavatia)

## License

Specify the license applicable to this repository before distributing or reusing the code. If the repository does not yet contain a `LICENSE` file, no open-source license should be assumed.

## Acknowledgements

This project was developed as an academic investigation into EEG signal processing, deep learning, and the classification of experimentally induced focus and distraction.

The work brings together concepts from biomedical signal processing, machine learning, and neural network architectures for EEG analysis.
