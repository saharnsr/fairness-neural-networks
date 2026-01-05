# fairness-neural-networks
Original PhD research code on bias mitigation and fairness-aware neural networks, including in-processing methods, saliency-based regularization, fuzzy controller and fairness metrics.

# Fairness-Aware Neural Networks

This repository contains original PhD research code on bias mitigation and fairness-aware
neural networks. The work focuses on in-processing approaches that integrate fairness
constraints directly into neural network training using fuzzy logics.

## Research Scope
- Bias mitigation during model training (in-processing)
- Fairness-aware loss functions and regularization
- Saliency- and attribution-guided fairness control
- Fuzzy logic–based controllers for fairness adjustment
- Evaluation using standard fairness metrics

## Methods and Tools
- PyTorch-based neural network models
- Gradient-based optimization with fairness regularization
- Feature attribution and saliency analysis
- Fairness metrics such as demographic parity and equalized odds

## Repository Structure

fairness-neural-networks/
├── models/        # Neural network architectures
├── training/      # Training loops with fairness constraints
├── mitigation/    # Bias mitigation and regularization methods
├── metrics/       # Fairness evaluation metrics
├── experiments/   # Reproducible experiment scripts


## Notes for Reviewers
This code represents selected, cleaned components of my doctoral research on
fairness-aware and explainable machine learning. The focus is on clarity,
reproducibility, and research-oriented implementation rather than production packaging.

## Scope and Limitations

This repository does not aim to reproduce all experiments or code associated
with my published papers. Instead, it provides a curated selection of
representative implementations that illustrate the core ideas, modeling
choices, and training strategies used across my research on fairness-aware
neural networks.

Full experimental pipelines and datasets are omitted intentionally to keep
the code concise, readable, and suitable for review.
