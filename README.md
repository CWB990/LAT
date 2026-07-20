# LAT: Vulnerability-Guided Layer-wise Adversarial Training for Robust Recommender Systems

Official implementation of the paper **"Vulnerability-Guided Layer-wise Adversarial Training for Robust Recommender Systems"** (submitted to *IEEE Transactions on Information Forensics and Security*).

## Overview

Layer-wise Adversarial Training (LAT) is a weight-space adversarial training framework for defending deep recommender systems against data poisoning attacks. Guided by a Layer-wise Vulnerability Score (LVS), LAT selectively hardens the most vulnerable layers with adaptive, hierarchically weighted perturbations, instead of applying uniform perturbations across all layers.

Key findings:

- Poisoning vulnerability is **non-uniform** across layers and **migrates** during training, invalidating uniform or fixed-layer perturbation strategies.
- LAT reduces targeted attack success by over 80% (CTHR@50) while maintaining clean recommendation accuracy.
- LAT generalizes to LLM-assisted recommender architectures, where conventional embedding-space defenses may even amplify poisoning signals.

## Repository Structure (TODO)

```
LAT/
├── data/               # Dataset preprocessing (ML-1M, LastFM, Gowalla)
├── models/             # Victim models: MLP, ConvNCF, NeuMF, RLMRec
├── attacks/            # Poisoning attacks: RevAdv, DP, ENT, Random
├── defenses/           # LAT and baselines: APR, VAT, RAWP, GraphRFI
├── lvs/                # Layer-wise Vulnerability Score evaluation
└── README.md
```

## Requirements (TODO)

- Python 3.x
- PyTorch
- (full dependency list to be added)

## Usage (TODO)

Training and evaluation commands will be released soon.

## Citation

```bibtex
@article{chen2026lat,
  title   = {Vulnerability-Guided Layer-wise Adversarial Training for Robust Recommender Systems},
  author  = {Chen, Wenbin and Chen, Hai and Xu, Qian and Zhao, Shu and Wu, Chenwang and Qian, Fulan},
  journal = {IEEE Transactions on Information Forensics and Security},
  note    = {under review},
  year    = {2026}
}
```
