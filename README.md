<p align="center">
  <img src="https://img.shields.io/badge/H3C--BEACON-v1.0-blue?style=for-the-badge" alt="Version"/>
  <img src="https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License"/>
</p>

<h1 align="center"> H3C-BEACON</h1>

<h3 align="center">
  <i>Hierarchical Hybrid Heterogeneous Control with<br/>
  Bayesian-Elites Adaptive COalition Network</i>
</h3>

<p align="center">
  A novel hierarchical Multi-Agent Reinforcement Learning (MARL)<br/>
</p>



---

##  Key Results

### Performance Comparison

| Environment | H3C-BEACON | MAPPO | FACMAC | QMIX | Improvement |
|-------------|------------|-------|--------|------|-------------|
| **simple_spread** | **-13.66** | -24.39 | -19.81 | -39.05 | **+44.0%** |
| **simple_world_comm** | **-2.05** | -2.57 | -4.72 | -17.18 | **+20.2%** |


### Algorithm Ranking

| Rank | Algorithm | simple_spread | simple_world_comm | Avg Rank |
|------|-----------|---------------|-------------------|----------|
| 1 | **H3C-BEACON** | **-13.66** | **-2.05** | **1.0** |
| 2 | MAPPO | -24.39 | -2.57 | 3.5 |
| 3 | FACMAC | -19.81 | -4.72 | 3.5 |
| 4 | IPPO | -24.88 | -3.14 | 4.0 |
| 5 | VDN | -32.55 | -13.07 | 5.5 |
| 6 | QMIX | -39.05 | -17.18 | 6.0 |
| 7 | COMA | -42.02 | -29.12 | 7.0 |

> **Win Rate: 100%** on both benchmarks against all baselines (MAPPO, IPPO, FACMAC, QMIX, VDN, COMA).

---

##  Architecture

H3C-BEACON integrates **6 synergistic components**:

| Component | Description | Contribution |
|-----------|-------------|--------------|
| **DGAT** | Dynamic Graph Attention with distance modulation | 2.5% |
| **Bayesian Fusion** | Belief fusion in natural parameter space | 2.0% |
| **Coalition Formation** | Dynamic grouping via spectral clustering | 1.6% |
| **Dual-Critic** | Adaptive global/local value mixing | 3.1% |
| **RTD++** | Elite anchoring via EMA | **4.3%** |
| **Entropy Control** | Cosine annealing with hard bounds | 2.2% |


# Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU)

### Standard Installation

```bash
# Clone repository
git clone https://github.com/mbezelebasile/H3C-BEACON.git
cd H3C-BEACON

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Google Colab

```python
!git clone https://github.com/mbezelebasile/H3C-BEACON.git
%cd H3C-BEACON
!pip install -q pettingzoo[mpe] gymnasium torch numpy
```

---

##  Quick Start

### Training

```bash
# Train H3C-BEACON on simple_spread (1M steps)
python train.py --env simple_spread --steps 1000000 --seeds 3

# Quick training (100K steps)
python train.py --env simple_spread --steps 100000 --seeds 1
```

### Ablation Study

```bash
# Full ablation study (7 variants × 3 seeds)
python run_ablation.py --env simple_spread --steps 250000 --seeds 3

# Specific variants
python run_ablation.py --variants no_dgat no_bayesian no_dual_critic

# Quick test
python run_ablation.py --quick
```

### Python API

```python
from modules import H3CTrainer, AblationConfig

# Initialize trainer
trainer = H3CTrainer(
    obs_dim=18,
    action_dim=5,
    n_agents=3,
    config={'hidden_dim': 128, 'lr_actor': 3e-4},
    ablation=AblationConfig()  # Full H3C-BEACON
)

# Get actions
import numpy as np
obs = np.random.randn(3, 18)
actions, probs, log_probs, values = trainer.get_actions(obs, explore=True)
```

---

##  Ablation Study


# Component Importance

```
 COMPONENT IMPORTANCE RANKING

   1. RTD++ ELITE      ████████████████████████  4.3%  ← Most Critical
   2. DUAL CRITIC      ██████████████████        3.1%
   3. DGAT             ██████████████            2.5%
   4. ENTROPY          █████████████             2.2%
   5. BAYESIAN         ████████████              2.0%
   6. COALITION        █████████                 1.6%
```

#Key Findings

- **RTD++ is most critical** (4.3%): Elite anchoring prevents catastrophic forgetting
- **Dual-Critic essential** (3.1%): Multi-scale credit assignment improves coordination
- **DGAT stabilizes training**: 5× lower variance with distance-aware attention
- **Components are synergistic**: Total contribution (15.7%) > sum of individual parts

---



#Configuration on Default Hyperparameters

```yaml
# Training
gamma: 0.99
gae_lambda: 0.95
lr_actor: 3e-4
lr_critic: 1e-3
clip_epsilon: 0.2
ppo_epochs: 4

# Architecture
hidden_dim: 128
dgat_heads: 4
coalition_sigma: 1.0

# RTD++
rtd_lambda: 0.01
rtd_alpha: 0.1

# Entropy
entropy_init: 0.5
entropy_final: 0.01
```

---

#Citation

```bibtex
@article{mbezele2025h3cbeacon,
  title     = {H3C-BEACON: Hierarchical Hybrid Heterogeneous Control 
               with Bayesian-Elites Adaptive Coalition Network for 
               Multi-Agent Reinforcement Learning},
  author    = {Mbezele Bete, Basile and Alo'o Abessolo, Ghislain},
  journal   = {arXiv preprint},
  year      = {2025},
  institution = {University of Yaoundé I, Faculty of Sciences}
}
```

---

#  Acknowledgments

This research was conducted at **University of Yaoundé I**, Faculty of Sciences, Department of Computer Science.



#License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

#Contact

<p align="center">
  <a href="https://github.com/mbezelebasile">
    <img src="https://img.shields.io/badge/GitHub-mbezelebasile-181717?style=for-the-badge&logo=github" alt="GitHub"/>
  </a>
  <a href="mailto:mbezelebetebasile@gmail.com">
    <img src="https://img.shields.io/badge/Email-Contact-D14836?style=for-the-badge&logo=gmail" alt="Email"/>
  </a>
</p>

<p align="center">
  <b>Authors:</b><br/>
  Basile BETE MBEZELE<sup>1</sup><br/>
  Ghislain ALO'O ABESSOLO<sup>2</sup>
</p>

<p align="center">
  <b>Institution:</b> University of Yaoundé I, Cameroon 🇨🇲
</p>
