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
  A novel hierarchical Multi-Agent Reinforcement Learning (MARL) framework<br/>
  achieving state-of-the-art performance on cooperative benchmarks.
</p>

---

##  1. Key Results

### Performance Comparison on MPE (1M Training Steps)

| Environment | H3C-BEACON | MAPPO | IPPO | FACMAC | VDN | COMA | QMIX |
|-------------|------------|-------|------|--------|-----|------|------|
| **simple_spread** | **-13.66** | -15.38 | -19.35 | -19.81 | -30.28 | -35.23 | -39.05 |
| **simple_world_comm** | **+0.56** | -1.99 | -2.13 | -4.72 | -4.28 | -6.08 | -38.59 |

> **H3C-BEACON is the ONLY algorithm to achieve positive reward on simple_world_comm (+0.56)**

### Algorithm Ranking

| Rank | Algorithm | simple_spread | simple_world_comm | Win Rate |
|------|-----------|---------------|-------------------|----------|
| **1** | **H3C-BEACON** | **-13.66** | **+0.56** | **100%** |
|  2 | MAPPO | -15.38 | -1.99 | 95% |
|  3 | IPPO | -19.35 | -2.13 | 85% |
| 4 | FACMAC | -19.81 | -4.72 | 65% |
| 5 | VDN | -30.28 | -4.28 | 45% |
| 6 | COMA | -35.23 | -6.08 | 50% |
| 7 | QMIX | -39.05 | -38.59 | 30% |

### Improvement vs Baselines

| Metric | H3C vs MAPPO | H3C vs IPPO | H3C vs Best Baseline |
|--------|--------------|-------------|----------------------|
| **simple_spread** | **+11.2%** | **+29.4%** | **+11.2%** |
| **simple_world_comm** | **+128.1%** | **+126.3%** | **+128.1%** |

---

##  2. H3C-BEACON components

H3C-BEACON integrates **6 synergistic components**:

| Component | Description | Contribution |
|-----------|-------------|--------------|
| **DGAT** | Dynamic Graph Attention with distance modulation | 2.5% |
| **Bayesian Fusion** | Belief fusion in natural parameter space | 2.0% |
| **Coalition Formation** | Dynamic grouping via spectral clustering | 1.6% |
| **Dual-Critic** | Adaptive global/local value mixing | 3.1% |
| **RTD++** | Elite anchoring via EMA | **4.3%** |
| **Entropy Control** | Cosine annealing with hard bounds | 2.2% |

```
                    
```

---

##  3. Installation

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

##  4. Quick Start

### Training

```bash
# Train H3C-BEACON on simple_spread (1M steps)
python train.py --algo H3C --env simple_spread --steps 1000000

# Train on simple_world_comm
python train.py --algo H3C --env simple_world_comm --steps 1000000

# Quick training (100K steps)
python train.py --algo H3C --env simple_spread --steps 100000
```

### Compare with Baselines

```bash
# Train all algorithms
python train.py --algo H3C MAPPO IPPO --env simple_spread --steps 1000000

# Evaluate
python evaluate.py --algo H3C --env simple_spread
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

##  5. Ablation Study

### Component Importance

```
 COMPONENT IMPORTANCE RANKING

   1. RTD++ ELITE      ████████████████████████  4.3%  ← Most Critical
   2. DUAL CRITIC      ██████████████████        3.1%
   3. DGAT             ██████████████            2.5%
   4. ENTROPY          █████████████             2.2%
   5. BAYESIAN         ████████████              2.0%
   6. COALITION        █████████                 1.6%
```

### 6. Run Ablation

```bash
# Full ablation study (7 variants × 3 seeds)
python run_ablation.py --env simple_spread --steps 250000 --seeds 3

# Specific variants
python run_ablation.py --variants no_dgat no_bayesian no_dual_critic

# Quick test
python run_ablation.py --quick
```

### Key Findings

- **RTD++ is most critical** (4.3%): Elite anchoring prevents catastrophic forgetting
- **Dual-Critic essential** (3.1%): Multi-scale credit assignment improves coordination
- **DGAT stabilizes training**: 5× lower variance with distance-aware attention
- **Components are synergistic**: Total contribution (15.7%) > sum of individual parts

---


## 7. Citation

```bibtex
@article{mbezele2025h3cbeacon,
  title     = {H3C-BEACON: Hierarchical Hybrid Heterogeneous Control 
               with Bayesian-Elites Adaptive Coalition Network for 
               Multi-Agent Reinforcement Learning},
  author    = {Mbezele Bete, Basile and Alo'o Abessolo, Ghislain},
  journal   = {},
  year      = {2025},
  institution = {University of Yaoundé I, Faculty of Sciences}
}
```

---

## 8. Acknowledgments

This research was conducted at **University of Yaoundé I**, Faculty of Sciences, Department of Computer Science.

---

## 9. License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---



## 10. Contact

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