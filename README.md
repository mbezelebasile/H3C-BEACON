# H3C-BEACON 🚀

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Hierarchical Hybrid Heterogeneous Communication with Bayesian-Enhanced Adaptive Coordination Network**

A novel Multi-Agent Reinforcement Learning (MARL) framework that achieves state-of-the-art performance through synergistic integration of six innovative components.

## 📊 Key Results

| Environment | H3C-BEACON | MAPPO | Improvement |
|-------------|------------|-------|-------------|
| simple_spread | **-13.66** | -24.39 | **+44.0%** |
| simple_world_comm | **-2.05** | -2.57 | **+20.2%** |

**Win Rate: 100%** on both benchmarks against all baselines (MAPPO, IPPO, FACMAC, QMIX, VDN, COMA).

## 🏗️ Architecture

H3C-BEACON integrates **6 synergistic components**:

| Component | Description | Contribution |
|-----------|-------------|--------------|
| **DGAT** | Dynamic Graph Attention with distance modulation | 2.5% |
| **Bayesian Fusion** | Belief fusion in natural parameter space | 2.0% |
| **Coalition Formation** | Dynamic grouping via spectral clustering | 1.6% |
| **Dual-Critic** | Adaptive global/local value mixing | 3.1% |
| **RTD++** | Elite anchoring via EMA | **4.3%** |
| **Entropy Control** | Cosine annealing with hard bounds | 2.2% |

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/mbezelebasile/H3C-BEACON.git
cd H3C-BEACON

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```



### Training

```bash
# Train H3C-BEACON on simple_spread
python scripts/train.py --env simple_spread --steps 1000000 --seeds 3
```

### Ablation Study

```bash
# Full ablation study (7 variants × 3 seeds)
python scripts/run_ablation.py --env simple_spread --steps 250000 --seeds 3

# Quick test
python scripts/run_ablation.py --quick
```

## 📈 Ablation Study Results

| Variant | Best Reward | Δ vs Full | Contribution |
|---------|-------------|-----------|--------------|
| **Full H3C-BEACON** | **-15.67 ± 0.27** | — | baseline |
| − RTD++ | -16.35 ± 0.58 | -4.3% | **4.3%** |
| − Dual-Critic | -16.15 ± 0.45 | -3.1% | 3.1% |
| − DGAT | -16.06 ± 1.37 | -2.5% | 2.5% |
| − Entropy | -16.02 ± 0.42 | -2.2% | 2.2% |
| − Bayesian | -15.98 ± 0.51 | -2.0% | 2.0% |
| − Coalition | -15.92 ± 0.62 | -1.6% | 1.6% |




University of Yaoundé I, Faculty of Sciences, Department of Computer Science.

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 📧 Contact

- **Author:** MBEZELE BETE Basile
- **GitHub:** [@mbezelebasile](https://github.com/mbezelebasile)
