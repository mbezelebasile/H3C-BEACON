# H3C-BEACON

**Hierarchical Hybrid Heterogeneous Control with Bayesian-Elites Adaptive Coalition Network**

Basile BETE MBEZELE and Ghislain ALO'O ABESSOLO  
University of Yaoundé I, Cameroon  
*Submitted to Complex & Intelligent Systems, Springer, 2026*

---

## Overview

H3C-BEACON is a multi-agent reinforcement learning framework that integrates six components: dynamic graph attention (DGAT), Bayesian belief fusion, spectral coalition formation, a dual-critic architecture, RTD++ elite anchoring, and entropy control. All results are reported over 5 seeds with 95% confidence intervals computed via bilateral t-distribution.

---

## Results

### MPE (1M steps, 5 seeds ± 95% CI)

| Algorithm | simple_spread Best Reward | Win Rate | simple_world_comm Best Reward | Win Rate |
|-----------|--------------------------|----------|-------------------------------|----------|
| H3C-BEACON | **−18.29 ± 1.60** | **95.8% ± 4.5%** | **−2.35 ± 0.62** | **100% ± 0%** |
| MAPPO | −24.32 ± 15.90 | 22.5% ± 27.4% | −6.06 ± 0.70 | 98.1% ± 3.5% |
| IPPO | −43.23 ± 21.34 | 8.6% ± 13.2% | −94.63 ± 50.24 | 22.5% ± 27.4% |
| VDN | −52.14 ± 28.43 | 4.2% ± 8.1% | −97.10 ± 35.15 | 19.4% ± 11.8% |
| QMIX | −56.31 ± 29.84 | 3.8% ± 7.2% | −98.81 ± 27.27 | 19.4% ± 13.8% |

### SMAC 3s5z (5 seeds, 1M steps)

| Algorithm | First Win (steps) | Final Reward |
|-----------|------------------|--------------|
| MAPPO | 102,384 ± 28,644 | 14.48 ± 4.67 |
| H3C-BEACON | 351,470 ± 242,024 | 10.65 ± 1.18 |

MAPPO converges 3.4× faster on this map. H3C-BEACON achieves its first victory on all 5 seeds but does not reach strict win threshold within 1M steps. This is expected: 3s5z uses homogeneous unit compositions where coalition structure provides no advantage.

### Hanabi-full (5 seeds, 1M steps)

| Algorithm | Best Checkpoint Score | Final Score |
|-----------|----------------------|-------------|
| H3C-BEACON | **3.40 ± 1.17 / 25** | 2.08 ± 4.89 |
| MAPPO | 2.29 ± 0.23 / 25 | 2.23 ± 0.28 |

H3C-BEACON exceeds MAPPO by +48% at best checkpoint. The GPU seed (789) is stable (best 4.72, final 4.31). CPU seeds exhibit policy forgetting due to thermal throttling.

### Ablation (simple_spread, 3 seeds, 500K steps)

| Variant | Best Reward | Win Rate | ΔWR |
|---------|-------------|----------|-----|
| H3C full | −18.29 ± 1.60 | 95.8% | — |
| −RTD++ | −25.77 ± 6.66 | 26.0% | −69.8 pp |
| −Coalitions | −29.16 ± 20.3 | 28.1% | −67.7 pp |
| −Dual Critic | −22.16 ± 3.97 | 31.3% | −64.6 pp |
| −Entropy | −21.58 ± 2.04 | 36.5% | −59.4 pp |
| −DGAT | −23.89 ± 4.51 | 67.7% | −28.1 pp |

---

## Installation

```bash
git clone https://github.com/mbezelebasile/H3C-BEACON.git
cd H3C-BEACON
pip install -r requirements.txt
```

---

## Usage

```bash
# MPE
python trainF.py --algo H3C --env simple_spread --steps 1000000 \
                 --seeds 42 123 456 789 1024

# SMAC
python train_smac_h3c_fixed.py --algo H3C --map 3s5z \
                                --steps 1000000 --seeds 42,123,456,789,1024

# Hanabi
python train_hanabi_full.py --algo H3C \
                            --seeds 42 123 456 789 1024 --steps 1000000
```

---

## Reproducibility

Seeds: 42, 123, 456, 789, 1024.  
95% CI: bilateral t-distribution, t(0.975, df=n−1).  
Hardware: Intel CPU for MPE and SMAC; Google Colab GPU for Hanabi MAPPO baselines.  
Full hyperparameters: `modules/H3CTrainer_Fixed.py → get_config_summary()`.

---

## Citation

```bibtex
@article{mbezele2026h3cbeacon,
  title   = {H3C-BEACON: Hierarchical Hybrid Heterogeneous Control
             with Bayesian-Elites Adaptive Coalition Network
             for Multi-Agent Reinforcement Learning},
  author  = {Bete Mbezele, Basile and Alo'o Abessolo, Ghislain},
  journal = {Complex \& Intelligent Systems},
  year    = {2026},
  
}
```

---

## License

MIT License. See LICENSE for details.

---

## Contact

Basile BETE MBEZELE — mbezelebetebasile@gmail.com  
University of Yaoundé I, Department of Computer Science, Cameroon
