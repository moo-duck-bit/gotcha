# GoTCHA

**Hierarchical Graph-of-Thoughts for Explainable Cognitive Distortion Classification**

[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3748522.3779920-B31B1B)](https://doi.org/10.1145/3748522.3779920)
[![SAC '26](https://img.shields.io/badge/SAC-'26-1F6FEB)](https://www.sigapp.org/sac/sac2026/)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-8957E5)](LICENSE)

Hayeon Yang, Jihuen Hong, Jin Gyeong Park, Kyungmin Kwon, Hayoung Oh
*Proceedings of the 41st ACM/SIGAPP Symposium on Applied Computing (SAC '26)*, pp. 192–200

GoTCHA classifies cognitive distortions by implementing Beck's hierarchical cognitive theory as an LLM graph reasoning framework. Instead of labeling distortions in one pass, it follows the clinical CBT reasoning order — core beliefs → fine-grained beliefs → distortions — where each level conditions on the one above it.

## Method

<img width="563" height="577" alt="gotcha" src="https://github.com/user-attachments/assets/fb79d85c-3178-4656-962c-73e06a14c36d" />



Each level runs the same six stages: `Generate → Score → KeepBestN⁽¹⁾ → Validate & Improve → Score_final → KeepBestN⁽²⁾`.

| Level | $N_T$ | $k_1^T$ | $k_2^T$ |
|---|---|---|---|
| Core belief | 3 | 2 | 1 |
| Fine-grained belief | 5 | 3 | 3 |
| Cognitive distortion | 5 | 5 (keep-all) | 3 |

**Validate & Improve** scores each surviving thought on four clinically grounded dimensions — contrastive balance, hierarchical consistency, definitional alignment against CD-Quest, and counter-questioning fulfillment:

$$S_\text{diag} = 0.2\,q_\text{con} + 0.3\,q_\text{coh} + 0.4\,q_\text{def} + 0.1\,q_\text{counter}$$

Thoughts below $\tau_\text{diag}=0.6$ are refined with dimension-specific feedback (up to 2 attempts), then backtracked to the parent node. Final ranking weights diagnostic validity over surface fluency: $S^\text{final} = S_T + \lambda S_\text{diag}$, $\lambda=3$.

## Results

F1 on [CBT-BENCH](https://github.com/mianzhang/CBT-Bench) Level II tasks, all models integrated with GoTCHA.

| Model | CBT-CD | CBT-PC | CBT-FC |
|---|---|---|---|
| **GPT-4o** | **0.695** | **0.850** | **0.803** |
| Llama-3.1-405B | 0.433 | 0.753 | 0.582 |
| Llama-3.1-70B | 0.439 | 0.758 | 0.554 |
| Llama-3.1-8B | 0.441 | 0.825 | 0.448 |
| Gemma-2-9B | 0.401 | 0.804 | 0.355 |
| Mistral-7B | 0.453 | 0.725 | 0.479 |
| *Human Expert* | *0.491* | *0.776* | *0.546* |

Gains hold without additional training data. Smaller models stay competitive, so structured reasoning does not require large-scale infrastructure.

### Ablation (CBT-CD)

| Configuration | F1 | Prec. | Rec. |
|---|---|---|---|
| Full model | 0.695 | 0.820 | 0.680 |
| w/o Validate & Improve | 0.439 | 0.520 | 0.558 |
| w/o Combined Scoring | 0.523 | 0.634 | 0.597 |
| w/o Distortion Keep-All | 0.613 | 0.779 | 0.635 |

Validate & Improve is the load-bearing component — removing it collapses F1 through cumulative error propagation. Within $S_\text{diag}$, dropping $q_\text{def}$ costs the most ($\Delta$F1 $=-0.074$), justifying its 0.4 weight. Refinement recovers 82.9% of validation failures in a single step, and backtracking fires 3.2× more often on cases with 3+ overlapping distortions.

## Setup

```bash
pip install graph-of-thoughts openai pandas matplotlib seaborn
cp config_template.json config.json   # add your API key
```

```bash
./evaluate_core_beliefs.sh      # CBT-PC
./evaluate_fine_grained.sh      # CBT-FC
./evaluate_distortions.sh       # CBT-CD
./evaluate_datasets.sh --dataset distortions --num-cases 10
```

Options: [CBT_EVALUATION_README.md](CBT_EVALUATION_README.md).

## Data

CBT-BENCH is **not redistributed here** — the source records are first-person mental-health narratives. Obtain it from the original distributor and place it as `CBT-Bench/{core,fine,distortions}/`. [merge.py](merge.py) joins the splits by `id` into inputs and gold labels.

## Citation

```bibtex
@inproceedings{yang2026gotcha,
  author    = {Yang, Hayeon and Hong, Jihuen and Park, Jin Gyeong and Kwon, Kyungmin and Oh, Hayoung},
  title     = {GoTCHA: Hierarchical Graph-of-Thoughts for Explainable Cognitive Distortion Classification},
  booktitle = {Proceedings of the 41st ACM/SIGAPP Symposium on Applied Computing},
  series    = {SAC '26},
  year      = {2026},
  pages     = {192--200},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3748522.3779920}
}
```

## License

BSD 3-Clause. Portions of `cbt/cbt_parser.py`, `cbt/cbt_prompter.py`, and `cbt/__init__.py` derive from the ETH Zurich [Graph of Thoughts](https://github.com/spcl/graph-of-thoughts) reference implementation and retain the original copyright notices. See [LICENSE](LICENSE).
