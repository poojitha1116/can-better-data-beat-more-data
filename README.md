# Can Better Data Beat More Data?

An experimental study investigating how training data quantity and selection affect emotion classification using a zero-shot baseline and LoRA fine-tuning.

## Project Overview

Does a model perform better when it learns from more examples, or can a smaller, carefully selected dataset achieve competitive results?

This project investigates that question using the [Dair-AI Emotion dataset](https://huggingface.co/datasets/dair-ai/emotion) and the **Qwen2.5-1.5B-Instruct** language model.

Four experiments compare zero-shot classification with fine-tuning on different amounts and selections of training data.

## Research Question

**Can a smaller, balanced training dataset outperform a larger random dataset for emotion classification?**

The experiments evaluate accuracy, macro-F1, and invalid-output rate on the same 2,000 test examples.

## Experiments

| Arm | Experiment | Training examples |
|---|---|---:|
| A | Zero-shot baseline | 0 |
| B | LoRA fine-tuning on random examples | 5,000 |
| C | LoRA fine-tuning on balanced, length-filtered examples | 500 |
| D | LoRA fine-tuning on random examples | 500 |

All fine-tuned models use the same base model and LoRA approach. The 500-example experiments use different data-selection strategies.

**Note:** Arm C combines class balancing with text-length filtering, so this experiment cannot isolate the individual effect of either technique.

## Results

All experiments were evaluated on the same 2,000 test examples.

| Arm | Accuracy | Macro-F1 | Invalid outputs |
|---|---:|---:|---:|
| A — Zero-shot | 51.25% | 0.3854 | 19 |
| B — 5,000 random | **91.55%** | **0.8641** | 0 |
| C — 500 balanced/filtered | 60.75% | 0.5618 | 4 |
| D — 500 random | 68.55% | 0.6007 | 1 |

**Best result:** Arm B achieved 91.55% accuracy and a macro-F1 of 0.8641.

Fine-tuning Arm B improved accuracy by 40.30 percentage points over the zero-shot baseline.

Among the two 500-example experiments, Arm D performed better overall than Arm C on both accuracy and macro-F1.

## Emotion Classification

The model classifies text into six emotion categories:

- Sadness
- Joy
- Love
- Anger
- Fear
- Surprise

Arm B achieved its highest per-emotion F1-score for sadness (0.9541) and its lowest for surprise (0.6772). Surprise also had the smallest test support, with 66 examples.

Per-emotion precision, recall, F1-score, and support are available in `per_emotion_results.csv`.

## Methodology

- **Base model:** Qwen/Qwen2.5-1.5B-Instruct
- **Dataset:** Dair-AI Emotion
- **Fine-tuning method:** Low-Rank Adaptation (LoRA)
- **Number of emotion classes:** 6
- **Evaluation set:** 2,000 test examples
- **Evaluation metrics:** Accuracy, macro-F1, per-emotion precision/recall/F1, and invalid-output rate

The fine-tuned models were trained for three epochs. The LoRA configuration used rank 8, alpha 16, and dropout 0.05.

The classification prompt instructed the model to return exactly one of the six emotion labels.

## Key Findings

1. Fine-tuning on 5,000 examples substantially improved performance over zero-shot classification.
2. The random 500-example subset outperformed the balanced, length-filtered 500-example subset in this experiment.
3. A smaller training set can still improve over the zero-shot baseline, but the size and selection strategy matter.
4. Surprise was the most difficult emotion for the best-performing model, based on per-emotion F1-score.

## Limitations

- Only one base model and one test set were evaluated.
- Each data-selection strategy was tested with one selected subset; repeated random seeds were not evaluated.
- Arm C combines balancing and length filtering, so their effects cannot be separated.
- Arm B used a different per-device batch size and gradient-accumulation configuration from Arms C and D, although the effective batch size was 16. This is a potential confounding factor.
- The results do not establish statistical significance or guarantee that the same ranking will hold on other datasets.

## Repository Contents

- `README.md` — Project overview, methodology, results, and limitations.
- `final_experiment_comparison.csv` — Overall comparison of the four experiments.
- `per_emotion_results.csv` — Per-emotion evaluation metrics.
- `arm_A_baseline_predictions.csv` — Zero-shot predictions.
- `arm_B_lora_predictions.csv` — Predictions from the 5,000-example model.
- `arm_C_lora_predictions.csv` — Predictions from the balanced/filtered 500-example model.
- `arm_D_lora_predictions.csv` — Predictions from the random 500-example model.
- `emotion_data_experiment_report.md` — Detailed experiment report.

## Future Improvements

- Repeat experiments with multiple random seeds.
- Compare balanced sampling separately from text-length filtering.
- Investigate data quality and label ambiguity.
- Analyze confusion matrices and errors for each emotion.
- Test whether the findings generalize to other datasets and base models.

## Acknowledgements

This project uses the Dair-AI Emotion dataset and the Qwen2.5-1.5B-Instruct model. Refer to their respective dataset and model documentation for licensing and usage terms.
