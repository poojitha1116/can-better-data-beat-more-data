# Can Better Data Beat More Data?
## Emotion Classification with Qwen2.5-1.5B-Instruct

### Abstract

This experiment compares a zero-shot Qwen2.5-1.5B-Instruct model with three
LoRA fine-tuning strategies on the dair-ai/emotion dataset. The experiments
compare 5,000 randomly sampled training examples with two 500-example subsets:
a class-balanced, length-filtered subset and a random subset. All arms were
evaluated on the same 2,000 test examples. The 5,000-example arm achieved the
highest test accuracy and macro-F1. Among the 500-example arms, random sampling
outperformed the balanced/filtered subset overall.

### 1. Research question

Can careful selection of a small training dataset compensate for using fewer
examples in emotion classification? Specifically, how do training quantity,
class balancing, and length filtering relate to test performance?

### 2. Dataset and evaluation

- Dataset: `dair-ai/emotion`
- Classes: sadness, joy, love, anger, fear, surprise
- Test set: 2,000 examples for every arm
- The training split was cleaned to remove exact text overlap with the test set.
- Evaluation metrics: accuracy, macro-F1, and invalid-output rate.
- Macro-F1 gives each emotion equal weight, making it useful when class
  frequencies differ.
- Invalid outputs are generated responses that do not exactly match one of
  the six permitted labels.

The prediction-file audit confirmed that all four files contain the same
2,000 test texts in the same order and have matching true labels.

### 3. Experimental design

- **Arm A — Zero-shot:** original model without LoRA fine-tuning.
- **Arm B — 5,000 random:** LoRA fine-tuning on 5,000 random examples from
  the cleaned training data.
- **Arm C — 500 balanced/filtered:** 500 examples selected with approximately
  equal class counts, after filtering texts by length and removing duplicate
  texts.
- **Arm D — 500 random:** 500 random examples from the same eligible pool
  used for Arm C, without deliberate class balancing.

All fine-tuned arms used the Qwen2.5-1.5B-Instruct base model, LoRA rank 8,
alpha 16, dropout 0.05, learning rate 0.0002, three epochs, and seed 42.

**Training-setting caveat:** Arm B used per-device batch size 4 and gradient
accumulation 4. Arms C and D used per-device batch size 1 and gradient
accumulation 16 in a manual single-GPU loop. Both configurations have an
effective batch size of 16, but the optimization procedures are not identical.
This difference is a confounding factor in the comparison.

### 4. Overall results

| Arm   | Experiment                    |   Training examples |   Test examples |   Accuracy |   Macro-F1 |   Invalid outputs |   Invalid-output rate |
|:------|:------------------------------|--------------------:|----------------:|-----------:|-----------:|------------------:|----------------------:|
| A     | Arm A - Zero-shot             |                   0 |            2000 |     0.5125 |     0.3854 |                19 |                0.0095 |
| B     | Arm B - 5,000 random          |                5000 |            2000 |     0.9155 |     0.8641 |                 0 |                0.0000 |
| C     | Arm C - 500 balanced/filtered |                 500 |            2000 |     0.6075 |     0.5618 |                 4 |                0.0020 |
| D     | Arm D - 500 random            |                 500 |            2000 |     0.6855 |     0.6007 |                 1 |                0.0005 |

### 5. Per-emotion results

| Arm   | Experiment                    | Emotion   |   Precision |   Recall |   F1-score |   Support |
|:------|:------------------------------|:----------|------------:|---------:|-----------:|----------:|
| A     | Arm A - Zero-shot             | sadness   |      0.4984 |   0.8262 |     0.6218 |  581.0000 |
| A     | Arm A - Zero-shot             | joy       |      0.7690 |   0.4216 |     0.5446 |  695.0000 |
| A     | Arm A - Zero-shot             | love      |      0.2617 |   0.4906 |     0.3414 |  159.0000 |
| A     | Arm A - Zero-shot             | anger     |      0.4850 |   0.5309 |     0.5069 |  275.0000 |
| A     | Arm A - Zero-shot             | fear      |      0.7742 |   0.1071 |     0.1882 |  224.0000 |
| A     | Arm A - Zero-shot             | surprise  |      0.5714 |   0.0606 |     0.1096 |   66.0000 |
| B     | Arm B - 5,000 random          | sadness   |      0.9429 |   0.9656 |     0.9541 |  581.0000 |
| B     | Arm B - 5,000 random          | joy       |      0.9314 |   0.9381 |     0.9348 |  695.0000 |
| B     | Arm B - 5,000 random          | love      |      0.8089 |   0.7987 |     0.8038 |  159.0000 |
| B     | Arm B - 5,000 random          | anger     |      0.9137 |   0.9236 |     0.9186 |  275.0000 |
| B     | Arm B - 5,000 random          | fear      |      0.9282 |   0.8661 |     0.8961 |  224.0000 |
| B     | Arm B - 5,000 random          | surprise  |      0.7049 |   0.6515 |     0.6772 |   66.0000 |
| C     | Arm C - 500 balanced/filtered | sadness   |      0.8253 |   0.5284 |     0.6443 |  581.0000 |
| C     | Arm C - 500 balanced/filtered | joy       |      0.8119 |   0.6273 |     0.7078 |  695.0000 |
| C     | Arm C - 500 balanced/filtered | love      |      0.3186 |   0.6792 |     0.4337 |  159.0000 |
| C     | Arm C - 500 balanced/filtered | anger     |      0.5950 |   0.6036 |     0.5993 |  275.0000 |
| C     | Arm C - 500 balanced/filtered | fear      |      0.4708 |   0.6830 |     0.5574 |  224.0000 |
| C     | Arm C - 500 balanced/filtered | surprise  |      0.3125 |   0.6818 |     0.4286 |   66.0000 |
| D     | Arm D - 500 random            | sadness   |      0.7400 |   0.6713 |     0.7040 |  581.0000 |
| D     | Arm D - 500 random            | joy       |      0.7031 |   0.8619 |     0.7744 |  695.0000 |
| D     | Arm D - 500 random            | love      |      0.4476 |   0.4025 |     0.4238 |  159.0000 |
| D     | Arm D - 500 random            | anger     |      0.6836 |   0.6364 |     0.6591 |  275.0000 |
| D     | Arm D - 500 random            | fear      |      0.6667 |   0.5268 |     0.5885 |  224.0000 |
| D     | Arm D - 500 random            | surprise  |      0.5682 |   0.3788 |     0.4545 |   66.0000 |

### 6. Interpretation

1. Arm B achieved the highest overall accuracy and macro-F1. In this set of
   experiments, fine-tuning on 5,000 random examples substantially improved
   performance over the zero-shot baseline.
2. Arm D outperformed Arm C overall despite both using 500 examples.
3. Arm C achieved higher F1 than Arm D for love, but Arm D achieved higher
   F1 for sadness, joy, anger, fear, and surprise.
4. The larger training set had the strongest results for every emotion in
   this experiment. Surprise remained the weakest class for the fine-tuned
   arms and had only 66 examples in the test set.
5. These results do not establish that random sampling is inherently better
   than curation. Arm C combines class balancing with length filtering, so
   the effects of those two choices cannot be isolated from this experiment.

### 7. Limitations

- Only one sampling seed was evaluated per arm; results may vary across
  alternative random samples and training seeds.
- The balanced/filtered strategy combines multiple selection rules.
- Arm B and Arms C/D used different per-device batch and gradient-accumulation
  settings, despite matching effective batch size.
- No confidence intervals or statistical significance tests were calculated.
- The experiment evaluates one base model, one dataset, and one test split.
- A high training-set fit does not by itself guarantee generalization.

### 8. Conclusion

Under the settings tested here, the 5,000-example LoRA model performed best.
With a 500-example budget, random sampling from the eligible training pool
outperformed the balanced, length-filtered selection strategy overall.
These findings support the importance of training-data quantity in this
particular setup, while showing that the selection strategy also matters.
Repeated seeds and experiments that separately vary class balancing and text
filtering would be needed for stronger causal conclusions.

### 9. Reproducibility files

The accompanying ZIP contains this report, overall metrics, per-emotion metrics,
and the four prediction CSVs. The training adapters are saved separately in
the Kaggle working directory.
