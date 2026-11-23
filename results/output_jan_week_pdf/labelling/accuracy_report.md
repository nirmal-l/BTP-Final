# Sentiment methods vs hand labels

Labelled articles: 48. Methods compared: transformer, vader, llm.

## Verdict

Highest macro-F1: **llm** (0.63). Significant differences: vader vs llm (p=0.0169)

## Overall

| method                     |   n |   accuracy |   acc_95ci_low |   acc_95ci_high |   macro_f1 |   cohen_kappa |
|:---------------------------|----:|-----------:|---------------:|----------------:|-----------:|--------------:|
| transformer                |  48 |      0.583 |          0.443 |           0.712 |      0.336 |         0.068 |
| vader                      |  48 |      0.396 |          0.27  |           0.537 |      0.387 |         0.119 |
| llm                        |  48 |      0.646 |          0.504 |           0.766 |      0.63  |         0.451 |
| baseline: always 'neutral' |  48 |      0.583 |          0.443 |           0.712 |      0.246 |         0     |

* accuracy = share of articles where the method's label equals yours
* macro-F1 = average F1 over positive/neutral/negative - not fooled by a method that just says 'neutral' to everything
* Cohen's kappa = agreement beyond chance (0 = chance, 1 = perfect)

## Is the difference real? (exact McNemar test)

| method_a    | method_b   |   n_common |   only_a_correct |   only_b_correct |   p_value | significant_at_0.05   |
|:------------|:-----------|-----------:|-----------------:|-----------------:|----------:|:----------------------|
| transformer | vader      |         48 |               16 |                7 |    0.0931 | False                 |
| transformer | llm        |         48 |               10 |               13 |    0.6776 | False                 |
| vader       | llm        |         48 |                5 |               17 |    0.0169 | True                  |

## Accuracy by page type and topic

| group_by    | group                               |   n |   transformer_accuracy |   vader_accuracy |   llm_accuracy |
|:------------|:------------------------------------|----:|-----------------------:|-----------------:|---------------:|
| page_type   | editorial                           |  15 |                  0.4   |            0.467 |          0.733 |
| page_type   | front                               |  27 |                  0.667 |            0.444 |          0.667 |
| page_type   | other                               |   6 |                  0.667 |            0     |          0.333 |
| topic_label | bjp / hindu / ambedkar / minister   |  11 |                  0.545 |            0.636 |          0.818 |
| topic_label | india / coffee / gas / government   |   8 |                  0.125 |            0.25  |          0.75  |
| topic_label | state / trump / hospital / manipur  |  11 |                  0.636 |            0.364 |          0.364 |
| topic_label | team / test / gambhir / bumrah      |   5 |                  0.8   |            0.4   |          0.8   |
| topic_label | women / iits / iit / india          |   5 |                  0.8   |            0.4   |          0.4   |
| topic_label | workers / data / security / digital |   8 |                  0.75  |            0.25  |          0.75  |

Small groups (n < 10) are anecdotal.

## Confusion matrices (rows = your label)

### transformer

| your label   |   positive |   neutral |   negative |
|:-------------|-----------:|----------:|-----------:|
| positive     |          0 |         9 |          0 |
| neutral      |          1 |        26 |          1 |
| negative     |          0 |         9 |          2 |

### vader

| your label   |   positive |   neutral |   negative |
|:-------------|-----------:|----------:|-----------:|
| positive     |          4 |         2 |          3 |
| neutral      |         13 |        10 |          5 |
| negative     |          4 |         2 |          5 |

### llm

| your label   |   positive |   neutral |   negative |
|:-------------|-----------:|----------:|-----------:|
| positive     |          5 |         1 |          3 |
| neutral      |          3 |        16 |          9 |
| negative     |          0 |         1 |         10 |
