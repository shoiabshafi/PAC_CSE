# PAC-CSE

This folder contains the final training-free PAC-CSE pipeline. Model response
generation is separated from scoring so model-specific processors and
checkpoints are explicit.

For each case, the target produces one held-out greedy answer, ten clean
stochastic answers, and ten answers after fixed content-preserving corruptions.
A fixed peer produces ten stochastic answers on the original image. The greedy
answer is used only for GREEN labeling and is excluded from PAC-CSE clustering.
All responses are clustered with bidirectional DeBERTa-MNLI entailment.

