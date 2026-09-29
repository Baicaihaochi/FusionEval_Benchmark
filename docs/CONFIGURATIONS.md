# Configurations

| Method | Qwen3-4B | Qwen3-8B |
|---|---|---|
| Soup | Uniform mean | Uniform mean |
| TA | Scale 0.3 | Scale 0.3 |
| TIES | Density 0.2, scale 1.0 | Density 0.2, scale 0.9 |
| DARE | Drop 0.2, scale 0.3 | Drop 0.1, scale 0.3 |
| DELLA | Drop 0.1, window 0.14, scale 1.0 | Same |
| L&S | Retained fraction 0.40 | Same |
| Fisher | 8 predictive samples, floor 1e-6 on all experts | 4 predictive samples, floor 1e-6 on first expert |
| AdaMerging | Init. 0.3, learning rate 3e-4, 500 steps, mean embeddings | Same |
| RegMean | Alpha 0.7, FP32 Gram, exact solve, averaged tied head | Same, with regression of the untied output head |
| TA + FeatCal | Lambda 1.0, rho 1.5, alpha 0.25 | Lambda 0.1, rho 1.5, alpha 0.1 |
| TA + Surgery | Rank 16, learning rate 6e-5, 250 steps | Rank 32, learning rate 1e-4, 500 steps |

Randomized methods use seed 42. Fisher uses equal coefficients and normalized
importance. Continual TIES uses density 0.3 and scale 1.0, separately from the
joint-fusion configuration.


## Inputs

Experts share architecture, tokenizer and vocabulary; their order is Math, Code, Agent, IF, Science. Fisher's first-expert floor refers to Math.

Fisher and AdaMerging use the prompt before the final assistant response; RegMean uses the constructed sequence. These inputs retain the first 4096 tokens.

FeatCal and Surgery use complete feature sequences up to 32768 tokens and reject longer inputs without truncation.
