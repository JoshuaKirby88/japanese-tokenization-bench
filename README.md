# Research

## Topic

Effects of different tokenization strategies on Japanese text on LLM text analysis tasks.

## Method

### Phase 1: Compare different tokenization strategies

**Tokenization Strategies:**

- Baseline: "猫が魚を食べた。"
- Character-Level De-Tokenization: "猫 が 魚 を 食 べ た 。"
- Morphology-Level De-Tokenization: "猫 が 魚 を 食べ た。"

**Steps:**

1. Feed input to LLM
2. Get response
3. Check against answer, and pick out winners

### Phase 2: Compare implementation strategies (Static VS Dynamic)

**Implementation Strategies:**

- Static: Pre-process the input based on the best tokenization strategy from phase 1.
- Dynamic: Instruct the LLM to apply the chosen tokenization strategy on text before performing any analysis.

## Hypothesis

De-Tokenization will improve LLM's ability to perform fine-grained analysis on individual characters.

## Details

**Models:**

[undecided]

**Dataset:**

- JCommonsenseQA (shunk031/JGLUE): Multiple choice questions on Japanese language, e.g. "What words are used when meeting someone for the first time?"
- JNLI (shunk031/JGLUE): True/False questions on whether the premise logically implies the hypothesis, e.g. Premise="A man is running." Hypothesis="A person is moving."
- JSQuAD (shunk031/JGLUE): Needle-in-a-haystack, e.g. Question="When did the capital move to Tokyo?" Answer="1868"
- Synthetic Data (character counting): Count number of occurrences of "が" in "私は蝶が好きですが、蛾が嫌いです。"
- Japanese Wikipedia Typo Dataset: Question="Identify all typos in ..." Answer="[Typo] -> [Correction]"
- Grammar pattern extraction (from my other project with ~600 samples): Extract all "〜たり〜たり" from "..."

## Related Papers

- "Inconsistent Tokenizations Cause Language Models to be Perplexed by Japanese Grammar"

## Other Variants

- Effects of "Forced De-Tokenization" on reasoning models
    - Length of reasoning tokens
    - Accuracy of final answer
    - Compare against hybrid dynamic de-tokenization
- Effects of "Forced De-Tokenization" on hallucination
- When does Forced De-Tokenization start to degrade output
    - Test on varying lengths of analysis target text.
