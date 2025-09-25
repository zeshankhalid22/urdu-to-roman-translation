# Urdu to Roman Urdu Neural Machine Translation

This repository contains our implementation of a Neural Machine Translation (NMT) system for translating Urdu text into Roman Urdu script.

## Team Members

- Zeeshan Khalid (22P-9230)
- Zahid Iqbal (22F-3394)

## Project Overview

We developed a sequence-to-sequence model with attention to translate Urdu poetry and prose into Roman Urdu while preserving meaning and style. This project explores the challenges of cross-script translation between languages with fundamentally different structures.

## Dataset

We used the Rekhta dataset, which contains parallel Urdu and Roman Urdu texts organized by poet. This dataset provided a rich source of literary text for our translation task.

# Methodology

Our approach involved:

1. **Language-Specific Tokenization**: We implemented customized Byte Pair Encoding (BPE) with word boundary markers for Urdu to address the morphological differences between scripts. This involved preventing BPE merges across word boundaries and expanding the Urdu vocabulary size to 4000 tokens.

2. **Seq2Seq with Attention**: We built a model with a Bidirectional LSTM encoder to process Urdu text, Bahdanau attention to focus on relevant input parts, and an LSTM decoder to generate Roman Urdu output token by token.

3. **Data-Driven Parameter Selection**: Based on EDA, we optimized for a maximum sequence length of 350 tokens and tested batch sizes ranging from 8 to 32 to balance computational efficiency with model performance.


# Results

Despite our efforts to tailor tokenization and architecture to the specific linguistic characteristics of Urdu, our model showed very limited success:

| Configuration | Test Loss | Perplexity | BLEU Score | CER |
|---------------|-----------|------------|------------|-----|
| BiLSTM base   | 7.22      | 1366.43    | 0.0024     | 0.7257 |
| With Attention| 7.32      | 1504.72    | 0.0001     | 0.9996 |
| Reduced model | 7.82      | 2507.70    | 0.0000     | 0.9988 |

## Lessons Learned

- **Tokenization Matters**: Standard NLP preprocessing techniques often fail with languages having fundamentally different structures.
- **Data Representation > Model Complexity**: Increasing model complexity didn't overcome fundamental data representation issues.
- **Script-Specific Challenges**: Urdu's non-concatenative morphology requires specialized processing that differs from approaches used for Latin-based scripts.

## Tools & Libraries

- PyTorch
- Pandas, NumPy
- Google Colab (T4 GPU)
- Kaggle (T4 GPU)
- Streamlit

# Future Improvements

- Implement specialized morphological analysis for Urdu
- Explore transformer-based architectures
- Leverage pre-trained multilingual models (mBERT, XLM-R)
- Expand training data through augmentation techniques


## Links

- [Streamlit App](https://urdu-to-roman-translation-66.streamlit.app/)
- [Google Colab Notebook](https://colab.research.google.com/drive/1odecof0SpMZrbjyJuNahW-sR_maPsSvI?usp=sharing)

## Instructor

Dr. Muhammad Usama
