Below is the structured format for the provided cases, organized consistently for clarity and comparison. Each case includes model architecture, hyperparameters, and results in a standardized format.

Case 1
Model Architecture:

Type: Seq2seq with encoder-decoder
Encoder: BiLSTM
Decoder: LSTM

Hyperparameters:

Embed Size: 256
Hidden Size: 512
Encoder Layers: 2
Decoder Layers: 4
Dropout: 0.3
Learning Rate: 0.001
Number of Epochs: 20
Epochs Taken: 10
Batch Size: 32

Results:

Training Loss: 7.139
Test Loss: 7.2200
Test Perplexity: 1366.4312
Error Rate: 0.7257
BLEU Score: 0.0024


Case 2
Model Architecture:

Type: Seq2seq with encoder-decoder
Additional Mechanism: Attention
Encoder: Not specified (assumed BiLSTM based on Case 1)
Decoder: Not specified (assumed LSTM based on Case 1)

Hyperparameters:

Embed Size: 300
Hidden Size: 512
Encoder Layers: 2
Decoder Layers: 3
Dropout: 0.2
Learning Rate: 0.0005
Number of Epochs: 20
Batch Size: 32

Results:

Training Loss: 7.274
Validation Loss (Best Model): 7.320
Test Loss: 7.3164
Test Perplexity: 1504.7222
Character Error Rate: 0.9996
BLEU Score: 0.0001


Case 3
Model Architecture:

Type: Seq2seq with encoder-decoder
Encoder: Not specified (assumed BiLSTM based on context)
Decoder: Not specified (assumed LSTM based on context)

Hyperparameters:

Input Size: len(urdu_vocab_map) + 1
Output Size: len(roman_vocab_map) + 1
Embed Size: 128
Hidden Size: 128
Encoder Layers: 1
Decoder Layers: 2
Dropout: 0.1
Learning Rate: 0.0001
Number of Epochs: 20
Epochs per Session: 10
Batch Size: 4
Max Length: 350

Results:

Test Loss: 7.8271
Test Perplexity: 2507.7010
Character Error Rate: 0.9988
BLEU Score: 0.0000


Case 4
Model Architecture:

Type: Seq2seq with encoder-decoder
Encoder: Not specified (assumed BiLSTM based on context)
Decoder: Not specified (assumed LSTM based on context)

Hyperparameters:

Input Size: len(urdu_vocab_map) + 1
Output Size: len(roman_vocab_map) + 1
Embed Size: 128
Hidden Size: 128
Encoder Layers: 1
Decoder Layers: 2
Dropout: 0.1
Learning Rate: 0.0001
Number of Epochs: 20
Epochs per Session: 10
Batch Size: 32
Max Length: 350

Results:

Test Loss: 7.8166
Test Perplexity: 2481.3671
Character Error Rate: 1.0009
BLEU Score: 0.0000
