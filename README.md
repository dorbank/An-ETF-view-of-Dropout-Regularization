# An-ETF-view-of-Dropout-Regularization
This is the git repository for the paper "An ETF view of Dropout Regularization".

In order to run the experiments for the Penn Tree Bank:
1. Go to the PTB directory.
2. For a run without the coherence regularization, run the "ptb_word_lm" file.
3. For a run with the coherence regularization, run the "ptb_word_lm" file.
Choose the model you want (small\medium) in the constants in the file.

In order to run the experiments for the Fashion MNIST:
1. go to the Fashion MNIST directory.
2. in the LenetMnistGraph file, change the appropriate constant to choose the coherence loss (None\Convolution\FC), and choose the dropout keep_prob.
3. Run the LenetMnist file.


