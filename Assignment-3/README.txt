1. How to run the POS tagger:
============================

$ python3 python3 murthy_sunil_PA3.py train_file.txt test_file.txt 

After running the above script, it creates a berp-out.txt file containing the predicted tags for the test sentences. The format is same as the training file i.e "word <tab> tag".


2. How to evaluate the HMM Tagger (evalPOSTagger script):
==========================================================

Now evaluate the prediction rate using, evalPOSTagger.py as,

$ python evalPOSTagger berp-key.txt berp-out.txt


3. Advanced evaluation (POSstats script):
==========================================

You can generate confusion matrix, better accuracy score and precision by running POSstats.py file as,

$ python3 POSstats.py berp-key.txt berp-out.txt 

The above scipt produces confusion_matrix.pdf file, prints accuracy and precision on the terminal.

3.1 sample out:
==================

Confusion matrix, without normalization
[[  2   0   0 ...,   0   0   0]
 [  0 455   0 ...,   0   0   0]
 [  0   0 838 ...,   0   0   0]
 ..., 
 [  0   0   0 ...,  84   7   0]
 [  0   0   0 ...,   1 149   0]
 [  0   0   0 ...,   0   0 233]]
Accuracy = 0.956999290171
Precision = 0.958743576703