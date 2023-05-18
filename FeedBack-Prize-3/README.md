
# Introduction

The goal of this competition is to assess the language proficiency of 8th-12th grade English Language Learners (ELLs). 

The dataset presented in this competition (the ELLIPSE corpus) comprises argumentative essays written by 8th-12th grade English Language Learners (ELLs). The essays have been scored according to six analytic measures: cohesion, syntax, vocabulary, phraseology, grammar, and conventions.

Each measure represents a component of proficiency in essay writing, with greater scores corresponding to greater proficiency in that measure. The scores range from 1.0 to 5.0 in increments of 0.5. Your task is to predict the score of each of the six measures for the essays given in the test set.

# Key takeaways from FB NLP Competition

**DeBERTa is SOTA model for NLP predictive modelling tasks**
This is generally true but do always check (by experimenting) the performance of base model across different architectures such as BERT & RoBERTa. In some few casses, other architectures like RoBERTa might overperform however very unlikely. 
1. Trained DeBERTa model of all sizes from XS to L. The bigger the model, the lower MRSCE score is.

### Training Optimization Techniques
1. Mixed-Precision (FP16)
    - Mixed precision is the combined use of different numerical precisions, both 16-bit and 32-bit floating-point types, in a computational method to make training run faster and use less memory
    - fastai has a detail and well explained note on [this method here](https://docs.fast.ai/callback.fp16.html). Another [resouces](https://towardsdatascience.com/understanding-mixed-precision-training-4b246679c7c4)
2. Gradient Accumulation
    - a technique where one can train on bigger batch sizes than your machine would normally be able to fit into memory. This is done by accumulating gradients over several batches, and only stepping the optimizer after a certain number of batches have been performed. here is a quick tutorial on how to [perform this method using PyTorch](https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/02/19/gradient-accumulation.html)

Techniques to explore SeFit.

### Performance Optimization Techniques
1. Pooling Layers
    - I have found that attention pooling layer works the best over mean, max and min layer for this competiton. There are a bunch of other layers like 

2. Layerwise Learning Rate Decay

Other techniques to explore for future project: 
1. Stochastic Weight Averaging - [SWA](https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/), Pytorch-Lighning has a nice callback to implement this.
2. Reinitialising few last layers

### Inference Optimisation Techniques
I use a DeBERTa XS Model to gurantee fast inference latency. 

Inference Otimisation Techniques available to explore for future project: 
1. KD via Techer Outlier Rejection Loss function (regression), 
2. Weight pruning
3. Quantization

### Evaluation: A road to Robust CV Approach
1. In general CV technique employed does show a positive correlation with the private leaderboard if
    - mean score of a model is outside one std away from the mean score of another model then the different in performance is significant meaning it does have correlation with the private leaderboard performance.
    - example optimised-derberta = baseline deberta despite the mean improvement. but optimised-deberta>deberta-xsmall
    - can double check this hypothesis by submitting roberta-prediction and should expect -> optimised-deberta>roberta-baseline>deberta-xsmall on the leaderboard.

Throughout the experiments, I managed to laverage **W&B** for keeping track of the metrics and numbers. Still new to this technology and will keep exploring and learning.

# Review on top solutions
1. CPMP &  Dieter 22nd solutions
    - neat and simple model.
    - focus on pseudo labelling from previous Competition. deal with the imbalanced pseudo-labelling dataset through weighted samples. use blending of different models for the labelling
    - use a SVR as regression output

2. Beware of the danger possessed by pseudo Labelling on CV
    - pseudo label data contains training data - provides leakage [Remedy] - stash samples with identical ID or use similarity metrics/propensity match and only keep sample below a certain threshold
    - folds use for training data is not the same as folds used in pseudo-labeling
      a. label a new data using models from different folds and take the average of the prediction as a label - indirect leakage
      b. remedy - pseudo on fold-level model and use these to train model in the same fold
    - classification - choose label with high probability - choose a high threshold like >0.99
    - regression - we can use all model from each fold to predict the target and then compute the standard deviation to approximate the confidence score. pick psedo label with the lowest score.
    - trust your cv score with good-care of pseudo-labelling best practices when the Competition sample data is small

3. KD for regression task
    - Distil Loss
      a. [Teacher Outlier Rejection](https://arxiv.org/pdf/2002.12597.pdf)

high impact solutions
1. AWS
2. Pseudo Labelling


### Resources
1. [Optimization approaches for Transformers ](https://www.kaggle.com/code/vad13irt/optimization-approaches-for-transformers)
1. [Submission Time Check](https://www.kaggle.com/code/yasufuminakama/fb3-submission-time)
2. [InnerRank API call](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/368175)

Top Efficiency Track Solution
1. [Turing](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369646)
