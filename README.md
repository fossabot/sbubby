# Influence Functions for Keras models**
> A Keras-compatible implementation of "Understanding Black-box Predictions via Influence Functions": https://arxiv.org/abs/1703.04730

This repository is an open source toolkit to understand deep learning models better. Deep learning is often referred as a black-box that is difficult to understand.
But, accountability and controllability could be critical to commercialize deep learning models. People often think that high accuracy on prepared dataset 
is enough to use the model for commercial products. However, well-performing models on prepared dataset often fail in real world usages and cause corner cases 
to be fixed. Moreover, it is necessary to explain the result to trust the system in some applications such as medical diagnosis, financial decisions, etc. We hope  
Influence functions can help you to understand the trained models, which could be used to debug failures, interpret decisions, and so on. 

Here, we provide functions to analyze deep learning model decisions easily applicable to any Tensorflow models (other models to be supported later).
Influence score can be useful to understand the model through training samples. The score can be used for filtering bad training samples that affects test performance negatively. 
It is useful to prioritize potential mislabeled examples to be fixed, and debug distribution mismatch between train and test samples.

## Demo 
- _Coming soon_

## Dependencies
- [Tensorflow](https://github.com/tensorflow/tensorflow)>=1.3.0

## Installation
Install alone
```bash
git clone https://github.com/matthew-mcateer/influence_functions
```

## Examples 
- _Coming soon_

## API Documentation
- _Coming soon_

## Communication
- [Issues](https://github.com/matthew-mcateer/influence_functions/issues): report issues, bugs, and request new features
- [Pull request](https://github.com/matthew-mcateer/influence_functions/pulls)
- Discuss: [Gitter](https://gitter.im/matthew-mcateer/influence_functions?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
- Email: [matthewmcateer0@gmail.com](mailto:matthewmcateer0@gmail.com) 

## Authors
[Matthew McAteer](https://github.com/matthew-mcateer)

## License
Apache License 2.0

## References

[1] Cook, R. D. and Weisberg, S. "[Residuals and influence in regression](https://www.casact.org/pubs/proceed/proceed94/94123.pdf)", New York: Chapman and Hall, 1982

[2] Koh, P. W. and Liang, P. "[Understanding Black-box Predictions via Influence Functions](https://arxiv.org/abs/1703.04730)" ICML2017

[3] Pearlmutter, B. A. "[Fast exact multiplication by the hessian](http://www.bcl.hamilton.ie/~barak/papers/nc-hessian.pdf)" Neural Computation, 1994

[4] Agarwal, N., Bullins, B., and Hazan, E. "[Second order stochastic optimization in linear time](https://arxiv.org/abs/1602.03943)" arXiv preprint arXiv:1602.03943

[5] Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra "[Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)" ICCV2017
