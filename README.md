# Image classification

Image classification is one of the most common tasks in pattern recognition. Moreover, the image classification model may be the first model built for many people. Most people expect to remember to build the first convolutional neural network on the handwriting recognition dataset. The traditional classifiers have been suffering from the weak ability of feature extractor, and deep learning is very effective in feature extraction from unstructured data. The model structure in image recognition is very diverse, and we mainly implement several epoch-making models.

## Environment Settings

In the experiment, we use the Tensorflow to implement the proposed deep-based models, and the training process runs on a GTX1080 GPU single machine. 

1. scikit-learn==0.19.1
2. tensorflow-gpu==1.10.1
3. tensorboard==1.10.0

In this survey, we have only selected two more common image datasets:

* *MNIST* [**MNIST**](http://yann.lecun.com/exdb/mnist/) The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples.

* *Cifar10* [**Cifar10**](https://www.cs.toronto.edu/~kriz/cifar.html) The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

## Experimental Results

Before the rise of deep learning, there have been a large number of algorithms for image classification tasks, which can be referred to the website [**MNIST**](http://yann.lecun.com/exdb/mnist/). In the deep learning framework i.e. Tensorflow, the simple algorithms, *Logistic Regression* and *MLP*, are implemented to solve handwriting recognition. 

Among such a large number of models, Alexnet (Krizhevsky et al.) is the first version of deep convolution model used in the image classification competition ILSVRC-2012. Among such a large number of models, Alexnet is the first version of deep convolution model used in the image classification competition. Based on AlexNet, VGGNet (Simonyan and Zisserman 2014) deepens the number of layers and reduces the size of the convolution kernel in the model, and greatly improves the classification accuracy of the model. ResNet (He et al. 2015) redefines the baseline in image classification, introducing a new convolutional architecture, namely direct map, which alleviates the gradient disappearance encountered when the VGGNet layer is deepened. The use of multi-channel small-scale kernel instead of the large-scale kernel can improve the generalization of the model, such that Google proposes many variant models based on this design concept (Szegedy et al. 2015).



## References

1. ImageNet Classiﬁcatio nwith Deep Convolutional Neural Networks, [Krizhevsky et al. (2012)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
2. Very Deep Convolutional Networks for Large-Scale Image Recognition, [Simonyan and Zisserman(2014)](https://arxiv.org/pdf/1409.1556)
3. Deep Residual Learning for Image Recognition, [He et al.(2015)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
4. Going deeper with convolutions, [Szegedy et al.(2015)](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf)