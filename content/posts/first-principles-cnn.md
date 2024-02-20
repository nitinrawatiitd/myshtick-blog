---
title: "Understanding convolutions from first principles"
date: 2024-02-20
author: Nitin
tags: [cnn][vision]
---

CNNs were and still quite are prevalant for problems in the vision space. First came LeNet that was developed by LeCun et al., 1989, where they successfully trained a CNN via backpropagation. But the first real breakthrough was done by AlexNet in 2012, how CNNs, and deep learning model in general, can reach SoTA performances on tasks that were initially thought impossible. Over the next few years there were various architectures that improved upon those concepts or introduced new ones. Recently vision transformers have become SoTA on many vision tasks. But CNNs remain relavant.

So CNNs are important. But why convolution? Why not train a typical, dense, fully connected neural network?

The main problem is the computational complexity. For a 1000x1000 pixel image, the input will have ${10^6}$ values, and if we go with just 1 hidden layer of a fraction of that size, say 1000 neurons, that would mean we need ${10^6} * {10^3} = {10^{9}}$ parameters. Again, this is is just 1 layer in and we are in 1 billion parameters space. Unless we have lots of GPUs, a talent for distributed optimization, and an extraordinary amount of patience, learning the parameters of this network may turn out to be infeasible.

So we have to look for some shortcuts. Luckily, there are some structures that we as humans use to make sense of an image and the same are exploited by CNNs. The main is spatial invariance i.e, if we want to detect an object in an image, it seems reasonable that whatever method we use to recognize objects should not be overly concerned with the precise location of the object in the image. This is the first principle. 

The other is locality. We can focus on some local aspects of the image first, like the boundaries or eyes etc in the initial layers and then make sense of all of it together in the later layers.

Using these principles, we can contraint a fully connected neural network to fewer parameters. And as we do that, we'll realise that it is becoming more and more like what we call a convolution network. The below set of images capture the process for first a 2x2 image, showing how we contraint the neural network parameters so that they have spatial invariance and then locality.

![](/img/first_principles_2x2.png "First principles illustrated on a 2x2 image. Notice how the number of parameters are reduced with each principle.")

Below set of images better capture how constraining the parameters leads to convolutions on a 3x3 image.

![](/img/first_principles_3x3.png "First principles illustrated on a 3x3 image. How the parameters become a convolution because of locality contraint becomes more apparent here.")

So now we know how constraining the parameters of a FCN leads us to a CNN. The parameters are reduced and shared across multiple regions of the image, and hence we have a much more parameter efficient model that can be trained reasonably. Let's recap how the parameters reduce and compare against a fully connected network for a 2x2 and 2x3 image.

![](/img/cnn_matrix_multiplications.png "How weights or parameters reduce from a FCN to a CNN. Note: In CNN the parameters repeat and hence are lower than the number of colored dots that appear in the illustration")