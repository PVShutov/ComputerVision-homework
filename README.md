## Computer Vision homework

#### 1. Gaussian Blur

![Blur](/images/1.png?raw=true)

#### 2. Multilayer Perceptron

* iterations count = __60000__
* batch size = __1__ 
* dataset: __MNIST__
* accuracy = __0.9436__

MLP architecture:
1. Fully-connected layer [784✕800]
2. ReLU activation
3. Fully-connected layer [800✕10]
4. Softmax 

![MLP](/images/2.png?raw=true)


#### 3. Conditional DC-GAN

* dataset: __MNIST__

![DCGAN](/images/3.png?raw=true)


#### 4. MiniVGG

* dataset: __CIFAR-10__

![CIFAR-10](/images/4.png?raw=true)


#### 5. Object Localization

* batch_size: 128
* Accuracy calc by samples with IoU >= 0.5
![Localization_1](/images/5_1.png?raw=true)

Worst sample on test set (IoU = 0.00599):
![Localization_2](/images/5_2.png?raw=true)

Best sample on test set (IoU = 0.98055):
![Localization_3](/images/5_3.png?raw=true)

* Accuracy on test set = 0.9534
* Only classification accuracy = 0.9862


#### 6. MAP

* MAP: 0.9653

#### 7. Line detection

![Lines](/images/7.png?raw=true)