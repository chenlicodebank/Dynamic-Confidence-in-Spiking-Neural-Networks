# The implementation of [Dynamic Confidence](([https://openaccess.thecvf.com/content/ICCV2023/html/Li_Unleashing_the_Potential_of_Spiking_Neural_Networks_with_Dynamic_Confidence_ICCV_2023_paper.html])) (ICCV2023) on PyTorch.

# Prologue
These probabilities are a much richer indication of how
the old model generalizes than just the label it thinks is most likely
\- Geoffrey Hinton

# When to use
If you have developed a spiking neural network (SNN) model that has been carefully optimized for latency, but you're looking for even better performance, Dynamic Confidence can help. By integrating Dynamic Confidence at the output of your SNN model, you can reduce latency and spike counts by up to 50% on CIFAR-10 and 30% on ImageNet, without any impact on accuracy.

# Why it is effective
The reason that Dynamic Confidence work so well is because Dynamic Confidence is a simple but super effective runtime optimization technique tailored for SNNs. It allows for varying simulation time steps for different input samples, rather than a fixed time step for all inputs. This means that Dynamic Confidence can provide quicker inference results for simple inputs and more reliable inference for challenging inputs by requesting longer simulation time steps. Overall, Dynamic Confidence's ability to adjust simulation time steps based on input complexity is what makes it so effective at improving SNN model performance.


# How to use
There are four steps to use Dynamic Confidence:
1. Choose any low-latency SNN algorithm (e.g. [QCFS](https://arxiv.org/pdf/2303.04347.pdf), [QFFS](https://www.frontiersin.org/articles/10.3389/fnins.2022.918793/full)...or other low-latency ANN-to-SNN conversion algorithms based on them/similar to them. We also intuitively think Dynamic Confidence can work on SNNs built by surrogate gradients).
2. Add the Dynamic Confidence module at the end of your SNN model.
3. Calculate the confidence threshold to
balance between the latency and accuracy (Do not worry too much about it, we promise that it is the only parameter needed in Dynamic Confidence and it is super insensitive to its value after using the confidence smooth technique proposed in our paper. In fact, finding the most appropriate confidence threshold is the most interesting part of the technique! There are various ways to do it!).
4. Run your SNN with/without Dynamic Confidence, to see how much latency and spike counts can be saved, and whether the accuracy is compromised or not. 

# Code details
The demo code uses the setting of CIFAR-10, 2-bit quantized ResNet-18, ANN-to-SNN conversion, QCFS. It runs an SNN two times, with/without Dynamic Confidence, respectively. Results show 63% less simulation time and 57% reduction in spike counts/image on an NVIDIA V100, with the same accuracy. Please check log.txt for the detailed output and simulation environments. Other settings will be available soon. I am working on wrapping Dynamic Confidence in a function in my spare time, to make it easier to use.


Related works:

* [QCFS](https://arxiv.org/pdf/2303.04347.pdf)(activation quantization + simulate longer to amortize noise);

* [QFFS](https://www.frontiersin.org/articles/10.3389/fnins.2022.918793/full) (activation quantization + completely correct all noise by generating negative spikes in an event-based manner);

* [SRP](https://arxiv.org/pdf/2302.02091.pdf) (activation quantization + identify noise source by running SNN once, and correct the majority of noise when running this SNN for the second time);

* [COS](https://arxiv.org/pdf/2302.10685.pdf) (activation quantization + identify noise source by running SNN once, and correct the majority of noise when running this SNN for the second time);
* There are several other SNN algorithms proposed in the last two years that also have the same core idea. These algorithms can also benefit from Dynamic Confidence.


# Others
For building low-latency SNNs by ANN-to-SNN conversion, the key is activation quantization + noise suppression + better training techniques + implementing max-pooling to prevent multiplication. Here we give some suggestions about how to do these [four steps](https://www.frontiersin.org/articles/10.3389/fnins.2022.918793/full) well:
* activation quantizaiton. Directly adopting an ANN quantization algorithm and modifying it with SNN constraints, do not develop a quantization algorithm from scratch.
* noise suppression. Please check the paper listed above to see how SNN researchers solve this problem and their pros and cons. You can develop your own noise suppression algorithm as well. 
* better training techniques. Refer SOTA ANN papers to find tricks boost your SNN\`s accuracy by improving ANN accuracy before ANN-to-SNN conversion. Some SNN research also provides some good training techniques and tricks to do this, such as [this](https://github.com/putshua/SNN_conversion_QCFS).
* implement max-pooling to prevent the multiplication of an FP32 activation (because average-pooling will average a spike as a non-binary value) and an FP32 weight. You can just use average-pooling and ignore this step if you do not plan to run your SNN algorithms on any neuromorphic hardware.


