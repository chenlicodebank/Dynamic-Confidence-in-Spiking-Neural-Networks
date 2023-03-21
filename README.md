The implementation of [Dynamic Confidence](https://arxiv.org/abs/2303.10276) on PyTorch.

If you have developed an SNN model that has been carefully optimized for latency, but you're looking for even better performance, Dynamic Confidence can help. By integrating Dynamic Confidence at the output of your SNN model, you can reduce latency and spike counts by up to 50% on CIFAR-10 and 30% on ImageNet, without any impact on accuracy.


The reason that Dynamic Confidence work so well is becasue that Dynamic Confidence is a simple but super effective runtime optimization technique tailored for SNNs. It allows for varying simulation time steps for different input samples, rather than a fixed time step for all inputs. This means that Dynamic Confidence can provide quicker inference results for simple inputs and more reliable inference for challenging inputs by requesting longer simulation time steps. Overall, Dynamic Confidence's ability to adjust simulation time steps based on input complexity is what makes it so effective at improving SNN model performance.

There are four steps to use Dynamic Confidence:
1. Choose any low-latency SNN algrotihm (e.g. [QCFS](https://arxiv.org/pdf/2303.04347.pdf), [QFFS](https://www.frontiersin.org/articles/10.3389/fnins.2022.918793/full)...or other low-latency algorithms based on them/similar to them).
2. Add the Dynamic Confidence module at the end of your SNN model.
3. Calculate the confidence threshold to
balance between the latency and accuracy (Do not worry too much about it, we promise that it is the only parameter needed in Dynamic Confidence and it is super insensitive to its value. In fact, finding the most appropriate confidence threshold is the most interesting part of the technique!).
4. Run your SNN with/without Dynamic Confidence, to see how much latency and spike counts can be saved, and whether the accuracy is compromised or not. 


The demo code uses the setting of CIFAR-10, 2-bit quantized ResNet-20, QCFS (Please check log.txt for the expected output and simulation enviorments.). Other settings will be available soon. I am working on wrap Dynamic Confidence in a function to make it easier to use.


Related works:

* [QCFS](https://arxiv.org/pdf/2303.04347.pdf)(activation quantizaiton + simulate longer to amortize noise);

* [QFFS](https://www.frontiersin.org/articles/10.3389/fnins.2022.918793/full) (activation quantization + completely correct all noise by generating negative spikes in an event-based manner);

* [SRP](https://arxiv.org/pdf/2302.02091.pdf) (activation quantization + identify noise source by running SNN once, and correct the majority of noise when running this SNN for the second time);

* [COS](https://arxiv.org/pdf/2302.10685.pdf) (activation quantization + identify noise source by running SNN once, and correct the majority of noise when running this SNN for the second time);
* ...



