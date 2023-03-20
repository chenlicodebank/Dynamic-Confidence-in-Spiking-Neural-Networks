The demo code is about using Dynamic Confidence to reduce the inferece latency further on a given low-latency SNN. 

Our aim is to: 
1. demonstrating the effectiveness of dynamic strategy (aka. dynamic networks, adaptive inference...) on reducing latency and spike counts of SNNs.
2. providing demo codes, so researchers can quickly get started to use dynamic strategy, and get good understanding on what needs to be careful when applying dynamic strategy to SNNs.

Dynamic Confidence is particularly interesting when applying it to a SNN model that have already been carefully
optimized in terms of latency. According to our experiments, Dynamic Confidence can bring about 50% latency and spike counts reduction on CIFAR-10, and 30% on ImageNet, without accuracy loss.

Dynamic Confidence allows for varying simulation time steps for different input samples, rather than a fixed time step for all inputs. Dynamic Confidence can provide quicker inference results for simple inputs and more reliable inference for challenging inputs by requesting longer simulation time steps.


There are four steps to use Dynamic Confidence:
1. Choose any low-latency SNN algrotihm (e.g. [QCFS](https://arxiv.org/pdf/2303.04347.pdf), [QFFS](https://www.frontiersin.org/articles/10.3389/fnins.2022.918793/full)...or other low-latency algorithms based on them/similar to them).
2. Add the Dynamic Confidence module at the end of your SNN model.
3. Calculate the confidence threshold in the Dynamic Confidence module (Do not worry too much about this step, the confidence threshold is super parameter-insensitive).
4. Run your SNN with/without Dynamic Confidence, to see how much latency and spike counts can be saved. 


The demo code uses the setting of CIFAR-10, 2-bit quantized ResNet-20, QCFS (Please check log.txt for the expected output and simulation enviorments.
). Other settings will be available soon. I am working on wrap Dynamic Confidence in a function to make it easier to use.


Related works:

* [QCFS](https://arxiv.org/pdf/2303.04347.pdf)(activation quantizaiton + simulate longer to amortize noise);

* [QFFS](https://www.frontiersin.org/articles/10.3389/fnins.2022.918793/full) (activation quantization + completely correct all noise by generating negative spikes in an event-based manner);

* [SRP](https://arxiv.org/pdf/2302.02091.pdf) (activation quantization + identify noise source by running SNN once, and correct the majority of noise when running this SNN for the second time);

* [COS](https://arxiv.org/pdf/2302.10685.pdf) (activation quantization + identify noise source by running SNN once, and correct the majority of noise when running this SNN for the second time);
* ...



