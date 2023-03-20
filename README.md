The demo code demonstrates using Dynamic Confidence to reduce the inferece latency further on a given low-latency SNN. 

Dynamic Confidence allows for varying simulation time steps for different input samples, rather than a fixed time step for all inputs. Dynamic Confidence can provide quicker inference results for simple inputs and more reliable inference for challenging inputs by requesting longer simulation time steps.

Our aim is to provide a valuable technique for SNN researchers who has built a low-latency SNN and want to further reduce its latency and spike counts. According to our experiments, Dynamic Confidence can bring about 50% latency and spike counts reduction on CIFAR-10, and 30% on ImageNet.



There are four steps to use Dynamic Confidence:
1. Choose any low-latency SNN algrotihm (e.g. [QCFS](https://arxiv.org/pdf/2303.04347.pdf), [QFFS](https://www.frontiersin.org/articles/10.3389/fnins.2022.918793/full)...or other algorithms based on them).
2. Add the Dynamic Confidence module at the end of your SNN model.
3. Calculate the only parameter, confidence threshold, in the Dynamic Confidence module (Do not worry too much on this step, this parameter is super robust to its value).
4. Run your SNN with/without Dynamic Confidence, to see how much latency and spike counts can be saved. 


The demo code uses the setting of CIFAR-10, ResNet-20, QCFS (Please check log.txt for the expected output and simulation enviorments.
). Other settings will be available soon.


Related works:

* [QCFS](https://arxiv.org/pdf/2303.04347.pdf)(activation quantizaiton + simulate longer to amortize noise);

* [QFFS](https://www.frontiersin.org/articles/10.3389/fnins.2022.918793/full) (activation quantization + completely correct all noise by generating negative spikes in an event-based manner);

* [SRP](https://arxiv.org/pdf/2302.02091.pdf) (activation quantization + identify noise source by running SNN once, and correct the majority of noise when running this SNN for the second time);

* [COS](https://arxiv.org/pdf/2302.10685.pdf) (activation quantization + identify noise source by running SNN once, and correct the majority of noise when running this SNN for the second time);
* ...


