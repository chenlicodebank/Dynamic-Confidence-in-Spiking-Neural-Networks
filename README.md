The demo code demonstrates using Dynamic Confidence to reduce the inferece latency further on low-latency SNNs. 

Dynamic Confidence allows for varying simulation time steps for different input samples, rather than a fixed time step for all inputs. Dynamic Confidence can provide quicker inference results for simple inputs and more reliable inference for challenging inputs by requesting longer simulation time steps.

Our aim is to provide a valuable technique for SNN researchers who want to improve the inference latency of their SNN algorithms further. 


There are four steps to use Dynamic Confidence,
1. Choose any low-latency SNN algrotihm.
2. Add the Dynamic Confidence module at the end of your SNN model.
3. Calculate the only parameter, confidence threshold, in the Dynamic Confidence module.
4. Run your SNN with/without Dynamic Confidence, to see how much latency and spike counts can be saved. 







By applying Dynamic Confidence, low-latency SNN algorithms such as QCFS and QFFS can achieve a latency reduction of up to 40% on CIFAR-10 and ImageNet. It should be effective on other low-latency SNN algorithms that are based on QCFS and QFFS. See below for details.

* [QCFS](https://arxiv.org/pdf/2303.04347.pdf)(activation quantizaiton + simulate longer to amortize noise);

* [QFFS](https://www.frontiersin.org/articles/10.3389/fnins.2022.918793/full) (activation quantization + completely correct all noise by generating negative spikes in an event-based manner);

* [SRP](https://arxiv.org/pdf/2302.02091.pdf) (activation quantization + identify noise source by running SNN once, and correct the majority of noise when running this SNN for the second time);

* [COS](https://arxiv.org/pdf/2302.10685.pdf) (activation quantization + identify noise source by running SNN once, and correct the majority of noise when running this SNN for the second time);
* ...


The demo code uses the setting of CIFAR-10, ResNet-20, QCFS. Other setting will be available soon.
This code will load a 2-bit ANN and convert it to a SNN and run two times (with/without Dynamic Confidence respectively). The lantecy is reduced from 27 time steps to 11.51 time steps, with the same accuracy of 94.27% on CIFAR-10. Please check log.txt for the expected output and simulation enviorments.
