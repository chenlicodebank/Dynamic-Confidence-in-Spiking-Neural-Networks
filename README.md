# Dynamic-Confidence-in-Spiking-Neural-Networks
The implementation of Dynamic Confidence on PyTorch.

The demo code demonstrates how a runtime optimization technique, Dynamic Confidence, can help reduce the inferece latency further on low-latency SNNs. Our hope is to provide a useful technique to SNNs when lower latency is desirable. The essence of this technique is allowing for varying simulation time steps for different input samples, rather than using a fixed time step for all inputs. By which, simple inputs can get inference results within only a few time steps, and difficult inputs can ask for longer simulation time steps to generate a more relibale inference. 


Dynamic Confidence can enable ~40% latency reduction on CIFAR-10 and ImageNet on low-latency SNN algorithms QCFS and QFFS, and it should be effective on other low-latency SNN algorithms that are based on QCFS and QFFS. See below for details.

[QCFS](https://arxiv.org/pdf/2303.04347.pdf)(activation quantizaiton + simulate longer to amortize noise);

[QFFS](https://www.frontiersin.org/articles/10.3389/fnins.2022.918793/full) (activation quantization + completely correct all noise by generating negative spikes in an event-based manner);

[SRP](https://arxiv.org/pdf/2302.02091.pdf) (activation quantization + identify noise source by running SNN once, and correct the majority of noise when running this SNN for the second time);

[COS](https://arxiv.org/pdf/2302.10685.pdf) (activation quantization + identify noise source by running SNN once, and correct the majority of noise when running this SNN for the second time);
...


The demo code uses the setting of CIFAR-10, ResNet-20, QCFS. Other setting will be available soon.
This code will load a 2-bit ANN and convert it to a SNN and run two times (with/without Dynamic Confidence respectively). The lantecy is reduced from 27 time steps to 11.51 time steps, with the same accuracy of 94.27% on CIFAR-10. Please check log.txt for the expected output and simulation enviorments.
