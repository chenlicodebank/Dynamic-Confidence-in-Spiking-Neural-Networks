# Dynamic-Confidence-in-Spiking-Neural-Networks
The implementation of Dynamic Confidence on PyTorch.

The demo code demonstrates how a runtime optimization technique, Dynamic Confidence, can help reduce the inferece latency further on low-latency SNNs. Our hope is to encourage the use of this technique when a SNN needs a lower latency, as it allows for varying simulation time steps for different input samples, rather than using a fixed time step for all inputs. This approach has proven to be remarkably effective and could greatly benefit SNNs in various applications.

This algorithm can enable ~40% latency reduction on low-latency SNN algorithms such as QCFS and QFFS, whose essential technique is activation quantization and noise suppression on CIFAR-10 and ImageNet.

Including:

[QCFS](https://arxiv.org/pdf/2303.04347.pdf)(activation quantizaiton + simulate longer to amortize noise);

[QFFS](https://www.frontiersin.org/articles/10.3389/fnins.2022.918793/full) (activation quantization + completely correct all noise by generating negative spikes in an event-based manner);

[SRP](https://arxiv.org/pdf/2302.02091.pdf) (activation quantization + identify noise source by running SNN once, and correct the majority of noise when running this SNN for the second time);

[COS](https://arxiv.org/pdf/2302.10685.pdf) (activation quantization + identify noise source by running SNN once, and correct the majority of noise when running this SNN for the second time);
...


The demo code uses the setting of CIFAR-10, ResNet-20, QCFS. Other setting will be available soon.
This code will load a 2-bit ANN and convert it to a SNN and run two times (with/without Dynamic Confidence respectively). The lantecy is reduced from 27 time steps to 11.51 time steps, with the same accuracy of 94.27% on CIFAR-10. Please check log.txt for the expected output and simulation enviorments.
