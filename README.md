# Dynamic-Confidence-in-Spiking-Neural-Networks
The implementation of Dynamic Confidence on PyTorch.

The demo code demonstrates how a runtime optimization technique, Dynamic Confidence, can help reduce the inferece latency further on low-latency SNNs. Our vision is that all SNNs should apply runtime optimization technique to let different input samples have different simulation time steps (instead of the same). This is simply because this new method is amazing and super effective!

This algorithm is expected to enable ~40% latency reduction on all current low-latency SNN algorithms whose essential technique is activation quantization and noise suppression on CIFAR-10 and ImageNet.

Including:
QCFS (activation quantizaiton + simulate longer to amortize noise);

QFFS (activation quantization + completely correct all noise by generating negative spikes in an event-based manner);

SRP (activation quantization + identify noise source by running SNN once, and correct the majority of noise when running this SNN for the second time);

COS (activation quantization + identify noise source by running SNN once, and correct the majority of noise when running this SNN for the second time);
...


The demo code uses the setting of CIFAR-10, ResNet-20, QCFS. Other setting will be available soon.


