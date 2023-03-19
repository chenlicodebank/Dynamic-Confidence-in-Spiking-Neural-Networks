# Dynamic-Confidence-in-Spiking-Neural-Networks
The implementation of Dynamic Confidence on PyTorch.

The demo code demonstrates how a runtime optimization technique, Dynamic Confidence, can help reduce the inferece latency further on low-latency SNNs. Our goal is to encourage the use of this technique across all SNNs, as it allows for varying simulation time steps for different input samples, rather than using a fixed time step for all. This approach has proven to be remarkably effective and could greatly benefit SNNs in various applications.

This algorithm is expected to enable ~40% latency reduction on the most of low-latency SNN algorithms whose essential technique is activation quantization and noise suppression on CIFAR-10 and ImageNet.

Including:

QCFS (activation quantizaiton + simulate longer to amortize noise);

QFFS (activation quantization + completely correct all noise by generating negative spikes in an event-based manner);

SRP (activation quantization + identify noise source by running SNN once, and correct the majority of noise when running this SNN for the second time);

COS (activation quantization + identify noise source by running SNN once, and correct the majority of noise when running this SNN for the second time);
...


The demo code uses the setting of CIFAR-10, ResNet-20, QCFS. Other setting will be available soon.
This code will load a 2-bit ANN and convert it to a SNN and run two times (with/without Dynamic Confidence respectively). The lantecy is reduced from 27 time steps to 11.51 time steps, with the same accuracy of 94.27% on CIFAR-10.

 
The output should looks like below:
Accuracy of the fake quant ANN on the 10000 test images: 93.79 %
quant model is above...............
activation1.scale
layer1.0.activation1.scale
layer1.0.activation2.scale
layer1.1.activation1.scale
layer1.1.activation2.scale
layer2.0.activation1.scale
layer2.0.activation2.scale
layer2.1.activation1.scale
layer2.1.activation2.scale
layer3.0.activation1.scale
layer3.0.activation2.scale
layer3.1.activation1.scale
layer3.1.activation2.scale
layer4.0.activation1.scale
layer4.0.activation2.scale
layer4.1.activation1.scale
layer4.1.activation2.scale
activation2.scale
[0.188307523727417, 0.16721558570861816, 0.23476727306842804, 0.1578773558139801, 0.2553340196609497, 0.15781289339065552, 0.2476247102022171, 0.16528929769992828, 0.2881423532962799, 0.16605308651924133, 0.19668614864349365, 0.14066271483898163, 0.24676495790481567, 0.09747104346752167, 0.14194826781749725, 0.024684183299541473, 0.4332151710987091, 2.3383164405822754]
[0.564922571182251, 0.5016467571258545, 0.7043018192052841, 0.4736320674419403, 0.7660020589828491, 0.47343868017196655, 0.7428741306066513, 0.49586789309978485, 0.8644270598888397, 0.498159259557724, 0.590058445930481, 0.4219881445169449, 0.740294873714447, 0.292413130402565, 0.42584480345249176, 0.07405254989862442, 1.2996455132961273, 7.014949321746826]
18
th is above...............
modelsize is below...............
start time (without Dynamic Confidence)
Wed Mar 15 21:36:45 2023
end time (without Dynamic Confidence)
Wed Mar 15 21:53:52 2023
The averaged spike/neuron per inference
3.0227136611938477
Accuracy of the SNN on the 10000  images: 94.270
Latency 1  Accuracy 78.76
Latency 2  Accuracy 84.28
Latency 3  Accuracy 87.07
Latency 4  Accuracy 88.89
Latency 5  Accuracy 90.11
Latency 6  Accuracy 90.96
Latency 7  Accuracy 91.68
Latency 8  Accuracy 92.17
Latency 9  Accuracy 92.49
Latency 10  Accuracy 92.79
Latency 11  Accuracy 93.08
Latency 12  Accuracy 93.18
Latency 13  Accuracy 93.4
Latency 14  Accuracy 93.53
Latency 15  Accuracy 93.63
Latency 16  Accuracy 93.76
Latency 17  Accuracy 93.85
Latency 18  Accuracy 93.92
Latency 19  Accuracy 94.01
Latency 20  Accuracy 94.01
Latency 21  Accuracy 94.08
Latency 22  Accuracy 94.13
Latency 23  Accuracy 94.22
Latency 24  Accuracy 94.23
Latency 25  Accuracy 94.22
Latency 26  Accuracy 94.22
Latency 27  Accuracy 94.27
SNN simulation without Dynamic Confidence end
start time (with Dynamic Confidence)
Wed Mar 15 21:53:52 2023
end time (with Dynamic Confidence)
Wed Mar 15 22:01:23 2023
The averaged spike/neuron per inference
1.2946898937225342
Latency 1.0  Accuracy 78.76
Latency 2.0  Accuracy 84.28
Latency 3.0  Accuracy 87.07
Latency 4.0  Accuracy 88.89
Latency 5.0  Accuracy 90.11
Latency 6.0  Accuracy 90.96
Latency 7.0  Accuracy 91.68
Latency 7.9988  Accuracy 92.17
Latency 8.9594  Accuracy 92.49
Latency 9.653  Accuracy 92.79
Latency 10.047  Accuracy 93.08
Latency 10.3062  Accuracy 93.18
Latency 10.5022  Accuracy 93.4
Latency 10.6575  Accuracy 93.53
Latency 10.7855  Accuracy 93.63
Latency 10.8938  Accuracy 93.76
Latency 10.9872  Accuracy 93.85
Latency 11.0685  Accuracy 93.92
Latency 11.1401  Accuracy 94.01
Latency 11.2036  Accuracy 94.01
Latency 11.2605  Accuracy 94.08
Latency 11.3108  Accuracy 94.13
Latency 11.3578  Accuracy 94.21
Latency 11.4004  Accuracy 94.22
Latency 11.44  Accuracy 94.21
Latency 11.4759  Accuracy 94.21
Latency 11.5097  Accuracy 94.27
Accuracy of the SNN on the 10000  images: 94.270
SNN simulation with Dynamic Confidence end

