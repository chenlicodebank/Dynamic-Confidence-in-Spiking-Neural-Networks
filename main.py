import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import copy
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torch.quantization.fake_quantize
from all_models import*
import time
import torch.quantization
import os


# Hyper-parameters
activation_bits=2
num_epochs_for_qat = 40
batch_size_train = 32
learning_rate = 0.01
number_of_classes=10
num_of_layer=18

log_file_name="log.txt"
valid_size=5000
batch_size_test = 1
quan_first=""
quan_last=""
simulation_time=27
snn_quant_scale=3

# create a log file
log = open(os.path.join('./', log_file_name), 'w')
def print_log(print_string, log):
  print("{}".format(print_string))
  log.write('{}\n'.format(print_string))
  log.flush()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




mean=(0.4914, 0.4822, 0.4465)
std=(0.2023, 0.1994, 0.2010)
train_transform = transforms.Compose(
    [ transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), transforms.ToTensor(),
     transforms.Normalize(mean, std)])

test_transform = transforms.Compose(
    [transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize(mean, std)])

train_dataset = torchvision.datasets.CIFAR10(root='../Dataset/',
                                           train=True,
                                           transform=train_transform,
                                           download=True)
test_dataset = torchvision.datasets.CIFAR10(root='../Dataset/',
                                          train=False,
                                          transform=test_transform,
                                          download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size_train,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size_test,
                                          shuffle=False)


def create_datasets(batch_size):
    # percentage of training set to use as validation
    valid_size = 0.01




    # obtain training indices that will be used for validation
    num_test = len(test_dataset)
    indices = list(range(num_test))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_test))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches

    valid_idx=[1540, 7210, 205, 3815, 8195, 3354, 6361, 4224, 4313, 4597, 880, 3688, 5263, 1380, 9989, 9328, 6488, 5952, 6220, 8383, 8556, 4504, 9248, 7405, 9069, 1916, 6704, 335, 7517, 8637, 4419, 8552, 8821, 6721, 4179, 4315, 3982, 6951, 5544, 8594, 7280, 360, 3745, 94, 1636, 5236, 5143, 9247, 5116, 1821, 8910, 8284, 9163, 3838, 1699, 6958, 6230, 428, 5156, 2609, 4420, 2323, 4042, 9290, 9771, 5918, 9577, 4422, 1671, 7397, 6317, 2974, 4542, 2628, 9943, 7901, 6952, 2564, 9806, 6170, 7590, 5559, 9384, 7644, 7089, 733, 6174, 3488, 8319, 349, 8402, 4562, 4230, 6930, 6318, 4438, 4762, 6307, 2572, 3493]

    valid_sampler = SubsetRandomSampler(valid_idx)
    valid_sampler = torch.utils.data.sampler.SequentialSampler(valid_idx)





    # load validation data in batches
    # valid_loader = torch.utils.data.DataLoader(test_dataset,
    #                                            batch_size=batch_size,
    #                                            sampler=valid_sampler,
    #                                            num_workers=0)
    valid_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=batch_size,
                                               sampler=valid_sampler,
                                               num_workers=0)


    return valid_loader

valid_loader = create_datasets(batch_size_test)


def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=80):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer

accuracy_list = []
accuracy_list_single_trial = [0] * simulation_time

confidence_list = []
confidence_list_single_trial = [0]* simulation_time

accuracy_list_reliability_diagram = []
confidence_list_reliability_diagram = []


class NeuralNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, quan_first=False, quan_last=False, constr_activation=None):
        super(NeuralNet, self).__init__()

        self.quan_first = quan_first
        self.quan_last = quan_last
        self.quan_activation = constr_activation is not None
        self.constr_activation = constr_activation

        if self.quan_first:
            self.first_act = LsqActivation(constr_activation) if self.quan_activation else Identity()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        if self.quan_activation:
            self.activation1 = LsqActivation(constr_activation)
        else:
            self.activation1 = nn.ReLU(inplace=True)

        in_channels = 64
        self.layer1, in_channels = make_layer(block, in_channels, planes=64, nblocks=layers[0],
                                               stride=1, constr_activation=constr_activation)
        self.layer2, in_channels = make_layer(block, in_channels, planes=128, nblocks=layers[1],
                                               stride=2, constr_activation=constr_activation)
        self.layer3, in_channels = make_layer(block, in_channels, planes=256, nblocks=layers[2],
                                               stride=2, constr_activation=constr_activation)
        self.layer4, in_channels = make_layer(block, in_channels, planes=512, nblocks=layers[3],
                                               stride=2, constr_activation=constr_activation)

        if self.quan_last:
            self.last_act = LsqActivation(constr_activation) if self.quan_activation else Identity()
            self.fc = Linear(512 * block.expansion, num_classes)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.activation2 = nn.ReLU(inplace=True)
    def forward(self, x, mode_index, targets, time_window = simulation_time):
        if mode_index == 0:
            step=0
            output = self.conv1(x)
            output = self.bn1(output)
            output = self.activation1(output)
            output = self.layer1(output)
            output = self.layer2(output)
            output = self.layer3(output)
            output = self.layer4(output)
            output = F.avg_pool2d(output, 4)
            output = output.view(output.size(0), -1)
            output = self.fc(output)
            output = self.activation2(output)
            return output
        if mode_index == 1:
            for step in range(time_window):

                output = self.conv1(x)
                output = self.bn1(output)
                output = self.activation1(output)
                BasicBlock.forward = forward2
                output = self.layer1(output)
                output = self.layer2(output)
                output = self.layer3(output)
                output = self.layer4(output)
                output = F.avg_pool2d(output, 4)
                output = output.view(output.size(0), -1)
                output = self.fc(output)
                if step==0:
                    output_of_the_last_layer=output/30
                else:

                    output_of_the_last_layer+=output.float()/30

                softmaxes = F.softmax(output_of_the_last_layer, dim=1)
                confidences, predicted = torch.max(softmaxes, 1)
                correct = float(predicted.eq(targets).sum().item())
                accuracy_list_accumulated[step] += correct
                accuracy_list_single_trial[step]=correct
                confidence_list_single_trial[step]=float(confidences.item())
            accuracy_list.append(copy.deepcopy(accuracy_list_single_trial))
            confidence_list.append(copy.deepcopy(confidence_list_single_trial))

            return output_of_the_last_layer
        if mode_index == 2:
            for step in range(time_window):

                output = self.conv1(x)
                output = self.bn1(output)
                output = self.activation1(output)
                BasicBlock.forward = forward2
                output = self.layer1(output)
                output = self.layer2(output)
                output = self.layer3(output)
                output = self.layer4(output)
                output = F.avg_pool2d(output, 4)
                output = output.view(output.size(0), -1)
                output = self.fc(output)
                if step==0:
                    output_of_the_last_layer=output/30
                else:

                    output_of_the_last_layer+=output.float()/30

                softmaxes = F.softmax(output_of_the_last_layer, dim=1)
                confidences, predicted = torch.max(softmaxes, 1)
                correct = float(predicted.eq(targets).sum().item())
                accuracy_list_accumulated[step] += correct
                accuracy_list_single_trial[step]=correct
                confidence_list_single_trial[step]=float(confidences.item())
                if confidences > 0.6:
                    break
            accuracy_list.append(copy.deepcopy(accuracy_list_single_trial))
            confidence_list.append(copy.deepcopy(confidence_list_single_trial))

            return output_of_the_last_layer



def resnet18(quan_first=False, quan_last=False, constr_activation=None, preactivation=False):
    block =  BasicBlock
    model = NeuralNet(block, [2, 2, 2, 2], quan_first=quan_first, quan_last=quan_last, constr_activation=constr_activation)
    return model

model = resnet18().to(device)



model.activation1=LsqActivation(activation_bits,torch.ones(1)*0.6)
model.activation2=LsqActivation(activation_bits,torch.ones(1)*0.5)
for name, module in model.named_children():
    if "layer" in name:
        for sub_name, sub_module in module.named_children():
            sub_module.activation1=LsqActivation(activation_bits,torch.ones(1)*0.3)
            sub_module.activation2 = LsqActivation(activation_bits,torch.ones(1)*0.4)


weight_bias = list()
scale_weight = list()
scale_activation = list()
for name, param in model.named_parameters():
    if 'activation' in name and 'scale' in name:
        scale_activation.append(param)
    else:
        weight_bias.append(param)
param_groups1 = [{'params': weight_bias, 'lr': learning_rate}]
param_groups2 = [
                 {'params': scale_activation, 'lr': learning_rate}]
param_groups = param_groups1 + param_groups2


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(param_groups, lr=learning_rate,
                      momentum=0.9, weight_decay=5e-4)

model.load_state_dict(torch.load("model_quant_highest_2bit.pth"))


model.to(device)
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images,0, labels)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print("valid",log)
    print_log('Accuracy of the fake quant ANN on the 10000 test images: {} %'.format(100 * correct / total), log)

print_log("quant model is above...............",log)



cal_th=[]
for name, module in model.named_parameters():
    if "activation" in name:
        print_log(name,log)
        cal_th.append(float(module))
th=cal_th
print_log(th,log)
th = [i * (2**activation_bits-1) for i in cal_th]
print_log(th,log)
print_log(len(th),log)
print_log("th is above...............",log)



print_log("modelsize is below...............",log)
size_list=[0]*num_of_layer
count=0
size_list[count]=list(model.activation1.output.size())
# ann_output[count]=model.activation1.output/cal_th[count]
for name, module in model.named_children():
    if "layer" in name:
        for sub_name, sub_module in module.named_children():
            for sub_sub_name, sub_sub_module in sub_module.named_children():
                if "activation1" in sub_sub_name:
                    count += 1
                    size_list[count] = list(sub_module.activation1.output.size())
                if "activation2" in sub_sub_name:
                    count += 1
                    size_list[count] = list(sub_module.activation2.output.size())
                if "activation3" in sub_sub_name:
                    count += 1
                    size_list[count] = list(sub_module.activation3.output.size())
count+=1
size_list[count]=list(model.activation2.output.size())




allspikecount=0
print_log("start time (without Dynamic Confidence)", log)
print_log(time.asctime(time.localtime(time.time())), log)
accuracy_list_accumulated = [0] * simulation_time
model.to(device)
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
    # for batch_idx, (inputs, targets) in enumerate(valid_loader):
        spike_output = []
        for i in range(num_of_layer):
            if i<(num_of_layer-1):
                locals()["c" + str(i + 1) + "_spike"]=locals()["c" + str(i + 1) + "_sumspike"]=torch.zeros(size_list[i][0], size_list[i][1], size_list[i][2], size_list[i][3], device=device)
                locals()["c" + str(i + 1) + "_mem"] = torch.zeros(size_list[i][0], size_list[i][1], size_list[i][2],
                                                                   size_list[i][3], device=device)+th[i]/2
            if i==(num_of_layer-1):
                locals()["c" + str(i + 1) + "_spike"] = locals()["c" + str(i + 1) + "_sumspike"] = torch.zeros(
                    size_list[i][0], size_list[i][1], device=device)
                locals()["c" + str(i + 1) + "_mem"] = torch.zeros(size_list[i][0], size_list[i][1], device=device) + th[i] / 2


        count = 0
        model.activation1 = SNNActivation(c1_mem, c1_spike, c1_sumspike, th[count], activation_bits)
        for name, module in model.named_children():
            if "layer" in name:
                for sub_name, sub_module in module.named_children():

                    for sub_sub_name, sub_sub_module in sub_module.named_children():

                        if "activation1" in sub_sub_name:
                            count += 1
                            sub_module.activation1 = SNNActivation(eval("c" + str(count + 1) + "_mem"),
                                                                   eval("c" + str(count + 1) + "_spike"),
                                                                   eval("c" + str(count + 1) + "_sumspike"), th[count],
                                                                   activation_bits)
                        if "activation2" in sub_sub_name:
                            count += 1
                            sub_module.activation2 = SNNActivation(eval("c" + str(count + 1) + "_mem"),
                                                                   eval("c" + str(count + 1) + "_spike"),
                                                                   eval("c" + str(count + 1) + "_sumspike"), th[count],
                                                                   activation_bits)
                        if "activation3" in sub_sub_name:
                            count += 1
                            sub_module.activation3 = SNNActivation(eval("c" + str(count + 1) + "_mem"),
                                                                   eval("c" + str(count + 1) + "_spike"),
                                                                   eval("c" + str(count + 1) + "_sumspike"), th[count],
                                                                   activation_bits)


        count += 1
        model.activation2 = SNNActivation(eval("c" + str(num_of_layer) + "_mem"),
                                                                   eval("c" + str(num_of_layer) + "_spike"),
                                                                   eval("c" + str(num_of_layer) + "_sumspike"), th[count],
                                                                   activation_bits)



        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        spike_output = model(inputs, 1, targets)

        _, predicted = spike_output.cuda().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
        number_of_layer=17
        sum_spike=[0]*number_of_layer
        number_of_neurons=[0]*number_of_layer
        count=0
        sum_spike[count]=model.activation1.sum_spike/th[count]
        number_of_neurons[count]=model.activation1.sum_spike.size(0)*model.activation1.sum_spike.size(1)*model.activation1.sum_spike.size(2)*model.activation1.sum_spike.size(3)
        for name, module in model.named_children():
            if "layer" in name:
                for sub_name, sub_module in module.named_children():

                    for sub_sub_name, sub_sub_module in sub_module.named_children():

                        if "activation1" in sub_sub_name:
                            count += 1
                            sum_spike[count] = sub_module.activation1.sum_spike/th[count]
                            number_of_neurons[count]=sub_module.activation1.sum_spike.size(0)*sub_module.activation1.sum_spike.size(1)*sub_module.activation1.sum_spike.size(2)*sub_module.activation1.sum_spike.size(3)
                        if "activation2" in sub_sub_name:
                            count += 1
                            sum_spike[count] = sub_module.activation2.sum_spike/th[count]

                            number_of_neurons[count] = sub_module.activation2.sum_spike.size(
                                0) * sub_module.activation2.sum_spike.size(1) * sub_module.activation2.sum_spike.size(
                                2) * sub_module.activation2.sum_spike.size(3)
                        if "activation3" in sub_sub_name:
                            count += 1
                            sum_spike[count] = sub_module.activation3.sum_spike / th[count]
                            number_of_neurons[count] = sub_module.activation3.sum_spike.size(
                                0) * sub_module.activation3.sum_spike.size(1) * sub_module.activation3.sum_spike.size(
                                2) * sub_module.activation3.sum_spike.size(3)
        count+=1
        plt_x=[0]*number_of_layer
        for i in range(number_of_layer):
            plt_x[i]=i
        plt_y = [0]*number_of_layer

        all_spikes=0
        all_neurons=0
        for i in range(number_of_layer):
            a=sum_spike[i].flatten().sum().to("cpu")
            plt_y[i]=a
            all_spikes=all_spikes+a
            all_neurons=all_neurons+ number_of_neurons[i]
        allspikecount=allspikecount+all_spikes/all_neurons
print_log("end time (without Dynamic Confidence)", log)
print_log(time.asctime(time.localtime(time.time())), log)
print_log("The averaged spike/neuron per inference", log)
print_log(allspikecount/10000, log)
print_log('Accuracy of the SNN on the 10000  images: %.3f' % (100 * correct / total), log)
# print_log(accuracy_list_accumulated, log)

for j in range(len(accuracy_list_accumulated)):
    print_log("Latency "+ str(j+1) + "  Accuracy "+str(accuracy_list_accumulated[j]/100),log)

print_log("SNN simulation without Dynamic Confidence end", log)



accuracy_list = []
accuracy_list_single_trial = [0] * simulation_time
confidence_list = []
confidence_list_single_trial = [0]* simulation_time
allspikecount=0
print_log("start time (with Dynamic Confidence)", log)
print_log(time.asctime(time.localtime(time.time())), log)
accuracy_list_accumulated = [0] * simulation_time
model.to(device)
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
    # for batch_idx, (inputs, targets) in enumerate(valid_loader):
        spike_output = []
        for i in range(num_of_layer):
            if i<(num_of_layer-1):
                locals()["c" + str(i + 1) + "_spike"]=locals()["c" + str(i + 1) + "_sumspike"]=torch.zeros(size_list[i][0], size_list[i][1], size_list[i][2], size_list[i][3], device=device)
                locals()["c" + str(i + 1) + "_mem"] = torch.zeros(size_list[i][0], size_list[i][1], size_list[i][2],
                                                                   size_list[i][3], device=device)+th[i]/2
            if i==(num_of_layer-1):
                locals()["c" + str(i + 1) + "_spike"] = locals()["c" + str(i + 1) + "_sumspike"] = torch.zeros(
                    size_list[i][0], size_list[i][1], device=device)
                locals()["c" + str(i + 1) + "_mem"] = torch.zeros(size_list[i][0], size_list[i][1], device=device) + th[i] / 2


        count = 0
        model.activation1 = SNNActivation(c1_mem, c1_spike, c1_sumspike, th[count], activation_bits)
        for name, module in model.named_children():
            if "layer" in name:
                for sub_name, sub_module in module.named_children():

                    for sub_sub_name, sub_sub_module in sub_module.named_children():

                        if "activation1" in sub_sub_name:
                            count += 1
                            sub_module.activation1 = SNNActivation(eval("c" + str(count + 1) + "_mem"),
                                                                   eval("c" + str(count + 1) + "_spike"),
                                                                   eval("c" + str(count + 1) + "_sumspike"), th[count],
                                                                   activation_bits)
                        if "activation2" in sub_sub_name:
                            count += 1
                            sub_module.activation2 = SNNActivation(eval("c" + str(count + 1) + "_mem"),
                                                                   eval("c" + str(count + 1) + "_spike"),
                                                                   eval("c" + str(count + 1) + "_sumspike"), th[count],
                                                                   activation_bits)
                        if "activation3" in sub_sub_name:
                            count += 1
                            sub_module.activation3 = SNNActivation(eval("c" + str(count + 1) + "_mem"),
                                                                   eval("c" + str(count + 1) + "_spike"),
                                                                   eval("c" + str(count + 1) + "_sumspike"), th[count],
                                                                   activation_bits)


        count += 1
        model.activation2 = SNNActivation(eval("c" + str(num_of_layer) + "_mem"),
                                                                   eval("c" + str(num_of_layer) + "_spike"),
                                                                   eval("c" + str(num_of_layer) + "_sumspike"), th[count],
                                                                   activation_bits)



        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        spike_output = model(inputs, 2, targets)

        _, predicted = spike_output.cuda().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
        number_of_layer=17
        sum_spike=[0]*number_of_layer
        number_of_neurons=[0]*number_of_layer
        count=0
        sum_spike[count]=model.activation1.sum_spike/th[count]
        number_of_neurons[count]=model.activation1.sum_spike.size(0)*model.activation1.sum_spike.size(1)*model.activation1.sum_spike.size(2)*model.activation1.sum_spike.size(3)
        for name, module in model.named_children():
            if "layer" in name:
                for sub_name, sub_module in module.named_children():

                    for sub_sub_name, sub_sub_module in sub_module.named_children():

                        if "activation1" in sub_sub_name:
                            count += 1
                            sum_spike[count] = sub_module.activation1.sum_spike/th[count]
                            number_of_neurons[count]=sub_module.activation1.sum_spike.size(0)*sub_module.activation1.sum_spike.size(1)*sub_module.activation1.sum_spike.size(2)*sub_module.activation1.sum_spike.size(3)
                        if "activation2" in sub_sub_name:
                            count += 1
                            sum_spike[count] = sub_module.activation2.sum_spike/th[count]

                            number_of_neurons[count] = sub_module.activation2.sum_spike.size(
                                0) * sub_module.activation2.sum_spike.size(1) * sub_module.activation2.sum_spike.size(
                                2) * sub_module.activation2.sum_spike.size(3)
                        if "activation3" in sub_sub_name:
                            count += 1
                            sum_spike[count] = sub_module.activation3.sum_spike / th[count]
                            number_of_neurons[count] = sub_module.activation3.sum_spike.size(
                                0) * sub_module.activation3.sum_spike.size(1) * sub_module.activation3.sum_spike.size(
                                2) * sub_module.activation3.sum_spike.size(3)
        count+=1
        plt_x=[0]*number_of_layer
        for i in range(number_of_layer):
            plt_x[i]=i
        plt_y = [0]*number_of_layer

        all_spikes=0
        all_neurons=0
        for i in range(number_of_layer):
            a=sum_spike[i].flatten().sum().to("cpu")
            plt_y[i]=a
            all_spikes=all_spikes+a
            all_neurons=all_neurons+ number_of_neurons[i]
        allspikecount=allspikecount+all_spikes/all_neurons
print_log("end time (with Dynamic Confidence)", log)
print_log(time.asctime(time.localtime(time.time())), log)
print_log("The averaged spike/neuron per inference", log)
print_log(allspikecount/10000, log)



for h in range(1):
    cretria_for_dynamic_quant =0.6
    for j in range(len(confidence_list[0])):
        dynamic_acc = 0
        dynamic_time = 0
        for i in range(len(confidence_list)):
            for k in range(j+1):
                if k==j:
                    dynamic_time += j + 1
                    dynamic_acc += accuracy_list[i][j]
                    break
                else:
                    if confidence_list[i][k] > cretria_for_dynamic_quant:
                        dynamic_time+=k+1
                        dynamic_acc+=accuracy_list[i][k]
                        break
        print_log("Latency "+ str(dynamic_time/10000) + "  Accuracy "+str(dynamic_acc/100),log)


print_log('Accuracy of the SNN on the 10000  images: %.3f' % (dynamic_acc/100), log)
print_log("SNN simulation with Dynamic Confidence end", log)



log.close()