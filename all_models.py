import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np




def get_constraint(bits, obj):
    if bits == 0:
        return None
    if 'activation' in obj:
        lower = 0
        upper = 2 ** bits
    elif 'swish' in obj:
        lower = -1
        upper = 2 ** bits - 1
    else:
        lower = -2 ** (bits - 1) + 1
        upper = 2 ** (bits - 1)
    constraint = np.arange(lower, upper)
    return constraint



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, planes, stride=1, downsample=False, constr_activation=None):
        super(BasicBlock, self).__init__()
        self.quan_activation = constr_activation is not None

        self.conv1 = Conv2d(in_channels, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=planes)
        self.activation1 = LsqActivation(constr_activation) if self.quan_activation else nn.ReLU(inplace=True)

        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=planes)
        self.activation2 = LsqActivation(constr_activation) if self.quan_activation else nn.ReLU(inplace=True)
        self.downsample = None
        if downsample:
            conv = Conv2d(in_channels, planes, kernel_size=1, stride=stride, padding=0, bias=False)
            bn = nn.BatchNorm2d(num_features=planes)
            self.downsample = nn.Sequential(*[conv, bn])

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.activation1(out)


        out = self.conv2(out)
        out = self.bn2(out)
        out += residual

        out = self.activation2(out)

        return out


def forward(self, x):
    residual = x if self.downsample is None else self.downsample(x)







    out = self.conv1(x)
    out = self.bn1(out)
    out=out+(self.bn1.running_mean*self.bn1.weight/((self.bn1.running_var+0.00001)**0.5)-self.bn1.bias).unsqueeze(
        0).unsqueeze(2).unsqueeze(3).expand_as(out)


    out = self.activation1(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out += residual
    if self.downsample:
        for name, module in self.downsample.named_children():
            if "1" in name:
                out=out+(module.running_mean * module.weight / ((module.running_var+0.00001)**0.5) - module.bias).unsqueeze(
        0).unsqueeze(2).unsqueeze(3).expand_as(out)

    out = out + (self.bn2.running_mean * self.bn2.weight / ((self.bn2.running_var+0.00001)**0.5) - self.bn2.bias).unsqueeze(
        0).unsqueeze(2).unsqueeze(3).expand_as(out)

    out = self.activation2(out)

    return out

def forward2(self, x):
    residual = x if self.downsample is None else self.downsample(x)

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.activation1(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out += residual
    out = self.activation2(out)

    return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, planes, stride=1, downsample=None, constr_activation=None):
        super(Bottleneck, self).__init__()
        self.quan_activation = constr_activation is not None

        self.conv1 = Conv2d(in_channels, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.activation1 = LsqActivation(constr_activation) if self.quan_activation else nn.ReLU(inplace=True)

        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.activation2 = LsqActivation(constr_activation) if self.quan_activation else nn.ReLU(inplace=True)

        self.conv3 = Conv2d(planes, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.activation3 = LsqActivation(constr_activation) if self.quan_activation else nn.ReLU(inplace=True)

        self.downsample = None
        if downsample:
            conv = Conv2d(in_channels, planes * 4, kernel_size=1, stride=stride, padding=0, bias=False)
            bn = nn.BatchNorm2d(num_features=planes * 4)
            self.downsample = nn.Sequential(*[conv, bn])

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.activation3(out)
        return out





def make_layer(block, in_channels, planes, nblocks, stride=1, constr_activation=None):
    layers = list()
    downsample = stride != 1 or in_channels != planes * block.expansion
    layers.append(block(in_channels, planes, stride, downsample, constr_activation))
    in_channels = planes * block.expansion
    for i in range(1, nblocks):
        layers.append(block(in_channels, planes, constr_activation=constr_activation))
    return nn.Sequential(*layers), planes * block.expansion


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, quan_first=False, quan_last=False, constr_activation=None):
        super(ResNet, self).__init__()
        self.quan_first = quan_first
        self.quan_last = quan_last
        self.quan_activation = constr_activation is not None
        self.constr_activation = constr_activation

        if self.quan_first:
            self.first_act = LsqActivation(constr_activation) if self.quan_activation else _Identity()
            self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        if self.quan_activation:
            self.activation1 = LsqActivation(constr_activation)
        else:
            self.activation1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        in_channels = 64
        self.layer1, in_channels = _make_layer(block, in_channels, planes=64, nblocks=layers[0],
                                               stride=1, constr_activation=constr_activation)
        self.layer2, in_channels = _make_layer(block, in_channels, planes=128, nblocks=layers[1],
                                               stride=2, constr_activation=constr_activation)
        self.layer3, in_channels = _make_layer(block, in_channels, planes=256, nblocks=layers[2],
                                               stride=2, constr_activation=constr_activation)
        self.layer4, in_channels = _make_layer(block, in_channels, planes=512, nblocks=layers[3],
                                               stride=2, constr_activation=constr_activation)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        if self.quan_last:
            self.last_act = LsqActivation(constr_activation) if self.quan_activation else _Identity()
            self.fc = Linear(512 * block.expansion, num_classes)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if self.quan_first:
            x = self.first_act(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.quan_last:
            x = self.last_act(x)
        x = self.fc(x)
        return x



class LsqActivationFun(autograd.Function):


    @staticmethod
    def forward(ctx,x,scale,constraint):
        ctx.constraint = constraint

        x = x
        scale = scale
        x_scale = torch.div(x, scale)
        x_clip = F.hardtanh(x_scale, 0, max_val=float((2**ctx.constraint-1)))

        x_round = torch.round(x_clip)


        x_restore = torch.mul(x_round, scale)
        ctx.save_for_backward(x_clip)

        return x_restore

    @staticmethod
    def backward(ctx, grad_output):
        grad_top = grad_output
        x_clip = ctx.saved_tensors[0]
        internal_flag = ((x_clip > 0) ^ (x_clip >= float((2**ctx.constraint-1))))

        # gradient for activation
        grad_activation = grad_top * internal_flag

        # gradient for scale
        grad_one = x_clip * internal_flag
        grad_two = torch.round(x_clip)
        grad_scale_elem = grad_two - grad_one
        grad_scale = (grad_scale_elem * grad_top).sum().view((1,))
        grad_scale=grad_scale/((len(x_clip)*float((2**ctx.constraint-1)))**0.5)
        return grad_activation, grad_scale, None








class Identity(nn.Module):
    def forward(self, x):
        return x





class LsqActivation(nn.Module):

    def __init__(self, constraint, scale_init=None, skip_bit=None):
        super(LsqActivation, self).__init__()
        self.constraint = constraint

        # scale_init = scale_init if scale_init is not None else torch.ones(1) * 100 / float(2 ** self.constraint - 1)
        scale_init = scale_init if scale_init is not None else torch.ones(1)
        self.scale = nn.Parameter(scale_init)
        self.skip_bit = skip_bit
        self.output=None

    def extra_repr(self):
        return 'constraint=%s' % self.constraint


    def forward(self, x):
        a=LsqActivationFun.apply(x,self.scale,self.constraint)
        self.output=a
        return a




class SNNActivationFun(autograd.Function):

    def __init__(self, mem, th):
        super(SNNActivationFun, self).__init__()
        # self.valmin = float(0)
        self.mem = mem
        self.th=th
    def forward(self, *args, **kwargs):
        mem = args[0]
        # return mem.gt(self.th).float()
        return (mem >= self.th).float()

    def backward(self, *grad_outputs):
        grad_activation=1
        grad_scale=1
        return None, None


class SNNActivation(nn.Module):
    def __init__(self, mem, spike, sum_spike, th, constraint):
        super(SNNActivation, self).__init__()
        self.mem=mem
        self.spike=spike
        self.sum_spike=sum_spike
        self.th=th
        self.constraint = constraint
    def forward(self, x):
        # self.mem = self.mem - self.spike + x +self.th/18
        self.mem = self.mem - self.spike + x

        spike = SNNActivationFun(self.mem, self.th)
        self.spike=spike.forward(self.mem, self.th)*self.th
        # self.spike = self.spike.mul((self.sum_spike<((2**self.constraint-1.5)*self.th)).float())
        self.spike = self.spike.float()
        # neg_spike=((-self.mem)>0).float().mul((self.sum_spike > (0.5*self.th)).float()) *self.th
        # self.spike = self.spike - neg_spike

        self.sum_spike=self.sum_spike+self.spike
        return self.spike
    # def backward(self, *grad_outputs):



class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size,stride,
                                     padding, dilation, groups, bias)
        self.wquantizer = None

    def forward(self, x):
        weight = self.weight if self.wquantizer is None else self.wquantizer(self.weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias)
        self.wquantizer = None

    def forward(self, x):
        weight = self.weight if self.wquantizer is None else self.wquantizer(self.weight)
        return F.linear(x, weight, self.bias)

