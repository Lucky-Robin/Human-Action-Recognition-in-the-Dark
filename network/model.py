from torch import nn
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights, resnet34

# model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
model = resnet34(weights='DEFAULT')

# model.fc = nn.Linear(2048, 10)    # resnext50_32x4d
model.fc = nn.Linear(512, 10)   # resnet34

# pthfile = r'network/checkpoint/ResNeXt50/pretrained/lr1e-5_32/network_parameter_15.pth'   # test_acc=32.5%
# pthfile = r'network/checkpoint/ResNet34/pretrained/lr1e-4/network_parameter_12.pth'   # test_acc=36.875%
# model.load_state_dict(torch.load(pthfile))

# print(model)

