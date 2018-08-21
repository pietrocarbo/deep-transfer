
import torch
import torch.nn as nn
import torch.legacy.nn as lnn

from functools import reduce
from torch.autograd import Variable

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))


vgg_normalised_conv3_1 = nn.Sequential( # Sequential,
	nn.Conv2d(3,3,(1, 1)),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(3,64,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64,64,(3, 3)),
	nn.ReLU(),
	nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64,128,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(128,128,(3, 3)),
	nn.ReLU(),
	nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(128,256,(3, 3)),
	nn.ReLU(),
)