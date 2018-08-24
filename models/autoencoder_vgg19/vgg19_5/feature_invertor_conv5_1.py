import torch.nn as nn
from functools import reduce


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


feature_invertor_conv5_1 = nn.Sequential( # Sequential,
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512,512,(3, 3)),
	nn.ReLU(),
	nn.UpsamplingNearest2d(scale_factor=2),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512,512,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512,512,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512,512,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(512,256,(3, 3)),
	nn.ReLU(),
    nn.UpsamplingNearest2d(scale_factor=2),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256,256,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256,256,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256,256,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(256,128,(3, 3)),
	nn.ReLU(),
    nn.UpsamplingNearest2d(scale_factor=2),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(128,128,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(128,64,(3, 3)),
	nn.ReLU(),
    nn.UpsamplingNearest2d(scale_factor=2),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64,64,(3, 3)),
	nn.ReLU(),
	nn.ReflectionPad2d((1, 1, 1, 1)),
	nn.Conv2d(64,3,(3, 3)),
)