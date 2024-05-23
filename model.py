import torch
import torch.nn as nn

# Key points regarding VGG 16 architecture from the original paper -
# -> pre-processing - subtraction of mean RGB value
# -> 3x3 kernel for all the conv layers
# -> 1x1 stride
# -> 1x1 padding for the 3x3 kernels (preserver the dimensions)
# -> 2x2 kernel and 2x2 stride for Max Pooling layers
# -> activation is ReLU
# -> 2 Double Convolution Layers (64 and 128 filters), 3 Triple Convolutional Layers (256, 512 and 512 filters)
# Instead of creating a Class for Double Conv and then for Triple Conv layers, it would be better for us to use a list
# with a certain number of filters at every stage and then use "M" to mention if it is a (max) pooling type of layer
VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']    # architecture
# This is followed by a fixed FCN type of layers and then softmax that out depending on the number of classes


class VGG(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_layers = self.create_conv_layers(architecture=VGG16)
        self.FCN = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            # 512*7*7 because we have 512 filters each of 7x7. (Due to 5 Pool layers, 224x224 reduces to 7x7)
            nn.ReLU(),
            nn.Dropout(p=0.5),   # optional
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),   # optional
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):

        x = self.conv_layers(x)
        print('after convolutional layers, shape of x is : ', x.shape)  # -> [1, 512, 7, 7]
        # flatten this out to 512*7*7
        x = x.reshape(x.shape[0], -1)   # -> [1, 25088]
        print('after reshaping x, shape of x is : ', x.shape)
        x = self.FCN(x)
        print('after FCN layers, shape of x is : ', x.shape)    # -> [1, 1000]

        return x

    def create_conv_layers(self, architecture):

        layers = []
        in_channels = self.in_channels
        for val in architecture:
            if type(val) == int:
                # conv layers
                out_channels = val
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                           nn.BatchNorm2d(val),
                           nn.ReLU()
                           ]
                in_channels = val
            elif val == 'M':
                layers += [self.pool]

        # print('layers list is :', layers)
        # print('Sequential module contains : ', nn.Sequential(*layers)) ->  total of 43 layers, but 16 weight layers
        return nn.Sequential(*layers)
        # Note -> This isn't the same as using a nn.ModuleList -> we'd have to separately define the forward pass here
        # But, now we are returning a Sequential type of module, hence the forward pass is already done


if __name__ == '__main__':
    model = VGG(3, 1000)
    x = torch.randn(1, 3, 224, 224)     # (B, C, H, W)
    print(model(x).shape)   # expected -> (1, 1000)
