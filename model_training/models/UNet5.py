import torch
import torch.nn as nn

from collections import OrderedDict

def generation_init_weights(module):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                    or classname.find('Linear') != -1):
            
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    module.apply(init_func)

def load_state_dict(module, state_dict, strict=False, logger=None):
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None

    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)
    return missing_keys

class conv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
        super(conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)

class upconv(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1),
                nn.InstanceNorm2d(dim_out, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)
    
class Encoder(nn.Module):
    def __init__(self, in_dim=3, out_dim=128):
        super(Encoder, self).__init__()
        self.in_dim = in_dim
        self.c1 = conv(in_dim, 8)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.c2 = conv(8, 16)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.c3 = conv(16, 32)
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.c4 = conv(32, 64)
        self.pool4 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.c5 = nn.Conv2d( 64, out_dim, 3, 1, 1)
        self.pool5=nn.Sequential(
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.BatchNorm2d(out_dim),
                nn.Tanh()
                )
        

    def init_weights(self):
        generation_init_weights(self)
        

    def forward(self, input):
        h1 = self.c1(input)#16,1
        # print('h1 shape:',h1.shape)
        h2 = self.pool1(h1)        
        h2 = self.c2(h1)#32,1/2
        # print('h2 shape:',h2.shape)
        h3 = self.pool2(h2)
        h3 = self.c3(h3)#64,1/4
        # print('h3 shape:',h3.shape)
        h4 = self.pool3(h3)#64,1/8
        h4 = self.c4(h4)
        # print('h4 shape:',h4.shape)
        h5 = self.pool4(h4)#64,1/8
        h5 = self.c5(h5)
        # print('h5 shape:',h4.shape)
        output= self.pool5(h5)
        # print('output shape:',output.shape)
        return (output,h5,h4,h3,h2)    

class Decoder(nn.Module):
    def __init__(self, out_dim=2, in_dim=128):
        super(Decoder, self).__init__()
        self.conv1 = conv(in_dim, 128)
        self.upc1 = upconv(128, 64)#skip
        self.conv2 = conv(64, 64)
        self.upc2 = upconv(64+128, 32)
        self.conv3 = conv(32, 32)
        self.upc3=upconv(32+64,16)
        self.conv4 = conv(16, 16)
        self.upc4=upconv(32+16,4)
        self.conv5 =  nn.Sequential(
                nn.Conv2d(4, out_dim, 3, 1, 1),
                nn.Sigmoid()
                )
        

    def init_weights(self):
        generation_init_weights(self)

    def forward(self, vals):
        d1 = self.conv1(vals[0])#64,1/8
        d2 = self.upc1(d1)
        
        d2 = self.conv2(d2)#32,1/4
        
        d3 = self.upc2(torch.cat((d2,vals[1]),dim=1))
        
        d3 = self.conv3(d3)#16,1/2
       

        d4 = self.upc3(torch.cat((d3, vals[2]), dim=1))
        d4 = self.conv4(d4) 
        
        d5 = self.upc4(torch.cat((d4, vals[3]), dim=1))
        output = self.conv5(d5)

        return output


class UNet5(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=2,
                 **kwargs):
        super().__init__()

        self.encoder = Encoder(in_dim=3)
        self.decoder = Decoder(out_dim=2)

    def forward(self, x):
        # print(x.shape[1])
        x = self.encoder(x)
        return self.decoder(x)

    def init_weights(self, pretrained=None, pretrained_transfer=None, strict=False, **kwargs):
        if isinstance(pretrained, str):
            new_dict = OrderedDict()
            weight = torch.load(pretrained, map_location='cpu')['state_dict']
            for k in weight.keys():
                new_dict[k] = weight[k]
            load_state_dict(self, new_dict, strict=strict, logger=None)
            print('Load state dict form {}'.format(pretrained))
        elif pretrained is None:
            generation_init_weights(self)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')
