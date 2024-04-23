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

class out_conv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False):
        super(out_conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)   

class Encoder(nn.Module):
    def __init__(self, in_dim=3, out_dim=64):
        super(Encoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim=out_dim
        self.scale_img_2 = nn.AvgPool2d(kernel_size=2)
        self.scale_img_3 = nn.AvgPool2d(kernel_size=2)
        self.scale_img_4 = nn.AvgPool2d(kernel_size=2)

        self.c1_1 = conv(self.in_dim, 8)  
        self.c1_2 = conv(8,8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.c2_1 = conv(self.in_dim, 16)
        self.c2_2 = conv(16+8, 16)       
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.c3_1 = conv(self.in_dim, 32)
        self.c3_2 = conv(32+16, 32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.c4_1 = conv(self.in_dim, out_dim)
        self.c4_2 = conv(out_dim+32, out_dim)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def init_weights(self):
        generation_init_weights(self)

    def forward(self, input):
        scale_img_2 = self.scale_img_2(input)#1/2
        scale_img_3 = self.scale_img_3(scale_img_2)#1/4
        scale_img_4 = self.scale_img_4(scale_img_3)#1/8

        c1=self.c1_1(input)#1,8
        # print('stage1.1:',c1.shape)
        c1=self.c1_2(c1)
        # print('stage1.2:',c1.shape)
        p1=self.pool1(c1)#1/2
        # print('stage1.3:',p1.shape)
        
        i2=self.c2_1(scale_img_2)#1/2,16
        # print('stage2.1:',i2.shape)
        i2= torch.cat((p1,i2),dim=1)
        # print('stage2.2:',i2.shape)
        c2=self.c2_2(i2)
        # print('stage2.3:',c2.shape)
        # c2=self.c2_3(c2)
        # print('stage2.4:',c2.shape)
        p2=self.pool2(c2)#1/4
        # print('stage2.5:',p2.shape)
        
        i3=self.c3_1(scale_img_3)#1/4,32
        # print('stage3.1:',i3.shape)
        i3= torch.cat((p2,i3),dim=1)
        # print('stage3.2:',i3.shape)
        c3=self.c3_2(i3)
        # print('stage3.3:',c3.shape)
        # c3=self.c3_3(c3)
        # print('stage3.4:',c3.shape)
        p3=self.pool3(c3)#1/8
        # print('stage3.5:',p3.shape)
        
        i4=self.c4_1(scale_img_4)#1/8,64
        # print('stage4.1:',i4.shape)
        i4= torch.cat((p3,i4),dim=1)
        # print('stage4.2:',i4.shape)
        c4=self.c4_2(i4)
        # print('stage4.3:',c4.shape)
        # c4=self.c4_3(c4)  
        # print('stage4.4:',c4.shape)
        p4=self.pool4(c4)#1/16,64
        
        return p1,p2,p3,p4 
 

class Decoder(nn.Module):
    def __init__(self, out_dim=2, in_dim=64):
        super(Decoder, self).__init__()

        self.upc1 = upconv(64, 32)#1/8--1/4
        self.c1_1= conv(32+32,32)
        self.c1_2=conv(32,32)

        self.upc2 = upconv(32, 16)#1/4--1/2
        self.c2_1= conv(16+16,16)
        self.c2_2=conv(16,16)
        
        self.upc3 = upconv(16, 8)#1/2--1
        self.c3_1= conv(8+8,8)
        self.c3_2=conv(8,8)       
        
        self.sc1=nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.sc2=nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.sc3=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.outlayer1=out_conv(32,2)
        self.outlayer2=out_conv(16,2)
        self.outlayer3=out_conv(8,2)

    def init_weights(self):
        generation_init_weights(self)

    def forward(self, vals):
        # print('##DECODER')
        u1= self.upc1(vals[3])
        # print('stage1.1:',u1.shape)
        
        u1= torch.cat((u1,vals[2]),dim=1)
        # print('stage1.2:',u1.shape)
        c1= self.c1_1(u1)
        # print('stage1.3:',c1.shape)
        c1= self.c1_2(c1)
        # print('stage1.4:',c1.shape)

        u2= self.upc2(vals[2])
        # print('stage2.1:',u2.shape)
        u2= torch.cat((u2,vals[1]),dim=1)
        # print('stage2.2:',u2.shape)
        c2= self.c2_1(u2)
        # print('stage2.3:',c2.shape)
        c2= self.c2_2(c2)
        # print('stage2.4:',c2.shape)
        
        u3= self.upc3(vals[1])
        # print('stage3.1:',u3.shape)
        u3= torch.cat((u3,vals[0]),dim=1)
        # print('stage3.2:',u3.shape)
        c3= self.c3_1(u3)
        # print('stage3.3:',c3.shape)
        c3= self.c3_2(c3)
        # print('stage3.4:',c3.shape)
        
        out3=self.sc3(c3)
        # print('stage4.1:',out3.shape)
        out3=self.outlayer3(out3)
        # print('stage4.1:',out3.shape)
        out2=self.sc2(c2)
        # print('stage4.2:',out2.shape)
        out2=self.outlayer2(out2)
        # print('stage4.2:',out2.shape)
        out1=self.sc1(c1)
        # print('stage4.3:',out1.shape)
        out1=self.outlayer1(out1)
        # print('stage4.3:',out1.shape)
        
        out=torch.mean(torch.stack([out1,out2,out3],dim=1),1,False)
        return out


class MNet(nn.Module):
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
