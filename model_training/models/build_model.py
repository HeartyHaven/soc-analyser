import models
import torch

def build_model(args):
    # print(args.pop('model_type'))
    model = models.__dict__[args.pop('model_type')](**args)
    
    if args['test_mode']:
        checkpoint = torch.load(args['pretrained'])
        # print(checkpoint['state_dict'])
        # 删除DataParallel前缀
        new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        
        model.load_state_dict(new_state_dict)
        model.eval()
    else:
        checkpoint = torch.load(args['pretrained'])
        # print(checkpoint['state_dict'])
        # 删除DataParallel前缀
        new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        
        model.load_state_dict(new_state_dict)
        # model.init_weights(**args)
    return model
