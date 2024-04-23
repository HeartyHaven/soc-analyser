import torch
x1=torch.tensor([[[1,2,3],
                 [4,5,6]]],dtype=torch.int)
x2=torch.tensor([[[1,2,3],
                 [4,5,6]],
                 [[1,2,3],
                 [4,5,6]]],dtype=torch.int)
print(x1.shape)
print(x2.shape)