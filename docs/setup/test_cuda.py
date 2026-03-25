import torch

print (f"torch.cuda.is_available(): {torch.cuda.is_available()}")

x = torch.rand(5, 3)
print(x)