import torch
import numpy as np

y = torch.arange((9)).view(3,3)
oppo_direction = np.array([[-1, 1], [1, 0], [1, 1], [-1, 0]])

print(y)
# z = torch.zeros((3,3))
# for i in range(4):
#     z = y.roll(oppo_direction[i][0].item(), oppo_direction[i][1].item())
#     print(z)

# z = y.roll(shifts=(-1,1),dims=(0,1))  # left&down
# z = y.roll(shifts=(0,1),dims=(0,1))     # left
# z = y.roll(shifts=(0,-1),dims=(0,1))    # right
# z = y.roll(shifts=(1,0),dims=(0,1))    # up
z = y.roll(shifts=(oppo_direction[0][0],oppo_direction[0][1]),dims=(0,1))    # down
print(z)

print(0.8+3.65+1.5+0.49+0.37+0.85+8+2.95+1.25+1.6+1.6)