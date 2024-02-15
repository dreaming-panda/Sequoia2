import torch
torch.set_printoptions(profile="full")
import json
from tqdm import tqdm
import sys
from copy import deepcopy
p = torch.tensor([0.0, 6.5079e-01, 1.1936e-01, 5.3246e-02, 2.8199e-02, 2.1128e-02, 1.5761e-02,
        1.2183e-02, 9.2861e-03, 8.0082e-03, 4.7708e-03, 4.6856e-03, 5.4524e-03,
        2.9818e-03, 3.3225e-03, 2.4706e-03, 2.6410e-03, 1.7891e-03, 1.8743e-03,
        1.6187e-03, 1.9594e-03, 1.1927e-03, 2.3854e-03, 6.8155e-04, 1.7039e-03,
        1.1075e-03, 1.5335e-03, 1.1927e-03, 7.6674e-04, 1.1075e-03, 9.3713e-04,
        1.1075e-03, 2.5558e-04])
# p = torch.tensor([0.0, 0.5403, 0.1035, 0.0532, 0.0318, 0.0210, 0.0188, 0.0128, 0.0118, 0.0085,
#         0.0084, 0.0068, 0.0052, 0.0045, 0.0038, 0.0044, 0.0048, 0.0024, 0.0032,
#         0.0031, 0.0015, 0.0019, 0.0028, 0.0025, 0.0024, 0.0020, 0.0023, 0.0020,
#         0.0017, 0.0016, 0.0017, 0.0016, 0.0018])

max_branch = p.shape[0] - 1

max_depth = 10

max_budget = 128

T = torch.zeros((max_budget + 1, max_depth + 1, max_branch + 1)).fill_(-torch.inf)
T_max = torch.zeros((max_budget + 1, max_depth + 1))
branch_map = {}
for l in range(1, max_depth + 1):
    for b in range(0, max_branch + 1):
        if b == 0:
            T[1][l][b] = 1.0
            branch_map[(1,l,b)] = []


for m in tqdm(range(2, max_budget+1)):
    for l in range(2, max_depth + 1):
        T[m][l][1] = 1 + p[1] * T[m-1][l-1].max()
        if T[m][l][1] > 0:
            branch_map[(m,l,1)] = [(m-1, l-1, T[m-1][l-1].argmax(dim=0).item())]
        for b in range(2, max_branch + 1):
            max_value = -torch.inf
            #new_y = -1
            for y in range(1, m):
                new_value = T[y][l][b-1] + p[b] * T[m-y][l-1].max()
                if new_value > max_value:
                    max_value = new_value
                    new_y = y
                max_value = max(max_value, new_value)
            T[m][l][b] = max_value
            if max_value >= 0:
                new_branch = T[m-new_y][l-1].argmax(dim=0).item()
                new_list :list = deepcopy(branch_map[(new_y, l, b-1)])
                new_list.append((m-new_y, l-1, new_branch))
                branch_map[(m,l,b)] = new_list

 
    

results = T.max(dim=2).values
print(results)
draft_inference_time = 0.006
target_verify_time = [
0.052310295343399045,
0.05358581018447876,
0.054559799671173095,
0.05556274938583374,
0.057551119565963746,
0.05863354706764221,
0.061939607143402096,
0.08187625646591186
                    ]


valid_budget = [1,2,4,8,16,32,64,128]

dec_time = torch.inf
pairs = None
for i, b in enumerate(valid_budget):
    target_time = target_verify_time[i]
    for d, ac_len in enumerate(results[b]):
        if ac_len < 0:
            continue
        x = ((d) * draft_inference_time + target_time) / ac_len
        if x < dec_time:
            dec_time = x
            pairs = (b,d)

print(dec_time, target_verify_time[0] / dec_time, pairs)

(m, l) = pairs
b = T[m][l].argmax(dim=0).item()

positions = [0]
states = [(m,l,b)]
active = [True]
depth = [0]
Successors = [[]]
attention_mask = torch.zeros(m,m).long()
parents = [-1]
expand_lists = []
expand_branches = []
num_nodes = 1
while True:

    expand = []
    expand_branch = []
    for i, act in enumerate(active):
        if act: 
            if parents[i] != -1:
                attention_mask[i] = attention_mask[parents[i]]
            attention_mask[i][i] = 1
            expand.append(i)
            active[i] = False
            (x,y,z) = states[i]
            expand_branch.append(z)
            positions.extend(list(range(num_nodes, num_nodes + z)))
            Successors[i].extend(list(range(num_nodes, num_nodes + z)))
            Successors.extend([[] for _ in range(z)])
            parents.extend([i for _ in range(z)])
            depth.extend([depth[i] + 1 for _ in range(z)])
            states.extend(branch_map[(x,y,z)])
            assert len(branch_map[(x,y,z)]) == z
            num_nodes = num_nodes + z
    if len(expand) == 0:
        break
    expand_lists.append(expand)
    expand_branches.append(expand_branch)
    active.extend([True for _ in range(sum(expand_branch))])


assert num_nodes == m
assert len(positions) == m
assert len(depth) == m
grow_map = {
    "roots": expand_lists,
    "branches": expand_branches,
    "Successors":Successors,
    "mask": attention_mask,
    "depth": torch.LongTensor(depth),
    "size": num_nodes
}

path = "./growmaps/A100-CNN-1.3b-33b-stochastic.pt"

torch.save(grow_map, path)

