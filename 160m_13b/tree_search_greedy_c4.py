import torch
torch.set_printoptions(profile="full")
import json
from tqdm import tqdm
import sys
from copy import deepcopy
# p = torch.tensor([0.0, 0.6242, 0.0992, 0.0466, 0.0294, 0.0179, 0.0153, 0.0110, 0.0073, 0.0065,
#         0.0066, 0.0042, 0.0046, 0.0042, 0.0040, 0.0031, 0.0031, 0.0024, 0.0021,
#         0.0026, 0.0020, 0.0016, 0.0017, 0.0013, 0.0013, 0.0018, 0.0010, 0.0013,
#         0.0015, 0.0011, 0.0011, 0.0014, 0.0014])
p = torch.tensor([0.0, 7.7430e-01, 7.9867e-02, 3.3726e-02, 2.0291e-02, 1.1396e-02, 7.9681e-03,
        4.5400e-03, 6.0224e-03, 3.4281e-03, 3.3355e-03, 2.5943e-03, 3.0575e-03,
        1.6677e-03, 2.0384e-03, 1.6677e-03, 1.1118e-03, 1.9457e-03, 1.4824e-03,
        9.2653e-04, 9.2653e-04, 1.0192e-03, 9.2653e-04, 7.4122e-04, 1.1118e-03,
        6.4857e-04, 3.7061e-04, 3.7061e-04, 6.4857e-04, 7.4122e-04, 5.5592e-04,
        5.5592e-04, 7.4122e-04])

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
draft_inference_time = 0.0016
target_verify_time = [
0.03912119388580322,
0.04097423076629639,
0.041138229370117185,
0.04153526782989502,
0.04215144872665405,
0.04696519374847412,
0.04878691196441651,
0.05441854238510132

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

path = "./growmaps/L40-C4-160m-13b-greedy.pt"

torch.save(grow_map, path)
