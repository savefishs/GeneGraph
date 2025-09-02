import torch
import torch.nn as nn
import torch.nn.functional as F

class TrainableGraphFuser(nn.Module):
    def __init__(self, edge_in_dim=4, gate_hidden=64, learn_eta=True):
        super().__init__()
        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(edge_in_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1)
        )
        # 可选：交互项参数
        if learn_eta:
            self.eta_layer = nn.Sequential(
                nn.Linear(edge_in_dim, gate_hidden),
                nn.ReLU(),
                nn.Linear(gate_hidden, 1)
            )
        else:
            self.eta_layer = None

    def forward(self, edges, w1, w2):
        """
        edges: [2, E] 边的索引
        w1, w2: [E] 来自 A1, A2 的边权
        """
        # 构造边特征
        feat = torch.stack([
            w1, w2, w1 - w2, w1 * w2
        ], dim=-1)   # [E,4]

        # 门控
        w = torch.sigmoid(self.gate(feat)).squeeze(-1)   # [E]

        # 融合
        fused = w * w1 + (1 - w) * w2

        # 交互项
        if self.eta_layer is not None:
            eta = self.eta_layer(feat).squeeze(-1)
            fused = fused + eta * (w1 * w2)

        return edges, fused
    

def coalesce_union_edges(
    edges1: torch.Tensor, w1: torch.Tensor,
    edges2: torch.Tensor, w2: torch.Tensor,
    num_nodes: int
):
    """
    两张图的边集合并，对齐边权
    edges*: LongTensor [2, E*]
    w*: FloatTensor [E*]

    返回:
      edges_u: [2, E_u]  # 并集后的边索引
      a1_u   : [E_u]     # 边在图1的权重（若不存在为0）
      a2_u   : [E_u]     # 边在图2的权重（若不存在为0）
    """
    # 把 (i,j) 转成唯一 key
    key1 = edges1[0] * num_nodes + edges1[1]
    key2 = edges2[0] * num_nodes + edges2[1]

    # 拼接
    all_keys = torch.cat([key1, key2])
    all_edges = torch.cat([edges1, edges2], dim=1)
    all_weights = torch.cat([w1, w2])
    tags = torch.cat([
        torch.zeros_like(w1, dtype=torch.long), 
        torch.ones_like(w2, dtype=torch.long)
    ])

    # 排序按 key
    order = torch.argsort(all_keys)
    all_keys = all_keys[order]
    all_edges = all_edges[:, order]
    all_weights = all_weights[order]
    tags = tags[order]

    # unique
    uniq_keys, first_idx = torch.unique_consecutive(all_keys, return_index=True)
    E_u = uniq_keys.size(0)
    a1_u = torch.zeros(E_u, device=w1.device)
    a2_u = torch.zeros(E_u, device=w1.device)

    # 遍历 unique key
    counts = torch.diff(torch.cat([first_idx, torch.tensor([all_keys.size(0)], device=w1.device)]))
    ptr = 0
    for u in range(E_u):
        c = counts[u].item()
        sl = slice(ptr, ptr+c)
        mask1 = (tags[sl] == 0)
        mask2 = (tags[sl] == 1)
        if mask1.any():
            a1_u[u] = all_weights[sl][mask1].max()  # 同一边可能重复 → 取max
        if mask2.any():
            a2_u[u] = all_weights[sl][mask2].max()
        ptr += c

    # 恢复边索引
    i_u = uniq_keys // num_nodes
    j_u = uniq_keys % num_nodes
    edges_u = torch.stack([i_u, j_u], dim=0)
    return edges_u, a1_u, a2_u