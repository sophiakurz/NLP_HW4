import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, TransformerConv, JumpingKnowledge


# -------------------------------------------------- #
#                    Time Encoder                    #
# -------------------------------------------------- #
class TimeEncode(nn.Module):
    """
    Sinusoidal time encoding:
    Maps a scalar Δt to a 2 * dim vector [sin, cos].
    """
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.rand(dim))   # learnable frequency parameter

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        # delta_t: [E, 1]
        ft = delta_t * self.w                   # broadcast to [E, dim]
        return torch.cat([torch.sin(ft), torch.cos(ft)], dim=1)  # [E, 2*dim]


# -------------------------------------------------- #
#            Bi-Heterogeneous Graph Transformer      #
#                   with Edge Attributes             #
# -------------------------------------------------- #
class BiHGT_Edge(nn.Module):
    """
    HGT with rich edge attributes.
    Edge attribute column order:
        0: w_norm      Folded weight (log1p normalized)
        1: dt_norm     Normalized time difference
        2: infl_flag   Influencer→follower flag (0 or 1)
        3: cos_sim     Cosine similarity of node features
    """
    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()

        # 1) Input linear projection
        self.lin_in  = nn.Linear(in_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.norms   = nn.ModuleList([
            nn.LayerNorm(hid_dim) for _ in range(num_layers)
        ])

        # 2) Time encoding for Δt
        te_dim        = hid_dim // num_heads    # per-head dimension
        self.time_enc = TimeEncode(te_dim)

        # Edge dimension: 2*te_dim (sin+cos) + 3 scalar features
        edge_dim = 2 * te_dim + 3

        # 3) Stack of heterogeneous TransformerConvs
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('tweet', 'TD', 'tweet'): TransformerConv(
                    in_channels=hid_dim,
                    out_channels=te_dim,
                    heads=num_heads,
                    edge_dim=edge_dim
                ),
                ('tweet', 'BU', 'tweet'): TransformerConv(
                    in_channels=hid_dim,
                    out_channels=te_dim,
                    heads=num_heads,
                    edge_dim=edge_dim
                ),
            }, aggr='sum')
            self.convs.append(conv)

        # 4) JumpingKnowledge and classification head
        self.jump    = JumpingKnowledge(mode='cat')
        self.lin_out = nn.Linear(hid_dim * (num_layers + 1), out_dim)

        # 5) Optional node reconstruction head
        self.decoder = nn.Linear(hid_dim, in_dim)

    def forward(self,
                x_dict: dict,
                edge_index_dict: dict,
                edge_attr_dict: dict):
        # 1) Node feature mapping
        h = {'tweet': F.relu(self.lin_in(x_dict['tweet']))}
        outs = [h['tweet']]

        # 2) Assemble edge attributes per relation
        enc_dict = {}
        for etype, attr in edge_attr_dict.items():
            w_norm    = attr[:, 0:1]  # folded weight
            dt_norm   = attr[:, 1:2]  # time diff
            infl_flag = attr[:, 2:3]  # influencer flag
            cos_sim   = attr[:, 3:4]  # feature similarity

            dt_enc = self.time_enc(dt_norm)  # [E, 2*te_dim]

            enc_dict[etype] = torch.cat(
                [dt_enc, w_norm, infl_flag, cos_sim],
                dim=1
            )

        # 3) Multiple layers of heterogeneous conv
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index_dict, enc_dict)
            t = self.dropout(F.relu(h['tweet']))
            h['tweet'] = self.norms[i](t)
            outs.append(h['tweet'])

        # 4) Classification
        out = self.lin_out(self.jump(outs))

        # 5) Auxiliary node reconstruction
        z     = outs[-1]
        x_rec = self.decoder(z)

        return out, z, x_rec


# -------------------------------------------------- #
#                  Self-test Script                  #
# -------------------------------------------------- #
if __name__ == '__main__':
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_nodes = 50
    in_dim    = 769
    hid_dim   = 128
    out_dim   = 4
    heads     = 4
    layers    = 2

    # Random node features
    x = torch.randn(num_nodes, in_dim, device=device)
    x_dict = {'tweet': x}

    # Random edge indices
    E = 100
    src = torch.randint(0, num_nodes, (E,), device=device)
    dst = torch.randint(0, num_nodes, (E,), device=device)
    edge_index_dict = {
        ('tweet', 'TD', 'tweet'): torch.stack([src, dst], dim=0),
        ('tweet', 'BU', 'tweet'): torch.stack([dst, src], dim=0),
    }

    # Random edge attributes (4 columns)
    edge_attr = torch.randn(E, 4, device=device)
    edge_attr[:, 0] = torch.rand(E)                # w_norm ∈ (0,1)
    edge_attr[:, 2] = torch.randint(0, 2, (E,))    # infl_flag
    edge_attr = edge_attr.view(E, 4)
    edge_attr_dict = {
        ('tweet', 'TD', 'tweet'): edge_attr,
        ('tweet', 'BU', 'tweet'): edge_attr.clone(),
    }

    model = BiHGT_Edge(
        in_dim=in_dim,
        hid_dim=hid_dim,
        out_dim=out_dim,
        num_heads=heads,
        num_layers=layers,
        dropout=0.2
    ).to(device)

    out, z, x_rec = model(x_dict, edge_index_dict, edge_attr_dict)
    print('out shape  :', out.shape)    # [50, 4]
    print('z shape    :', z.shape)      # [50, 128]
    print('x_rec shape:', x_rec.shape)  # [50, 769]
