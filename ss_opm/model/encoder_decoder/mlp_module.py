import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
device = "cuda" if torch.cuda.is_available() else "cpu"


class LinearBlock(nn.Module):
    def __init__(self, h_dim=128, skip=False, dropout_p=0.1, activation="relu", norm="none"):
        super(LinearBlock, self).__init__()
        self.skip = skip
        self.fc = nn.Linear(h_dim, h_dim, bias=False)
        if norm == "batch_norm":
            self.norm = nn.BatchNorm1d(h_dim)
            if self.skip:
                nn.init.zeros_(self.norm.weight)
        elif norm == "layer_nome":
            self.norm = nn.LayerNorm(h_dim)
            if self.skip:
                nn.init.zeros_(self.norm.weight)
        else:
            self.norm = None
        if dropout_p > 0:
            self.dropout = nn.Dropout(p=dropout_p)
        else:
            self.dropout = None
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            # print("activation", activation)
            self.act = nn.GELU()
        else:
            raise RuntimeError()

    def forward(self, x):
        h = x
        h_prev = x
        h = self.act(h)
        if self.norm is not None:
            h = self.norm(h)
        if self.dropout is not None:
            h = self.dropout(h)
        h = self.fc(h)
        if self.skip:
            h = h + h_prev
        return h


class MLPBModule(nn.Module):
    def __init__(self, input_dim, output_dim, n_block, h_dim=128, skip=False, dropout_p=0.1, activation="relu", norm="bn"):
        super(MLPBModule, self).__init__()
        self.requires_preprocessed_input = True
        self.in_fc = None
        if input_dim is not None:
            self.in_fc = nn.Linear(input_dim, h_dim)
        layers = []
        for _ in range(n_block):
            layers.append(LinearBlock(h_dim=h_dim, skip=skip, dropout_p=dropout_p, activation=activation, norm=norm))
        self.layers = nn.ModuleList(layers)
        self.out_fc = None
        if output_dim is not None:
            self.out_fc = nn.Linear(h_dim, output_dim)

    def forward(self, x):
        h = x
        if self.in_fc is not None:
            h = self.in_fc(h)
        for layer in self.layers:
            h = layer(h)
        if self.out_fc is not None:
            y = self.out_fc(h)
        else:
            y = h
        return y, h



class HierarchicalMLPBModule(nn.Module):
    def __init__(self, input_dim, output_dim, n_block, h_dim=128, skip=False, dropout_p=0.1, activation="relu", norm="bn"):
        super(HierarchicalMLPBModule, self).__init__()
        self.in_fc = None
        if input_dim is not None:
            self.in_fc = nn.Linear(input_dim, h_dim)
        layers = []
        for _ in range(n_block):
            layers.append(LinearBlock(h_dim=h_dim, skip=skip, dropout_p=dropout_p, activation=activation, norm=norm))
        self.layers = nn.ModuleList(layers)
        self.out_fc = None
        if output_dim is not None:
            self.out_fc = nn.Linear(h_dim, output_dim)

    def forward(self, x):
        h = x
        if self.in_fc is not None:
            h = self.in_fc(h)
        hs = [h]
        for layer in self.layers:
            h = layer(h)
            hs.append(h)
        if self.out_fc is not None:
            y = self.out_fc(h)
        else:
            y = h
        return y, hs

###########
ts = np.load(r'D:\04_project\03_RNA2ADT\data\processed\train_cite_inputs_idxcol.npz', allow_pickle=True)


genes = ts['columns']  # 可能是 dtype=object 的 gene/protein 名
genes = np.char.partition(genes.astype('U'), '_')[:, 0]
import sys
sys.path.append(r'D:\04_project\03_RNA2ADT')
def load_gene_embeddings(gene_emb_path, pretrain_gene_list):
    """加载基因embeddings，只读取一次"""
    if gene_emb_path is None:
        gene_emb_path = r'D:\02_bioinformatics\04_st_imputaiton\scPRINT\data\main\gene_embeddings.parquet'
    
    print(f"Loading gene embeddings from {gene_emb_path}...")
    all_embeddings = pd.read_parquet(gene_emb_path)
    
    # 检查哪些基因在embedding文件中可用
    available_genes = [gene for gene in pretrain_gene_list if gene in all_embeddings.index]
    missing_genes = [gene for gene in pretrain_gene_list if gene not in all_embeddings.index]
    
    if len(available_genes) == 0:
        raise ValueError(
            f"the gene embeddings file {gene_emb_path} does not contain any of the genes given to the model"
        )
    elif len(available_genes) < len(pretrain_gene_list):
        print(
            "Warning: only a subset of the genes available in the embeddings file."
        )
        print("number of genes: ", len(available_genes))
    
    if len(missing_genes) > 0:
        print(f"Warning: {len(missing_genes)} genes not found in ESM2 embeddings, will use random initialization for them")
        print(f"Missing genes: {missing_genes[:10]}..." if len(missing_genes) > 10 else f"Missing genes: {missing_genes}")
    
    # 提取可用的embeddings
    available_embeddings = all_embeddings.loc[available_genes]
    
    print(f"Successfully loaded {len(available_genes)} gene embeddings")
    return available_embeddings, available_genes, missing_genes

gene_embeddings_data = load_gene_embeddings(r'D:\02_bioinformatics\04_st_imputaiton\scPRINT\data\main\gene_embeddings.parquet', genes)



class OmicsEmbedder(nn.Module):
    def __init__(self, pretrained_gene_list, num_hid, gene_emb=None, fix_embedding=False, 
                precpt_gene_emb=None, gene_embeddings_data=None):
        super().__init__()
        self.pretrained_gene_list = pretrained_gene_list
        self.gene_index = dict(zip(pretrained_gene_list, list(range(len(pretrained_gene_list)))))
        self.num_hid = num_hid

        # Handle ESM2 embeddings - prefer pre-loaded data over file path
        if gene_embeddings_data is not None:
            # Use pre-loaded embeddings data
            available_embeddings, available_genes, missing_genes = gene_embeddings_data
            
            if len(available_genes) == 0:
                raise ValueError(
                    f"the pre-loaded gene embeddings do not contain any of the genes given to the model"
                )
            elif len(available_genes) < len(pretrained_gene_list):
                print(
                    "Warning: only a subset of the genes available in the embeddings file."
                )
                print("number of genes: ", len(available_genes))
            
            if len(missing_genes) > 0:
                print(f"Warning: {len(missing_genes)} genes not found in ESM2 embeddings, will use random initialization for them")
                print(f"Missing genes: {missing_genes[:10]}..." if len(missing_genes) > 10 else f"Missing genes: {missing_genes}")
            
            # Initialize embeddings tensor with random values
            self.emb = nn.Parameter(torch.randn([len(pretrained_gene_list), num_hid], dtype=torch.float32)*0.005)
            
            # Fill in ESM2 embeddings for available genes
            if len(available_genes) > 0:
                sembeddings = torch.nn.AdaptiveAvgPool1d(num_hid)(
                    torch.tensor(available_embeddings.values, dtype=torch.float32)
                )
                
                # Map available genes to their positions in pretrained_gene_list
                for i, gene in enumerate(pretrained_gene_list):
                    if gene in available_genes:
                        gene_idx_in_available = available_genes.index(gene)
                        self.emb.data[i] = sembeddings[gene_idx_in_available]
            
            # Create mask for which embeddings should be frozen (ESM2 embeddings)
            self.esm2_mask = torch.zeros(len(pretrained_gene_list), dtype=torch.bool)
            for i, gene in enumerate(pretrained_gene_list):
                if gene in available_genes:
                    self.esm2_mask[i] = True
            
            if fix_embedding:
                # Only freeze ESM2 embeddings, allow missing genes to be trained
                self.emb.requires_grad = True
                # We'll handle freezing in forward pass or optimizer step
                
        elif precpt_gene_emb is not None:
            # Fallback to loading from file (original behavior)
            if precpt_gene_emb is None:
                precpt_gene_emb = '/l/users/yu.li/zgy/scPRINT/data/main/gene_embeddings.parquet'
            
            # Load all embeddings from parquet file
            all_embeddings = pd.read_parquet(precpt_gene_emb)
            
            # Check which genes are available in the embedding file
            available_genes = [gene for gene in pretrained_gene_list if gene in all_embeddings.index]
            missing_genes = [gene for gene in pretrained_gene_list if gene not in all_embeddings.index]
            
            if len(available_genes) == 0:
                raise ValueError(
                    f"the gene embeddings file {precpt_gene_emb} does not contain any of the genes given to the model"
                )
            elif len(available_genes) < len(pretrained_gene_list):
                print(
                    "Warning: only a subset of the genes available in the embeddings file."
                )
                print("number of genes: ", len(available_genes))
            
            if len(missing_genes) > 0:
                print(f"Warning: {len(missing_genes)} genes not found in ESM2 embeddings, will use random initialization for them")
                print(f"Missing genes: {missing_genes[:10]}..." if len(missing_genes) > 10 else f"Missing genes: {missing_genes}")
            
            # Initialize embeddings tensor with random values
            self.emb = nn.Parameter(torch.randn([len(pretrained_gene_list), num_hid], dtype=torch.float32)*0.005)
            
            # Fill in ESM2 embeddings for available genes
            if len(available_genes) > 0:
                available_embeddings = all_embeddings.loc[available_genes]
                sembeddings = torch.nn.AdaptiveAvgPool1d(num_hid)(
                    torch.tensor(available_embeddings.values, dtype=torch.float32)
                )
                
                # Map available genes to their positions in pretrained_gene_list
                for i, gene in enumerate(pretrained_gene_list):
                    if gene in available_genes:
                        gene_idx_in_available = available_genes.index(gene)
                        self.emb.data[i] = sembeddings[gene_idx_in_available]
            
            # Create mask for which embeddings should be frozen (ESM2 embeddings)
            self.esm2_mask = torch.zeros(len(pretrained_gene_list), dtype=torch.bool)
            for i, gene in enumerate(pretrained_gene_list):
                if gene in available_genes:
                    self.esm2_mask[i] = True
            
            if fix_embedding:
                # Only freeze ESM2 embeddings, allow missing genes to be trained
                self.emb.requires_grad = True
                # We'll handle freezing in forward pass or optimizer step
                
        elif gene_emb is not None:
            self.emb = nn.Parameter(gene_emb, requires_grad=not fix_embedding)
            self.esm2_mask = None
        else:
            self.emb = nn.Parameter(torch.randn([len(pretrained_gene_list), num_hid], dtype=torch.float32)*0.005)
            self.esm2_mask = None
            if fix_embedding:
                self.emb.requires_grad = False
        self._esm_hook_handle = None

    def freeze_esm2_embeddings(self):
        """Freeze only ESM2 embeddings while allowing others to be trained."""
        # 已注册过就别再注册
        if getattr(self, "_esm_hook_handle", None) is not None:
            return
        if hasattr(self, 'esm2_mask') and self.esm2_mask is not None:
            mask = self.esm2_mask.to(self.emb.device)  # 确保同device
            def hook(grad):
                grad = grad.clone()            # 避免原地修改
                grad[mask] = 0                 # 冻结 ESM2 对应行
                return grad
            self._esm_hook_handle = self.emb.register_hook(hook)

    def forward(self, x_dict, input_gene_list=None):

        x = x_dict
        # if 'dropout' in x_dict:
        #     indices = x._indices().t()
        #     values = x._values()
        #     temp = values.sum()
        #     values = values.float()
        #     values = torch.distributions.binomial.Binomial(values, x_dict['dropout']).sample()
        #     x = torch.sparse.FloatTensor(indices.t(), values, x.shape)

        # x = torch.log1p(x)
        # x = sparse_tpm(x)
        if input_gene_list is not None:
            gene_idx = torch.tensor([self.gene_index[o] for o in input_gene_list if o in self.gene_index]).long()
            # x_dict['input_gene_mask'] = gene_idx
        else:
            if x.shape[1] != len(self.pretrained_gene_list):
                raise ValueError('The input gene size is not the same as the pretrained gene list. Please provide the input gene list.')
            gene_idx = torch.arange(x.shape[1]).long()
        gene_idx = gene_idx.to(x.device)
        feat = F.embedding(gene_idx, self.emb)
        feat = torch.sparse.mm(x, feat)

        gene_emb = x.to_dense().unsqueeze(-1) * self.emb[gene_idx].unsqueeze(0)

        return feat, gene_emb

class OmicsEmbeddingLayer(nn.Module):
    def __init__(self, gene_list, num_hidden, norm, activation='gelu', dropout=0.3, pe_type=None, cat_pe=True, gene_emb=None,
                 inject_covariate=False, batch_num=None, precpt_gene_emb=None, freeze_embeddings=False, gene_embeddings_data=None):
        super().__init__()

        self.pe_type = pe_type
        self.cat_pe = cat_pe
        self.act = nn.ReLU()#create_activation(activation)
        self.norm0 = nn.LayerNorm(num_hidden) #create_norm(norm, num_hidden) #nn.LayerNorm(num_hidden)
        self.dropout = nn.Dropout(dropout)
        self.extra_linear = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            # create_norm(norm, num_hidden),
            nn.LayerNorm(num_hidden)
        )
        if pe_type is not None:
            if cat_pe:
                num_emb = num_hidden // 2
            else:
                num_emb = num_hidden
            self.pe_enc = select_pe_encoder(pe_type)(num_emb)
        else:
            self.pe_enc = None
            num_emb = num_hidden

        if gene_emb is None and precpt_gene_emb is None and gene_embeddings_data is None:
            self.feat_enc = OmicsEmbedder(gene_list, num_emb)
        else:
            self.feat_enc = OmicsEmbedder(gene_list, num_emb, gene_emb, freeze_embeddings, precpt_gene_emb, gene_embeddings_data)

        if inject_covariate:
            self.cov_enc = nn.Embedding(batch_num, num_emb)
            self.inject_covariate = True
        else:
            self.inject_covariate = False
        # OmicsEmbeddingLayer.__init__ 里，创建完 self.feat_enc 之后加
        if hasattr(self.feat_enc, 'esm2_mask') and self.feat_enc.esm2_mask is not None:
            self.feat_enc.freeze_esm2_embeddings()  # 只在构造时注册一次

    def forward(self, x_dict, input_gene_list=None):
        # Apply gradient masking if using ESM2 embeddings
        # if hasattr(self.feat_enc, 'esm2_mask') and self.feat_enc.esm2_mask is not None:
        #     self.feat_enc.freeze_esm2_embeddings()
        
        x, gene_emb = self.feat_enc(x_dict, input_gene_list)#self.act(self.feat_enc(x_dict, input_gene_list))

        if self.pe_enc is not None:
            pe_input = x_dict[self.pe_enc.pe_key]
            pe = self.pe_enc(pe_input) #0.
            if self.inject_covariate:
                pe = pe + self.cov_enc(x_dict['batch'])
            if self.cat_pe:
                x = torch.cat([x, pe], 1)
            else:
                x = x + pe
        x = self.extra_linear(x)

        return gene_emb, x


# embedder = OmicsEmbeddingLayer(gene_list=genes, num_hidden = 64, norm='layernorm', activation='gelu',
#                                 dropout=0.2, pe_type=None, cat_pe=True, gene_emb=None, gene_embeddings_data=gene_embeddings_data,
#                                 inject_covariate=False, batch_num=None)
# embedder.to(device=device)

# gene_to_idx = {g: i for i, g in enumerate(genes)}   # 或者用全量基因表建好映射
# unk_id = 0  # 可选，用来处理未知基因

# gene_idx = torch.tensor(
#     [gene_to_idx.get(g, unk_id) for g in genes],
#     dtype=torch.long,
#     device=next(embedder.parameters()).device
# )
# device = 'cuda:0'
# x = train_inputs[:10,].toarray()
# x = torch.from_numpy(x).to(device=device, dtype=torch.float32)
# gene_emb, x = embedder(x, genes.tolist())
# x.shape


class GeneFlowAttention(nn.Module):
    """专门用于基因自注意力的 Flow Attention 实现"""
    def __init__(self, d_model, n_heads, drop_out=0.01, eps=1e-6):
        super(GeneFlowAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(drop_out)
        self.eps = eps

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def dot_product(self, q, k, v):
        kv = torch.einsum("nhld,nhlm->nhdm", k, v)
        qkv = torch.einsum("nhld,nhdm->nhlm", q, kv)
        return qkv

    def forward(self, x):
        # input: (B, L, D); output: (B, L, D)
        B, L, _ = x.shape
        
        # 1. Linear projection
        queries = self.query_projection(x).view(B, L, self.n_heads, self.head_dim)
        keys = self.key_projection(x).view(B, L, self.n_heads, self.head_dim)
        values = self.value_projection(x).view(B, L, self.n_heads, self.head_dim)
        
        queries = queries.transpose(1, 2)  # (B, n_heads, L, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # 2. Non-negative projection
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)
        
        # 3. Flow-Attention
        # (1) Calculate incoming and outgoing flow
        sink_incoming = 1.0 / (torch.einsum("nhld,nhd->nhl", queries + self.eps, keys.sum(dim=2) + self.eps))
        source_outgoing = 1.0 / (torch.einsum("nhld,nhd->nhl", keys + self.eps, queries.sum(dim=2) + self.eps))
        
        # (2) conservation refine for source and sink
        conserved_sink = torch.einsum("nhld,nhd->nhl", queries + self.eps,
                                      (keys * source_outgoing[:, :, :, None]).sum(dim=2) + self.eps)
        conserved_source = torch.einsum("nhld,nhd->nhl", keys + self.eps,
                                        (queries * sink_incoming[:, :, :, None]).sum(dim=2) + self.eps)
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)  # for stability
        
        # (3) Competition & Allocation
        sink_allocation = torch.sigmoid(conserved_sink * (float(queries.shape[2]) / float(keys.shape[2])))
        source_competition = torch.softmax(conserved_source, dim=-1) * float(keys.shape[2])
        
        # (4) dot product
        x = (self.dot_product(queries * sink_incoming[:, :, :, None],  # for value normalization
                              keys,
                              values * source_competition[:, :, :, None])  # competition
             * sink_allocation[:, :, :, None]).transpose(1, 2)  # allocation
        
        # (5) Final projection
        x = x.reshape(B, L, -1)
        x = self.out_projection(x)
        x = self.dropout(x)
        return x


class GeneFlowformerLayer(nn.Module):
    """专门用于基因自注意力的 Flowformer Layer"""
    def __init__(self, embed_dim, num_heads, dropout=0.0, norm='layernorm', norm_first=True):
        super(GeneFlowformerLayer, self).__init__()
        self.self_attn = GeneFlowAttention(embed_dim, num_heads, dropout)
        self._ff_block = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim) if norm == 'layernorm' else nn.BatchNorm1d(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim) if norm == 'layernorm' else nn.BatchNorm1d(embed_dim)
        self.norm_first = norm_first

    def forward(self, x, attn_mask=None, output_attentions=False):
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x))
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x))
            x = self.norm2(x + self._ff_block(x))
        return x
    
    def _sa_block(self, x):
        # x 已经是 (B, L, D) 格式，直接传递给 GeneFlowAttention
        x = self.self_attn(x)
        return self.dropout1(x)


class Flowformer(nn.Module):
    """
    将可学习的 CLS（cell embedding）添加到 gene embedding 前，再做 self-attention（无注意力掩码）
    """
    def __init__(self, d: int = 1024, heads: int = 4, nlayers: int = 2, 
                 dropout: float = 0.1, cell_emb_style: str = "cls", cross_attn: bool = False):
        super().__init__()
        if cell_emb_style not in ["cls", "mean"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")
        self.cell_emb_style = cell_emb_style
        self.d = d
        self.heads = heads

        head_dim = d // heads
        if head_dim * heads != d:
            raise ValueError(f"d ({d}) must be divisible by heads ({heads}).")

        # Layer Norms
        self.ln_g = nn.LayerNorm(d)
        self.ln_c = nn.LayerNorm(d)

        # Gene+Cell 的 self-attention（沿用你已有的 GeneFlowformerLayer 定义）
        self.flowformer = nn.Sequential(*[
            GeneFlowformerLayer(embed_dim=d, num_heads=heads, dropout=dropout, norm='layernorm')
            for _ in range(nlayers)
        ])

        # ===== 新增：可学习 CLS token =====
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.cls_token, std=0.02)  # 典型初始化
        self.dp = nn.Dropout(dropout)

    def forward(
        self,
        gene_tok: torch.Tensor,      # (B, G, d)
        cell_tok: torch.Tensor = None,  # (B, d) 或 None
        *,
        use_pool_cell: bool = True,     # 是否采用 pooling 生成 cell embedding（仅当 cell_emb_style=="mean" 时生效）
        weights: torch.Tensor = None    # w-pool 的权重 (B, G) 或 (B, G, 1)
    ):
        B, G, d = gene_tok.shape
        device = gene_tok.device

        # === 生成 cell_src ===
        if self.cell_emb_style == "cls":
            # 使用可学习的 CLS；若你传入了 cell_tok 想覆盖，也可以替换为:
            # cell_src = cell_tok if (cell_tok is not None) else self.cls_token.expand(B, 1, d).squeeze(1)
            cell_src = self.cls_token.expand(B, 1, d).squeeze(1)  # (B, d)
        else:
            # mean pooling（无 mask；若给了 weights 则做加权均值）
            if (not use_pool_cell) and (cell_tok is not None):
                cell_src = cell_tok  # 显式传入时直接用
            else:
                if weights is None:
                    cell_src = gene_tok.mean(dim=1)  # (B, d)
                else:
                    if weights.dim() == 2:
                        weights = weights.unsqueeze(-1)  # (B,G,1)
                    w = weights / (weights.sum(dim=1, keepdim=True).clamp(min=1e-6))
                    cell_src = (gene_tok * w).sum(dim=1)  # (B, d)

        # 拼接 CLS 到最前： (B, G+1, d)
        cell_cls = cell_src.unsqueeze(1)                 
        gene_cell_tok = torch.cat([cell_cls, gene_tok], dim=1)

        # 无掩码，直接 LN + Transformer
        g_norm = self.ln_g(gene_cell_tok)
        gene_cell_out = self.flowformer(g_norm)
        gene_cell_tok = gene_cell_tok + self.dp(gene_cell_out)

        # 拆回
        cell_tok_out = gene_cell_tok[:, 0, :]    # (B, d)

        return cell_tok_out

class FlowformerEncoder(nn.Module):
    """
    Wraps OmicsEmbeddingLayer + Flowformer + LatentModel into a single encoder module.
    The module expects a dense tensor of gene expressions and returns the latent cell embedding
    together with the Flowformer output and KL loss from the latent layer.
    """
    def __init__(
        self,
        gene_list=None,
        embed_dim=128,
        flow_heads=4,
        flow_layers=2,
        dropout=0.1,
        cell_emb_style="cls",
        latent_dim=None,
        gene_embeddings_data=None,
    ):
        super().__init__()
        if gene_list is None:
            if "genes" not in globals():
                raise ValueError("FlowformerEncoder requires a gene_list or global genes definition.")
            gene_list = genes.tolist() if isinstance(genes, np.ndarray) else list(genes)
        self.gene_list = list(gene_list)
        if gene_embeddings_data is None and "gene_embeddings_data" in globals():
            gene_embeddings_data = globals()["gene_embeddings_data"]
        if latent_dim is None:
            latent_dim = embed_dim // 2

        self.embedder = OmicsEmbeddingLayer(
            gene_list=self.gene_list,
            num_hidden=embed_dim,
            norm="layernorm",
            activation="gelu",
            dropout=dropout,
            pe_type=None,
            cat_pe=True,
            gene_embeddings_data=gene_embeddings_data,
            inject_covariate=False,
        )
        self.flowformer = Flowformer(
            d=embed_dim,
            heads=flow_heads,
            nlayers=flow_layers,
            dropout=dropout,
            cell_emb_style=cell_emb_style,
        )
        self.requires_preprocessed_input = False
        self.latent = LatentModel()
        self.latent.add_layer(
            type="vae",
            enc_hid=embed_dim,
            latent_dim=latent_dim,
            kl_weight=1.0,
            warmup_step=10000,
        )

    def forward(self, x, input_gene_list=None):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        device = next(self.parameters()).device
        x = x.to(device)
        if not x.is_sparse:
            x = x.to_sparse()
        else:
            x = x.coalesce()

        gene_emb, pooled_repr = self.embedder(x, input_gene_list or self.gene_list)
        cell_repr = self.flowformer(gene_emb, pooled_repr)
        latent_input = {"h": cell_repr}
        z, latent_loss = self.latent(latent_input)
        return z, cell_repr, latent_loss

# encoder = Flowformer(d=64, heads=4, nlayers=2, dropout=0.2, cell_emb_style="cls")
# encoder.to(device=device)
# h = encoder(gene_emb, x)  # x: (B, G, d) 格式
# h.shape
##############

import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn import mixture
from torch import nn
import torch

############################
# 你的原始占位层 & 容器
############################
class PlaceholderLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.is_adversarial = False

    def forward(self, x_dict):
        return x_dict['h'], torch.tensor(0.).to(x_dict['h'].device)

class LatentModel(nn.Module):
    def __init__(self, configs=None):
        super().__init__()
        self.layers = nn.ModuleList([PlaceholderLayer()])
        self.alias_dict = {}
        if configs is not None:
            for c in configs:
                self.layers.append(create_latent_layer(**c))

    def forward(self, x_dict):
        total_loss = 0
        for layer in self.layers:
            x_dict['h'], loss = layer(x_dict)
            total_loss += loss
        return x_dict['h'], total_loss

    def add_layer(self, **config):
        if 'alias' in config:
            self.alias_dict[config['alias']] = len(self.layers)
        else:
            self.alias_dict[config['type']] = len(self.layers)
        self.layers.append(create_latent_layer(**config))

    def get_layer(self, alias):
        return self.layers[self.alias_dict[alias]]

    def d_train(self, x_dict):
        loss = 0
        for layer in self.layers:
            if getattr(layer, 'is_adversarial', False):
                loss += layer.d_iter(x_dict)
        return loss

############################
# 你的原始 VAE 模块（不改动）
############################
class VAELatentLayer(nn.Module):
    def __init__(self, input_dim, latent_dim, kl_weight=1., warmup_step=10000, lamda=1.0, **kwargs):
        super().__init__()
        self.hid_2mu = nn.Linear(input_dim, latent_dim)
        self.hid_2sigma = nn.Linear(input_dim, latent_dim)
        self.kl_weight = 0
        self.max_kl_weight = kl_weight
        self.step_count = 0
        self.warmup_step = warmup_step
        self.is_adversarial = False
        self.lamda = lamda

    def kl_schedule_step(self):
        self.step_count += 1
        if self.step_count < self.warmup_step:
            self.kl_weight = self.kl_weight + self.max_kl_weight / self.warmup_step
        elif self.step_count == self.warmup_step:
            pass

    def forward(self, h, var_eps=True):
        mu = self.hid_2mu(h)
        log_var = torch.clamp(self.hid_2sigma(h), -5, 5)
        if var_eps:
            sigma = (torch.exp(log_var) + 1e-4).sqrt()
            log_var = 2 * torch.log(sigma)
        else:
            sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)

        if self.training:
            z = mu + sigma * eps
            kl_loss = -0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum(1).mean() * self.kl_weight
            if kl_loss < self.lamda:
                kl_loss = 0
            self.kl_schedule_step()
        else:
            z = mu
            kl_loss = 0
        return z, kl_loss

############################
# 极薄适配器：把 x_dict -> (h_new, loss)
############################
class VAEAsLatentLayerCompat(nn.Module):
    """
    让 VAELatentLayer 兼容 LatentModel 的调用：
      (x_dict) -> (x_dict['h'], loss)
    """
    def __init__(self, input_dim=None, enc_hid=None, latent_dim=32, **kwargs):
        super().__init__()
        # 兼容老参数名 enc_hid
        if input_dim is None and enc_hid is not None:
            input_dim = enc_hid
        if input_dim is None:
            raise ValueError("VAEAsLatentLayerCompat: need input_dim or enc_hid")
        self.vae = VAELatentLayer(input_dim=input_dim, latent_dim=latent_dim, **kwargs)
        self.is_adversarial = False

    def forward(self, x_dict):
        h = x_dict['h']
        z, kl = self.vae(h)
        return z, kl

############################
# 工厂：最小只做 'vae' -> 适配器
############################
def create_latent_layer(**config) -> nn.Module:
    t = config.get('type', None)
    if t == 'vae':
        return VAEAsLatentLayerCompat(**config)
    else:
        raise ValueError(f"Unrecognized latent model name (only 'vae' supported here): {t}")

class VAELatentLayer(nn.Module):
    def __init__(self, input_dim, latent_dim, kl_weight=1., warmup_step=10000, lamda=1.0, **kwargs):#400*160
        super().__init__()
        self.hid_2mu = nn.Linear(input_dim, latent_dim)#, bias=False)
        self.hid_2sigma = nn.Linear(input_dim, latent_dim)#, bias=False)
        self.kl_weight = 0#kl_weight
        self.max_kl_weight = kl_weight
        self.step_count = 0
        self.warmup_step = warmup_step
        self.is_adversarial = False
        self.lamda = lamda

    def kl_schedule_step(self):
        self.step_count += 1
        if self.step_count < self.warmup_step:
            self.kl_weight = self.kl_weight + self.max_kl_weight / self.warmup_step
        elif self.step_count == self.warmup_step:
            pass

    def forward(self, h, var_eps=True):
        h = h
        mu = self.hid_2mu(h)
        log_var = torch.clamp(self.hid_2sigma(h), -5, 5) #+ 1e-4
        if var_eps:
            sigma = (torch.exp(log_var) + 1e-4).sqrt()
            log_var = 2 * torch.log(sigma)
        else:
            sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)

        if self.training:
            z = mu + sigma * eps
            kl_loss = -0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum(1).mean() * self.kl_weight
            if kl_loss < self.lamda:
                kl_loss = 0
            self.kl_schedule_step()
        else:
            z = mu
            kl_loss = 0
        return z, kl_loss
