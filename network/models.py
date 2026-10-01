import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn

from network.cond_tokens import PERSONA_LEARNED, PersonaEmbed, StyleEmbed, persona_tokens


class MotionSkelDiffusionMLP(nn.Module):
    def __init__(self, input_feats, shape_dim, njoints, nfeats, rot_req, clip_len,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.2,
                 ablation=None, activation="gelu", legacy=False, 
                 cond_mask_prob=0, style_cond='none', persona_cond='text'):
        super().__init__()

        self.legacy = legacy
        self.training = True
        
        self.rot_req = rot_req
        self.nfeats = nfeats
        self.njoints = njoints
        self.clip_len = clip_len
        self.input_feats = input_feats

        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.ablation = ablation
        self.activation = activation
        self.cond_mask_prob = cond_mask_prob
        
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.future_motion_process = MotionProcess(self.input_feats, self.latent_dim)
        self.past_motion_process = MotionProcess(self.input_feats, self.latent_dim)
        self.traj_trans_process = TrajProcess(2, self.latent_dim)
        self.traj_pose_process = TrajProcess(6, self.latent_dim) 
        
        # global conditions. persona_cond='text' + style_cond='none' (the defaults) = one CLIP token of the fused
        # "persona; style" prompt.  The paper's baseline receives the same structured conditions as our prior
        # (style_cond=embed, persona_cond=id_attr = a learned performer-ID token plus three typed attribute tokens).
        self.shape_process = TrajProcess(shape_dim, self.latent_dim)
        self.style_cond, self.persona_cond = style_cond, persona_cond
        self.use_style, self.use_text = style_cond == 'embed', persona_cond == 'text'
        if self.use_text:
            self.text_process = TrajProcess(512, self.latent_dim)
        self.persona = PersonaEmbed(self.latent_dim, persona_cond) if persona_cond in PERSONA_LEARNED else None
        if self.use_style:
            self.style_embed = StyleEmbed(self.latent_dim)
        
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)
        self.seqEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)
        self.output_process = OutputProcessMLP(self.input_feats, self.latent_dim, self.njoints, self.nfeats, self.latent_dim*2)

        # per-frame "this frame is clean history" flag (continuity mode). Bias-free and zero-initialised so that
        # with is_clean == 0 the output AND the gradient are exactly those of the model without the flag.
        self.clean_token = nn.Parameter(torch.zeros(self.latent_dim))
    

    @property
    def text_embed(self):
        """The name network.cond_tokens.persona_tokens expects; a property, so the state_dict keeps exactly the
        `text_process.*` parameter names (an alias attribute would register them twice)."""
        return self.text_process

    def mask_cond(self, cond, force_mask=False):
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask_shape = [1] * len(cond.shape)
            mask_shape[0] = bs
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(mask_shape)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond


    def forward(self, x, timesteps, past_motion, traj_pose, traj_trans, shape_feat, text_feat, is_clean=None,
                style_idx=None, persona=None):
        """is_clean: optional [bs, nframes] float flag (1 = clean history frame); None == all zeros.
        style_idx / persona: the structured-condition variants only (style_cond=embed, persona_cond != text); with
        the defaults the sequence is [time, shape, text, traj..., past, future]."""
        bs, njoints, nfeats, nframes = x.shape
        
        time_emb = self.embed_timestep(timesteps)  # [1, bs, L]
        traj_trans_emb = self.traj_trans_process(traj_trans) # [N/2, bs, L]     
        traj_pose_emb = self.traj_pose_process(traj_pose) # [N/2, bs, L] 
        past_motion_emb = self.past_motion_process(past_motion)  # [past_frames, bs, L] 
        shape_feat_emb = self.shape_process(shape_feat)  # [1, bs, L]
        global_emb = [time_emb, shape_feat_emb]
        if self.use_style:
            global_emb.append(self.style_embed(style_idx))
        global_emb += persona_tokens(self, text_feat, persona)   # [text token] | ID + typed attributes | nothing
        
        future_motion_emb = self.future_motion_process(x)
        if is_clean is None:
            is_clean = x.new_zeros(bs, nframes)
        future_motion_emb = future_motion_emb + is_clean.t().unsqueeze(-1) * self.clean_token  # [nframes, bs, L]
        
        xseq = torch.cat((*global_emb,
                          traj_trans_emb, traj_pose_emb,
                          past_motion_emb, future_motion_emb), axis=0)
        
        xseq = self.sequence_pos_encoder(xseq)
        output = self.seqEncoder(xseq)[-nframes:] 
        output = self.output_process(output)  
        return output
        

    def interface(self, x, timesteps, y=None, is_clean=None, **kw):
        """
            x: [batch_size, frames, njoints, nfeats], denoted x_t in the paper 
            timesteps: [batch_size] (int)
            y: a dictionary containing conditions
        """
        bs, njoints, nfeats, nframes = x.shape
        
        past_motion = y['past_motion']
        traj_pose = y['traj_pose']
        traj_trans = y['traj_trans']
        shape_feat = y['shape_feat']
        text_feat = y['text_feat']
        
        # CFG on past motion at same time (training only; eval keeps the full history)
        # keep_batch_idx = torch.rand(bs, device=text_feat.device) < (1-self.cond_mask_prob)
        # text_feat = text_feat * keep_batch_idx.view((bs, 1, 1))
        if self.training and self.cond_mask_prob > 0:
            keep_batch_idx = torch.rand(bs, device=past_motion.device) < (1-self.cond_mask_prob)
            past_motion = past_motion * keep_batch_idx.view((bs, 1, 1, 1))
        
        return self.forward(x, timesteps, past_motion, traj_pose, traj_trans, shape_feat, text_feat, is_clean, **kw)


class MotionSkelDiffusionSmooth(nn.Module):
    def __init__(self, input_feats, shape_dim, njoints, nfeats, rot_req, clip_len,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.2,
                 ablation=None, activation="gelu", legacy=False, 
                 cond_mask_prob=0):
        super().__init__()

        self.legacy = legacy
        self.training = True
        
        self.rot_req = rot_req
        self.nfeats = nfeats
        self.njoints = njoints
        self.clip_len = clip_len
        self.input_feats = input_feats

        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.ablation = ablation
        self.activation = activation
        self.cond_mask_prob = cond_mask_prob
        
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.future_motion_process = MotionProcess(self.input_feats, self.latent_dim)
        self.past_motion_process = MotionProcess(self.input_feats, self.latent_dim)
        self.traj_trans_process = TrajProcess(2, self.latent_dim)
        self.traj_pose_process = TrajProcess(6, self.latent_dim) 
        
        # global conditions
        self.shape_process = TrajProcess(shape_dim, self.latent_dim)
        self.text_process = TrajProcess(512, self.latent_dim)
        
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)
        self.seqEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)
        self.output_process = OutputProcessMLP(self.input_feats, self.latent_dim, self.njoints, self.nfeats, self.latent_dim*2)
    

    def mask_cond(self, cond, force_mask=False):
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask_shape = [1] * len(cond.shape)
            mask_shape[0] = bs
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(mask_shape)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond


    def forward(self, x, timesteps, past_motion, traj_pose, traj_trans, shape_feat, text_feat):
        bs, njoints, nfeats, nframes = x.shape
        
        time_emb = self.embed_timestep(timesteps)  # [1, bs, L]
        traj_trans_emb = self.traj_trans_process(traj_trans) # [N/2, bs, L]     
        traj_pose_emb = self.traj_pose_process(traj_pose) # [N/2, bs, L] 
        past_motion_emb = self.past_motion_process(past_motion)  # [past_frames, bs, L] 
        shape_feat_emb = self.shape_process(shape_feat)  # [1, bs, L]
        text_feat_emb = self.text_process(text_feat)  # [1, bs, L]
        
        future_motion_emb = self.future_motion_process(x)
        
        xseq = torch.cat((time_emb, shape_feat_emb, text_feat_emb,
                          traj_trans_emb, traj_pose_emb,
                          past_motion_emb, future_motion_emb), axis=0)
        
        xseq = self.sequence_pos_encoder(xseq)
        output = self.seqEncoder(xseq)[-nframes:] 
        output = self.output_process(output)  
        
        fade_in_frames = 5
        last_past_frame = past_motion[:, :, :, -1].unsqueeze(-1)
        fade_weights = torch.linspace(0.2, 1.0, fade_in_frames).unsqueeze(0).unsqueeze(1).unsqueeze(2).to(output.device)
        inverse_fade_weights = (1.0 - fade_weights).to(output.device)
        output[:, :, :, :fade_in_frames] = (
            inverse_fade_weights * last_past_frame.squeeze(-1).unsqueeze(-1) +
            fade_weights * output[:, :, :, :fade_in_frames]
        )
        
        return output
        

    def interface(self, x, timesteps, y=None):
        """
            x: [batch_size, frames, njoints, nfeats], denoted x_t in the paper 
            timesteps: [batch_size] (int)
            y: a dictionary containing conditions
        """
        bs, njoints, nfeats, nframes = x.shape
        
        past_motion = y['past_motion']
        traj_pose = y['traj_pose']
        traj_trans = y['traj_trans']
        shape_feat = y['shape_feat']
        text_feat = y['text_feat']
        
        # CFG on past motion at same time
        # keep_batch_idx = torch.rand(bs, device=text_feat.device) < (1-self.cond_mask_prob)
        # text_feat = text_feat * keep_batch_idx.view((bs, 1, 1))
        keep_batch_idx = torch.rand(bs, device=past_motion.device) < (1-self.cond_mask_prob)
        past_motion = past_motion * keep_batch_idx.view((bs, 1, 1, 1))
        
        return self.forward(x, timesteps, past_motion, traj_pose, traj_trans, shape_feat, text_feat)


class MotionProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats) 
        x = self.poseEmbedding(x)  
        return x


class TrajProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs,  nfeats, nframes = x.shape
        x = x.permute((2, 0, 1))
        x = self.poseEmbedding(x)  
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        output = self.poseFinal(output)  
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  
        return output
    

class OutputProcessMLP(nn.Module):
    def __init__(self, input_feats, latent_dim, njoints, nfeats, hidden_dim=512): # add hidden_dim as parameter
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.hidden_dim = hidden_dim # store hidden dimension
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, self.input_feats)
        )

    def forward(self, output):
        nframes, bs, d = output.shape
        output = self.mlp(output)  # use MLP instead of single linear layer
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  
        return output
    
    
class EmbedSparse(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input.to(torch.long) 
        output = self.action_embedding[idx]
        return output

