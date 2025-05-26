import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from radl.radl_layers import CBAM, CrossAttention, LayoutAttention
from transformers import CLIPTokenizer, CLIPTextModel

class FourierEmbedder():
    def __init__(self, num_freqs=64, temperature=100):
        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** ( torch.arange(num_freqs) / num_freqs )

    @ torch.no_grad()
    def __call__(self, x, cat_dim=-1):
        out = []
        for freq in self.freq_bands:
            out.append( torch.sin( freq*x ) )
            out.append( torch.cos( freq*x ) )
        return torch.cat(out, cat_dim)  


class PositionNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_freqs=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs)
        self.position_dim = num_freqs * 2 * 4  

        # -------------------------------------------------------------- #
        self.linears_position = nn.Sequential(
            nn.Linear(self.position_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, boxes):
        boxes = boxes[..., :4]
        xyxy_embedding = self.fourier_embedder(boxes)
        xyxy_embedding = self.linears_position(xyxy_embedding)
        return xyxy_embedding
    
class VerbEmbeddingMLP(nn.Module):
    def __init__(self, in_dim=768, out_dim=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )
    
    def forward(self, x):  # x: (B, 768)
        return self.mlp(x).unsqueeze(-1).unsqueeze(-1)  # → (B, 768, 1, 1)
    
class VerbCrossAttention(nn.Module):
    def __init__(self, C, heads):
        super().__init__()
        self.heads = heads
        self.vca = CrossAttention(query_dim=C, context_dim=768,
                             heads=heads, dim_head=C // heads,
                             dropout=0.0)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    
    def forward(self, x, instance_num, verbs, guidance_mask):
        B, HW, C = x.shape
        x = x.view(B//instance_num, instance_num, HW, C)[:, 0, ...] # (B, HW, C)
        verb_token = self.tokenizer(
            verbs, padding='max_length', max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        ).to(x.device)
        verb_token_ids = verb_token.input_ids
        outputs = self.text_encoder(verb_token_ids)
        verb_embeddings = outputs[0] # (B, seq_len, 768)
        # verb_embeddings = verb_embeddings.unsqueeze(1).repeat(B*instance_num, 1,1, 1).squeeze(1) # (B*ins_num, seq_len, 768)
        
        guidance_mask = guidance_mask.sum(dim=1, keepdims=True).clip(0, 1)
        guidance_mask = guidance_mask.view(B//instance_num, HW, 1)
        # print(x.shape, verb_embeddings.shape, guidance_mask.shape)
        
        attn_feature, _ = self.vca(x, context=verb_embeddings[:, 1:, ...], return_attn=True)
        attn_feature = attn_feature * guidance_mask
        # print(attn_feature.shape)
        # exit()


        return attn_feature
        
        

class SAC(nn.Module):
    def __init__(self, C, number_pro=30):
        super().__init__()
        self.C = C
        self.number_pro = number_pro
        self.conv1 = nn.Conv2d(C + 1, C, 1, 1)
        self.cbam1 = CBAM(C)
        self.conv2 = nn.Conv2d(C, 1, 1, 1)
        self.cbam2 = CBAM(number_pro, reduction_ratio=1)
        self.verb_embedding_mlp = VerbEmbeddingMLP(768, C)
        self.text_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    def forward(self, x, guidance_mask, verbs, sac_scale=None):
        '''
        :param x: (B, phase_num, HW, C)
        :param guidance_mask: (B, phase_num, H, W)
        :return:
        '''
        B, phase_num, HW, C = x.shape
        _, _, H, W = guidance_mask.shape
        guidance_mask = guidance_mask.view(guidance_mask.shape[0], phase_num, -1)[
            ..., None]  # (B, phase_num, HW, 1)

        null_x = torch.zeros_like(x[:, [0], ...]).to(x.device)
        null_mask = torch.zeros_like(guidance_mask[:, [0], ...]).to(guidance_mask.device)

        x = torch.cat([x, null_x], dim=1)
        guidance_mask = torch.cat([guidance_mask, null_mask], dim=1)
        phase_num += 1


        scale = torch.cat([x, guidance_mask], dim=-1)  # (B, phase_num, HW, C+1)
        scale = scale.view(-1, H, W, C + 1)  # (B * phase_num, H, W, C+1)
        scale = scale.permute(0, 3, 1, 2)  # (B * phase_num, C+1, H, W)
        scale = self.conv1(scale)  # (B * phase_num, C, H, W)
        scale = self.cbam1(scale)  # (B * phase_num, C, H, W)
        scale = self.conv2(scale)  # (B * phase_num, 1, H, W)
        scale = scale.view(B, phase_num, H, W)  # (B, phase_num, H, W)

        null_scale = scale[:, [-1], ...]
        scale = scale[:, :-1, ...]
        x = x[:, :-1, ...]

        pad_num = self.number_pro - phase_num + 1

        ori_phase_num = scale[:, 1:-1, ...].shape[1]
        phase_scale = torch.cat([scale[:, 1:-1, ...], null_scale.repeat(1, pad_num, 1, 1)], dim=1)
        shuffled_order = torch.randperm(phase_scale.shape[1])
        inv_shuffled_order = torch.argsort(shuffled_order)

        random_phase_scale = phase_scale[:, shuffled_order, ...]

        scale = torch.cat([scale[:, [0], ...], random_phase_scale, scale[:, [-1], ...]], dim=1)
        # (B, number_pro, H, W)

        scale = self.cbam2(scale)  # (B, number_pro, H, W)
        scale = scale.view(B, self.number_pro, HW)[..., None]  # (B, number_pro, HW)

        random_phase_scale = scale[:, 1: -1, ...]
        phase_scale = random_phase_scale[:, inv_shuffled_order[:ori_phase_num], :]
        if sac_scale is not None:
            instance_num = len(sac_scale)
            for i in range(instance_num):
                phase_scale[:, i, ...] = phase_scale[:, i, ...] * sac_scale[i]


        scale = torch.cat([scale[:, [0], ...], phase_scale, scale[:, [-1], ...]], dim=1)

        scale = scale.softmax(dim=1)  # (B, phase_num, HW, 1)
        out = (x * scale).sum(dim=1, keepdims=True)  # (B, 1, HW, C)

        return out, scale


class RaDL(nn.Module):
    def __init__(self, C, attn_type='base', context_dim=768, heads=8):
        super().__init__()
        self.ea = CrossAttention(query_dim=C, context_dim=1536,
                             heads=heads, dim_head=C // heads,
                             dropout=0.0)
        self.la = LayoutAttention(query_dim=C,
                                    heads=heads, dim_head=C // heads,
                                    dropout=0.0)
        self.vca = VerbCrossAttention(C, heads)
        self.norm = nn.LayerNorm(C)
        self.sac = SAC(C)
        self.pos_net = PositionNet(in_dim=768, out_dim=768, fourier_freqs=32)

    def forward(self, ca_x, guidance_mask, other_info, return_fuser_info=False):
        full_H = other_info['height']
        full_W = other_info['width']
        B, _, HW, C = ca_x.shape
        instance_num = guidance_mask.shape[1]
        down_scale = int(math.sqrt(full_H * full_W // ca_x.shape[2]))
        H = full_H // down_scale
        W = full_W // down_scale
        guidance_mask = F.interpolate(guidance_mask, size=(H, W), mode='bilinear')

        supplement_mask = other_info['supplement_mask']
        supplement_mask = F.interpolate(supplement_mask, size=(H, W), mode='bilinear')
        image_token = other_info['image_token']
        assert image_token.shape == ca_x.shape
        context = other_info['context_pooler']

        context = other_info['context_pooler']  # [B * (instance_num + 1), 768]
        context = context.view(B, instance_num + 1, -1)  # [B, 7, 768]
        context = context[:, 1:, :]  # [B, 6, 768]
        context = context.reshape(B * instance_num, 1, -1)  # [12, 1, 768]

        box = other_info['box']  # [B, instance_num, 4]
        box = box.reshape(B * instance_num, 1, -1)  # [12, 1, 4]
        box_token = self.pos_net(box)  # [12, 1, 768]

        context = torch.cat([context, box_token], dim=2)  # [12, 1, 1536]

        ca_scale = other_info.get('ca_scale', None)
        ea_scale = other_info.get('ea_scale', None)
        sac_scale = other_info.get('sac_scale', None)

        ea_x, ea_attn = self.ea(self.norm(image_token[:, 1:, ...].reshape(B * instance_num, HW, C)),
                        context=context, return_attn=True)
        ea_x = ea_x.view(B, instance_num, HW, C)
        ea_x = ea_x * guidance_mask.view(B, instance_num, HW, 1)
        # print("e shape: ", ea_x.shape)

        vca_x = self.vca(self.norm(image_token[:, 1:, ...].reshape(B * instance_num, HW, C)), instance_num,
                                    verbs=other_info['verbs'], guidance_mask=guidance_mask)
        vca_x = vca_x.view(B, 1, HW, C)
        # vca_x = vca_x.view(B, instance_num, HW, C)
        # vca_x = vca_x * guidance_mask.view(B, instance_num, HW, 1)

        ca_x = ca_x.clone()  # <- Fix: avoid in-place op on view
        ca_x[:, 1:, ...] = ca_x[:, 1:, ...] * guidance_mask.view(B, instance_num, HW, 1)

        if ca_scale is not None:
            assert len(ca_scale) == instance_num
            for i in range(instance_num):
                ca_x[:, i+1, ...] = ca_x[:, i+1, ...] * ca_scale[i] + ea_x[:, i, ...] * ea_scale[i]
        else:
            ca_x[:, 1:, ...] = ca_x[:, 1:, ...] + ea_x

        ori_image_token = image_token[:, 0, ...]  # (B, HW, C)
        fusion_template = self.la(x=ori_image_token, guidance_mask=torch.cat([guidance_mask[:, :, ...], supplement_mask], dim=1))
        fusion_template = fusion_template.view(B, 1, HW, C)

        ca_x = torch.cat([ca_x, fusion_template, vca_x], dim=1)
        ca_x = ca_x.clone()  # <- Fix: avoid in-place op on view
        ca_x[:, 0, ...] = ca_x[:, 0, ...] * supplement_mask.view(B, HW, 1)

        guidance_mask = torch.cat([
            supplement_mask,
            guidance_mask,
            torch.ones(B, 2, H, W).to(guidance_mask.device)
            # torch.ones(B, 1, H, W).to(guidance_mask.device)
        ], dim=1)

        out_RaDL, sac_scale = self.sac(ca_x, guidance_mask, other_info['verbs'], sac_scale=sac_scale)

        if return_fuser_info:
            fuser_info = {
                'sac_scale': sac_scale.view(B, instance_num + 2, H, W),
                'ea_attn': ea_attn.mean(dim=1).view(B, instance_num, H, W, 2),
            }
            return out_RaDL, fuser_info
        else:
            return out_RaDL

class NaiveFuser(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, ca_x, guidance_mask, other_info, return_fuser_info=False):
        # ca_x: (B, instance_num+1, HW, C)
        # guidance_mask: (B, instance_num, H, W)
        # box: (instance_num, 4)
        # image_token: (B, instance_num+1, HW, C)
        full_H = other_info['height']
        full_W = other_info['width']
        B, _, HW, C = ca_x.shape
        instance_num = guidance_mask.shape[1]
        down_scale = int(math.sqrt(full_H * full_W // ca_x.shape[2]))
        H = full_H // down_scale
        W = full_W // down_scale
        guidance_mask = F.interpolate(guidance_mask, size=(H, W), mode='bilinear')   # (B, instance_num, H, W)
        guidance_mask = torch.cat([torch.ones(B, 1, H, W).to(guidance_mask.device), guidance_mask * 10], dim=1)  # (B, instance_num+1, H, W)
        guidance_mask = guidance_mask.view(B, instance_num + 1, HW, 1)
        out_RaDL = (ca_x * guidance_mask).sum(dim=1) / (guidance_mask.sum(dim=1) + 1e-6)
        if return_fuser_info:
            return out_RaDL, None
        else:
            return out_RaDL