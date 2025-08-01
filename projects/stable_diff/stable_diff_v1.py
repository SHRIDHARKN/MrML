import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    
    def __init__(self, in_channels, out_channels, 
                 img_size, time_in_feat, time_out_feat, 
                 text_in_feat, text_out_feat, kernel_size=3): 
        
        super(Block, self).__init__()
        
        self.img_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.time_dense = nn.Sequential(
            nn.Linear(in_features=time_in_feat, out_features=time_out_feat),
            nn.LayerNorm(img_size),
            nn.ReLU()
            ) 
        self.text_dense = nn.Sequential(
            nn.Linear(in_features=text_in_feat, out_features=text_out_feat),
            nn.LayerNorm(img_size),
            nn.ReLU()
            )
        self.norm = nn.LayerNorm([out_channels, img_size, img_size])

    def forward(self, img, time_feat, text_feat): 
        
        img_feats = F.relu(self.img_conv(img))
        time_feat_processed = F.relu(self.time_dense(time_feat)).unsqueeze(-1).unsqueeze(-1)
        text_feat_processed = F.relu(self.text_dense(text_feat)).unsqueeze(-1).unsqueeze(-1)
        feats = img_feats * time_feat_processed * text_feat_processed
        out = F.relu(self.norm(feats))
        return out


class DiffusionModel(nn.Module):
    def __init__(self, img_size=64, img_channels=3, text_embed_dim=384, time_embed_dim=64):
        super(DiffusionModel, self).__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        self.text_embed_dim = text_embed_dim
        self.time_embed_dim = time_embed_dim
        self.half_img_size = self.img_size//2
        self.quarter_img_size = self.img_size//4
        self.final_decoder_channels = self.img_size*self.img_channels
        self.EncoderBlock1 = Block(in_channels=self.img_channels,
                                   out_channels=self.img_size,
                                   img_size=self.img_size,
                                   kernel_size=3,
                                   time_in_feat=self.time_embed_dim,
                                   time_out_feat=self.time_embed_dim,
                                   text_in_feat=self.text_embed_dim,
                                   text_out_feat=self.img_size
                                   )
        
        self.EncoderBlock2 = Block(in_channels=self.img_channels,
                                   out_channels=self.half_img_size,
                                   img_size=self.half_img_size,
                                   kernel_size=3,
                                   time_in_feat=self.time_embed_dim,
                                   time_out_feat=self.time_embed_dim//2,
                                   text_in_feat=self.text_embed_dim,
                                   text_out_feat=self.half_img_size
                                   )
        
        self.BottleNeck = nn.Sequential(
                                    nn.Linear(in_features=(self.half_img_size*\
                                    self.quarter_img_size*self.quarter_img_size) + self.time_embed_dim,
                                    out_features=self.half_img_size), 
                                    nn.LayerNorm(self.half_img_size),
                                    nn.ReLU(),
                                    nn.Linear(self.half_img_size, 
                                              self.half_img_size * self.half_img_size * self.half_img_size),
                                    nn.LayerNorm(self.half_img_size * self.half_img_size * self.half_img_size),
                                    nn.ReLU()
                                    )
        
        
        self.DecoderBlock2 = Block(in_channels=self.img_size, 
                                    out_channels=self.half_img_size, 
                                    img_size=self.half_img_size, 
                                    kernel_size=3, 
                                    time_in_feat=self.img_size, 
                                    time_out_feat=self.half_img_size,
                                    text_in_feat=self.text_embed_dim, 
                                    text_out_feat=self.half_img_size
                                )
        
        self.DecoderBlock1 = Block(in_channels=self.final_decoder_channels, 
                                    out_channels=self.img_size, 
                                    img_size=self.img_size, 
                                    kernel_size=3, 
                                    time_in_feat=self.img_size, 
                                    time_out_feat=self.img_size,
                                    text_in_feat=self.text_embed_dim, 
                                    text_out_feat=self.img_size
                                )

        
        
    def forward(self, img_inp, time_inp, text_inp):
        
        x_l = self.EncoderBlock1(img=img_inp, 
                                time_feat=time_inp, 
                                text_feat=text_inp
                                )
        
        x = F.max_pool2d(x_l,2)
        
        x_m = self.EncoderBlock2(img=x,
                                 time_feat=time_inp,
                                 text_feat=text_inp)
        
        x = F.max_pool2d(x_m,2)
        
        x_flat = nn.Flatten()(x)
        x_concat = torch.cat([x_flat, time_inp], dim=1)
        
        x = self.BottleNeck(x_concat)
        x = x.view(-1, self.half_img_size, self.half_img_size, self.half_img_size)
        x = torch.cat([x, x_m], dim=1)
        
        x = self.DecoderBlock2(img=x, time_feat=time_inp, text_feat=text_inp)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, x_l], dim=1)
        
        x = self.DecoderBlock1(img=x, time_feat=time_inp, text_feat=text_inp)
        
        out = nn.Conv2d(self.img_size,self.img_channels,kernel_size=1)(x)
        
        return out
