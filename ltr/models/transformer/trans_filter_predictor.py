import torch
import torch.nn as nn

from ltr.models.layers.normalization import InstanceL2Norm
from ltr.models.transformer.position_encoding import PositionEmbeddingSine
from ltr.models.transformer.transformer import  TransformerEncoderLayer, TransformerEncoder


class FusionModule(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=4, dim_feedforward=2048,
                 dropout=0.1, query_from='flex_emb', activation="relu", normalize_before=False,
                 norm_scale = None, no_conv=False, no_flex_emb=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.flex_embed = nn.Embedding(1, self.d_model) if not no_flex_emb else None
        self.extract_index = ['res_feat', 'trans_feat', 'flex_emb'].index(query_from)
        norm_scale = 1/16.0 if norm_scale is None else norm_scale
        if not no_conv:
            feat_layers = []
            feat_layers.append(nn.Conv2d(d_model, d_model*4, kernel_size=1, padding=0, bias=False, stride=1))
            feat_layers.append(nn.ReLU(inplace=True))
            feat_layers.append(nn.Conv2d(d_model * 4, d_model, kernel_size=1, padding=0, bias=False, stride=1))
            feat_layers.append(InstanceL2Norm(scale=norm_scale))
            self.conv = nn.Sequential(*feat_layers)
        else:
            self.conv = None
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, res_feat, trans_feat):
        #
        n_tr, b, c, h, w = res_feat.shape
        # N_tr, B, C, H, W ==permute(2, 0, 1, 3, 4)==> C, N_tr, B, H, W
        # C, N_tr, B, H, W ==flatten(1)==> C, N_tr*B*H*W
        # C, N_tr*B*H*W ==permute(1, 0)==> N_tr*B*H*W, C
        # N_tr*B*H*W, C ==unsqueeze(0)==> 1, N_tr*B*H*W, C
        res = res_feat.permute(2, 0, 1, 3, 4).flatten(1).permute(1, 0).unsqueeze(0)
        trans = trans_feat.permute(2, 0, 1, 3, 4).flatten(1).permute(1, 0).unsqueeze(0)
        # flex_embed = (1, C) , ones = (1, N_tr*B*H*W, 1)
        if self.flex_embed is not None:
            flex = self.flex_embed.weight * torch.ones(*res.shape[:-1], 1, device=res.device)
            feat = torch.cat([res, trans, flex], dim=0)
        else:
            feat = torch.cat([res, trans], dim=0)
        feat = self.encoder(feat)[self.extract_index]
        # N_tr*B*H*W, C ==permute(1,0)==> C, N_tr*B*H*W
        # C, N_tr*B*H*W ==view(C, N_tr, B, H, W)==> C, N_tr, B, H, W
        # C, N_tr, B, H, W ==permute(1,2,0,3,4)==> N_tr, B, C, H, W
        feat = feat.permute(1, 0).view(c, n_tr, b, h, w).permute(1, 2, 0, 3, 4)
        # x = feat
        feat = feat.reshape(-1, *feat.shape[-3:])
        if self.conv is not None:
            feat = self.conv(feat)
        feat = feat.reshape(n_tr, b, c, h, w)
        return feat


class NewFusionModule(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=4, dim_feedforward=2048,
                 dropout=0.1, query_from='flex_emb', activation="relu", normalize_before=False,
                 norm_scale = None, no_conv=False, no_flex_emb=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.flex_embed = nn.Embedding(1, self.d_model) if not no_flex_emb else None
        self.extract_index = ['res_feat', 'trans_feat', 'flex_emb'].index(query_from)
        norm_scale = 1/16.0 if norm_scale is None else norm_scale
        if not no_conv:
            feat_layers = []
            feat_layers.append(nn.Conv2d(d_model, d_model*4, kernel_size=1, padding=0, bias=False, stride=1))
            feat_layers.append(nn.ReLU(inplace=True))
            feat_layers.append(nn.Conv2d(d_model * 4, d_model, kernel_size=1, padding=0, bias=False, stride=1))
            feat_layers.append(InstanceL2Norm(scale=norm_scale))
            self.conv = nn.Sequential(*feat_layers)
        else:
            self.conv = None
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, res_feat, trans_feat):
        #
        n_tr, b, c, h, w = res_feat.shape
        # N_tr, B, C, H, W ==permute(2, 0, 1, 3, 4)==> C, N_tr, B, H, W
        # C, N_tr, B, H, W ==flatten(1)==> C, N_tr*B*H*W
        # C, N_tr*B*H*W ==permute(1, 0)==> N_tr*B*H*W, C
        # N_tr*B*H*W, C ==unsqueeze(0)==> 1, N_tr*B*H*W, C
        res = res_feat.permute(2, 0, 1, 3, 4).flatten(1).permute(1, 0).unsqueeze(0)
        trans = trans_feat.permute(2, 0, 1, 3, 4).flatten(1).permute(1, 0).unsqueeze(0)
        # flex_embed = (1, C) , ones = (1, N_tr*B*H*W, 1)
        if self.flex_embed is not None:
            flex = self.flex_embed.weight * torch.ones(*res.shape[:-1], 1, device=res.device)
            feat = torch.cat([res, trans, flex], dim=0)
        else:
            feat = torch.cat([res, trans], dim=0)
        feat = self.encoder(feat)[self.extract_index]
        # N_tr*B*H*W, C ==permute(1,0)==> C, N_tr*B*H*W
        # C, N_tr*B*H*W ==view(C, N_tr, B, H, W)==> C, N_tr, B, H, W
        # C, N_tr, B, H, W ==permute(1,2,0,3,4)==> N_tr, B, C, H, W
        feat = feat.permute(1, 0).view(c, n_tr, b, h, w).permute(1, 2, 0, 3, 4)
        # x = feat
        feat = feat.reshape(-1, *feat.shape[-3:])
        if self.conv is not None:
            feat = self.conv(feat)
        feat = feat.reshape(n_tr, b, c, h, w)
        return feat





class ConvFusionModule(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 norm_scale = None, no_conv=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        norm_scale = 1/16.0 if norm_scale is None else norm_scale
        if not no_conv:
            feat_layers = []
            feat_layers.append(nn.Conv2d(2*d_model, d_model*4, kernel_size=1, padding=0, bias=False, stride=1))
            feat_layers.append(nn.ReLU(inplace=True))
            feat_layers.append(nn.Conv2d(4*d_model, d_model*2, kernel_size=1, padding=0, bias=False, stride=1))
            feat_layers.append(nn.ReLU(inplace=True))
            feat_layers.append(nn.Conv2d(2 * d_model, d_model * 4, kernel_size=1, padding=0, bias=False, stride=1))
            feat_layers.append(nn.ReLU(inplace=True))
            feat_layers.append(nn.Conv2d(4 * d_model, d_model * 2, kernel_size=1, padding=0, bias=False, stride=1))
            feat_layers.append(nn.ReLU(inplace=True))
            feat_layers.append(nn.Conv2d(2 * d_model, d_model * 4, kernel_size=1, padding=0, bias=False, stride=1))
            feat_layers.append(nn.ReLU(inplace=True))
            feat_layers.append(nn.Conv2d(4 * d_model, d_model * 2, kernel_size=1, padding=0, bias=False, stride=1))
            feat_layers.append(nn.ReLU(inplace=True))
            feat_layers.append(nn.Conv2d(2*d_model, d_model*4, kernel_size=1, padding=0, bias=False, stride=1))
            feat_layers.append(nn.ReLU(inplace=True))
            feat_layers.append(nn.Conv2d(4 * d_model, d_model * 2, kernel_size=1, padding=0, bias=False, stride=1))
            feat_layers.append(nn.ReLU(inplace=True))
            feat_layers.append(nn.Conv2d(2*d_model, d_model*4, kernel_size=1, padding=0, bias=False, stride=1))
            feat_layers.append(nn.ReLU(inplace=True))
            feat_layers.append(nn.Conv2d(4 * d_model, d_model * 2, kernel_size=1, padding=0, bias=False, stride=1))
            feat_layers.append(nn.ReLU(inplace=True))
            feat_layers.append(nn.Conv2d(2*d_model, d_model*4, kernel_size=1, padding=0, bias=False, stride=1))
            feat_layers.append(nn.ReLU(inplace=True))
            feat_layers.append(nn.Conv2d(d_model * 4, d_model, kernel_size=1, padding=0, bias=False, stride=1))
            feat_layers.append(InstanceL2Norm(scale=norm_scale))
            self.convx = nn.Sequential(*feat_layers)
        else:
            self.convx = None

    def forward(self, res_feat, trans_feat):
        #
        n_tr, b, c, h, w = res_feat.shape
        # N_tr, B, C, H, W ==permute(2, 0, 1, 3, 4)==> C, N_tr, B, H, W
        # C, N_tr, B, H, W ==flatten(1)==> C, N_tr*B*H*W
        # C, N_tr*B*H*W ==permute(1, 0)==> N_tr*B*H*W, C
        # N_tr*B*H*W, C ==unsqueeze(0)==> 1, N_tr*B*H*W, C
        # res = res_feat.permute(2, 0, 1, 3, 4).flatten(1).permute(1, 0).unsqueeze(0)
        # trans = trans_feat.permute(2, 0, 1, 3, 4).flatten(1).permute(1, 0).unsqueeze(0)
        # flex_embed = (1, C) , ones = (1, N_tr*B*H*W, 1)
        feat = torch.cat([res_feat.reshape(-1, c,h,w), trans_feat.reshape(-1, c,h,w)], dim=1)
        # feat = self.encoder(feat)[self.extract_index]
        # N_tr*B*H*W, C ==permute(1,0)==> C, N_tr*B*H*W
        # C, N_tr*B*H*W ==view(C, N_tr, B, H, W)==> C, N_tr, B, H, W
        # C, N_tr, B, H, W ==permute(1,2,0,3,4)==> N_tr, B, C, H, W
        # feat = feat.permute(1, 0).view(c, n_tr, b, h, w).permute(1, 2, 0, 3, 4)
        # x = feat
        # feat = feat.reshape(-1, *feat.shape[-3:])
        feat = self.convx(feat)
        feat = feat.reshape(n_tr, b, c, h, w)
        return feat


def MLP(channels, do_bn=True):
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class FilterPredictor(nn.Module):
    def __init__(self, transformer, feature_sz, fusion_module, use_test_frame_encoding=True):
        super().__init__()
        self.transformer = transformer
        self.feature_sz = feature_sz
        self.fusion_module = fusion_module
        self.use_test_frame_encoding = use_test_frame_encoding

        self.box_encoding = MLP([4, self.transformer.d_model//4, self.transformer.d_model, self.transformer.d_model])

        self.query_embed_fg = nn.Embedding(1, self.transformer.d_model)

        if self.use_test_frame_encoding:
            self.query_embed_test = nn.Embedding(1, self.transformer.d_model)

        self.query_embed_fg_decoder = self.query_embed_fg

        self.pos_encoding = PositionEmbeddingSine(num_pos_feats=self.transformer.d_model//2, sine_type='lin_sine',
                                                  avoid_aliazing=True, max_spatial_resolution=feature_sz)

    def forward(self, train_feat, test_feat, trans_train_feat, trans_test_feat, train_label, train_ltrb_target, *args, **kwargs):
        return self.predict_filter(train_feat, test_feat, trans_train_feat, trans_test_feat, train_label, train_ltrb_target, *args, **kwargs)

    def get_positional_encoding(self, feat):
        nframes, nseq, _, h, w = feat.shape

        mask = torch.zeros((nframes * nseq, h, w), dtype=torch.bool, device=feat.device)
        pos = self.pos_encoding(mask)

        return pos.reshape(nframes, nseq, -1, h, w)

    def predict_filter(self, train_feat, test_feat, trans_train_feat, trans_test_feat, train_label, train_ltrb_target, *args, **kwargs):
        def printv(x, y):
            print(f"{x}: {y.shape} , mean:{y.mean():.4f}, min:{y.min():.4f}, max:{y.max():.4f}, std:{y.std():.4f}")

        #train_label size guess: Nf_tr, Ns, H, W.
        if train_feat.dim() == 4:
            train_feat = train_feat.unsqueeze(1)
        if test_feat.dim() == 4:
            test_feat = test_feat.unsqueeze(1)
        if trans_train_feat.dim() == 4:
            trans_train_feat = trans_train_feat.unsqueeze(1)
        if trans_test_feat.dim() == 4:
            trans_test_feat = trans_test_feat.unsqueeze(1)
        if train_ltrb_target.dim() == 4:
            train_ltrb_target = train_ltrb_target.unsqueeze(1)

        h, w = test_feat.shape[-2:]

        test_pos = self.get_positional_encoding(test_feat) # Nf_te, Ns, C, H, W
        train_pos = self.get_positional_encoding(train_feat) # Nf_tr, Ns, C, H, W
        # printv("FP:train_feat", train_feat)
        # printv("FP:train_pos", train_pos)

        # Fusion is happening.
        train_feat = self.fusion_module(train_feat, trans_train_feat)
        test_feat = self.fusion_module(test_feat, trans_test_feat)
        # print("only trans")
        # train_feat = self.fusion_module(trans_train_feat, trans_train_feat)
        # test_feat = self.fusion_module(trans_test_feat, trans_test_feat)

        test_feat_seq = test_feat.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1) # Nf_te*H*W, Ns, C
        train_feat_seq = train_feat.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1) # Nf_tr*H*W, Ns, C
        train_label_seq = train_label.permute(1, 0, 2, 3).flatten(1).permute(1, 0).unsqueeze(2) # Nf_tr*H*W,Ns,1
        train_ltrb_target_seq_T = train_ltrb_target.permute(1, 2, 0, 3, 4).flatten(2) # Ns,4,Nf_tr*H*W

        test_pos = test_pos.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)
        train_pos = train_pos.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)

        fg_token = self.query_embed_fg.weight.reshape(1, 1, -1)
        train_label_enc = fg_token * train_label_seq

        # printv('train_ltrb_target_seq_T', train_ltrb_target_seq_T)
        train_ltrb_target_enc = self.box_encoding(train_ltrb_target_seq_T).permute(2,0,1) # Nf_tr*H*H,Ns,C
        #
        # printv('train_feat_seq', train_feat_seq)
        # printv('train_label_enc', train_label_enc)
        # printv('train_ltrb_target_enc', train_ltrb_target_enc)
        # printv('test_feat_seq', test_feat_seq)

        if self.use_test_frame_encoding:
            test_token = self.query_embed_test.weight.reshape(1, 1, -1)
            test_label_enc = torch.ones_like(test_feat_seq) * test_token
            feat = torch.cat([train_feat_seq + train_label_enc + train_ltrb_target_enc, test_feat_seq + test_label_enc], dim=0)
        else:
            feat = torch.cat([train_feat_seq + train_label_enc + train_ltrb_target_enc, test_feat_seq], dim=0)

        pos = torch.cat([train_pos, test_pos], dim=0)

        output_embed, enc_mem = self.transformer(feat, mask=None, query_embed=self.query_embed_fg_decoder.weight, pos_embed=pos)

        enc_opt = enc_mem[-h*w:].transpose(0, 1)
        dec_opt = output_embed.squeeze(0).transpose(1, 2)

        return dec_opt.reshape(test_feat.shape[1], -1, 1, 1), enc_opt.permute(0, 2, 1).reshape(test_feat.shape)

    def predict_cls_bbreg_filters_parallel(self, train_feat, test_feat, train_label, num_gth_frames, train_ltrb_target, *args, **kwargs):
        """
        Note: This method does not expect 'trans_train_feat, trans_test_feat' as params because it expect train_feat, &
        test_feat to be already fused features
        """
        # train_label size guess: Nf_tr, Ns, H, W.
        # classifier and regressor both heads require differently attended transformer outputs. For Regression the
        # ground truth features are masked
        if train_feat.dim() == 4:
            train_feat = train_feat.unsqueeze(1)
        if test_feat.dim() == 4:
            test_feat = test_feat.unsqueeze(1)
        # if trans_train_feat.dim() == 4:
        #     trans_train_feat = trans_train_feat.unsqueeze(1)
        # if trans_test_feat.dim() == 4:
        #     trans_test_feat = trans_test_feat.unsqueeze(1)
        if train_ltrb_target.dim() == 4:
            train_ltrb_target = train_ltrb_target.unsqueeze(1)

        h, w = test_feat.shape[-2:]
        H, W = train_feat.shape[-2:]

        # train_feat = self.fusion_module(train_feat, trans_train_feat)
        # test_feat = self.fusion_module(test_feat, trans_test_feat)

        train_feat_stack = torch.cat([train_feat, train_feat], dim=1)
        test_feat_stack = torch.cat([test_feat, test_feat], dim=1)
        train_label_stack = torch.cat([train_label, train_label], dim=1)
        train_ltrb_target_stack = torch.cat([train_ltrb_target, train_ltrb_target], dim=1)

        test_pos = self.get_positional_encoding(test_feat)  # Nf_te, Ns, C, H, W
        train_pos = self.get_positional_encoding(train_feat)  # Nf_tr, Ns, C, H, W

        test_feat_seq = test_feat_stack.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)  # Nf_te*H*W, Ns, C
        train_feat_seq = train_feat_stack.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)  # Nf_tr*H*W, Ns, C
        train_label_seq = train_label_stack.permute(1, 0, 2, 3).flatten(1).permute(1, 0).unsqueeze(2)  # Nf_tr*H*W,Ns,1
        train_ltrb_target_seq_T = train_ltrb_target_stack.permute(1, 2, 0, 3, 4).flatten(2)  # Ns,4,Nf_tr*H*W

        test_pos = test_pos.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)
        train_pos = train_pos.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)

        fg_token = self.query_embed_fg.weight.reshape(1, 1, -1)
        train_label_enc = fg_token * train_label_seq

        train_ltrb_target_enc = self.box_encoding(train_ltrb_target_seq_T).permute(2, 0, 1)  # Nf_tr*H*H,Ns,C

        if self.use_test_frame_encoding:
            test_token = self.query_embed_test.weight.reshape(1, 1, -1)
            test_label_enc = torch.ones_like(test_feat_seq) * test_token
            feat = torch.cat([train_feat_seq + train_label_enc + train_ltrb_target_enc, test_feat_seq + test_label_enc], dim=0)
        else:
            feat = torch.cat([train_feat_seq + train_label_enc + train_ltrb_target_enc, test_feat_seq], dim=0)

        pos = torch.cat([train_pos, test_pos], dim=0)

        # in the other batch (which is duplicated above), the mask prevents attention on some embeddings
        # this mask prevents attention on the ground truth frame in train images
        src_key_padding_mask = torch.zeros(feat.shape[1], feat.shape[0]).bool()
        src_key_padding_mask[1, num_gth_frames*H*W:-h*w] = 1.
        src_key_padding_mask = src_key_padding_mask.bool().to(feat.device)

        output_embed, enc_mem = self.transformer(feat, mask=src_key_padding_mask,
                                                 query_embed=self.query_embed_fg_decoder.weight,
                                                 pos_embed=pos)

        enc_opt = enc_mem[-h * w:].transpose(0, 1).permute(0, 2, 1).reshape(test_feat_stack.shape)
        dec_opt = output_embed.squeeze(0).transpose(1, 2).reshape(test_feat_stack.shape[1], -1, 1, 1)

        cls_enc_opt = enc_opt[:, 0].unsqueeze(1)
        bbreg_enc_opt = enc_opt[:, 1].unsqueeze(1)
        cls_dec_opt = dec_opt[0].unsqueeze(0)
        bbreg_dec_opt = dec_opt[1].unsqueeze(0)

        return cls_dec_opt, bbreg_dec_opt, cls_enc_opt, bbreg_enc_opt
