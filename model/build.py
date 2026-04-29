from model import objectives
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn


class TAGPR(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size,
                                                                      args.stride_size)
        self.embed_dim = base_cfg['embed_dim']
        self.logit_scale = torch.ones([]) * (1 / args.temperature)


        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)


    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')


    def encode_image(self, image):
        image_feats = self.base_model.encode_image(image)
        return image_feats[:, 0, :].float()

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, batch, alpha):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        cam_id = batch['camid']

        image_feats, text_feats = self.base_model(images, caption_ids)

        i_feats = image_feats[:, 0, :].float()
        # local_feats = image_feats[:, 1:, :].float()

        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if 'TAL' in self.current_task:
            ret.update({'TAL_loss': objectives.compute_TAL(i_feats, t_feats, batch['pids'], self.args.tau, self.args.margin)})

        if 'nitc' in self.current_task:
            nitc_loss = objectives.compute_nitc(i_feats, t_feats, batch['pids'], alpha, logit_scale)

            nitc_loss_all = nitc_loss * self.args.nitc_loss_weight

            ret.update({'nitc_loss': nitc_loss_all})

        if 'ritc' in self.current_task:
            ritc_loss = objectives.compute_ritc(i_feats, t_feats, batch['pids'], self.args.eps, logit_scale)

            ritc_loss_all = ritc_loss * self.args.ritc_loss_weight

            ret.update({'ritc_loss': ritc_loss_all})

        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret.update(
                {'id_loss': objectives.compute_id(image_logits, text_logits, batch['pids']) * self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})

        return ret


def build_model(args, num_classes=11003):
    model = TAGPR(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
