import torch.nn as nn

from collections import OrderedDict
from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.modules.decoder_block import DeepLabHead, ASPPModule
from lib.models.modules.projection import ProjectionHead
from lib.models.tools.module_helper import ModuleHelper


class DeepLabV3Contrast(nn.Module):
    def __init__(self, configer):
        super(DeepLabV3Contrast, self).__init__()

        self.num_classes = configer.get('data', 'num_classes')
        self.bn_type = configer.get('network', 'bn_type')
        self.proj_dim = configer.get('contrast', 'proj_dim')

        # extra added layers
        self.backbone = BackboneSelector(configer).get_backbone()
        if "wide_resnet38" in configer.get('network', 'backbone'):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]

        self.proj_head = ProjectionHead(dim_in=in_channels[1], proj_dim=self.proj_dim, bn_type=self.bn_type)
        self.layer_dsn = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
                                       ModuleHelper.BNReLU(256, bn_type=self.bn_type),
                                       nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True))
        self.layer_aspp = ASPPModule(2048, 512, bn_type=self.bn_type)

        self.classifier = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=False)),
            ('bn1', ModuleHelper.BatchNorm2d(bn_type=self.bn_type)(512)),
            ('cls', nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, bias=True)),
        ]))

        for modules in [self.proj_head, self.layer_dsn, self.layer_aspp, self.classifier]:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, x_, with_embed=False, is_eval=False):
        x = self.backbone(x_)    # return intermediate layers
        embedding = self.proj_head(x[-1])

        # auxiliary supervision
        x_dsn = self.layer_dsn(x[-2])
        # aspp module
        x_aspp = self.layer_aspp(x[-1])
        # refine module
        x_seg = self.classifier(x_aspp)

        return {'embed': embedding, 'seg_aux': x_dsn, 'seg': x_seg, 'h': x_aspp}

class DeepLabV3(nn.Module):
    def __init__(self, configer):
        super(DeepLabV3, self).__init__()

        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        self.decoder = DeepLabHead(num_classes=self.num_classes, bn_type=self.configer.get('network', 'bn_type'))

        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x_):
        x = self.backbone(x_)

        x = self.decoder(x[-4:])

        return x[1], x[0]
