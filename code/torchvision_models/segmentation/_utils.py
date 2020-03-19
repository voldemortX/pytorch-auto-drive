from collections import OrderedDict
from torch import nn
# from torch.nn import functional as F


class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier, aux_classifier=None, recon_head=None):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier
        self.recon_head = recon_head

    def forward(self, x):
        # input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        # x = F.interpolate(x, size=(513, 513), mode='bilinear', align_corners=True)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            # x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
            result["aux"] = x

        # Reconstruction
        if self.recon_head is not None:
            x = features["recon"]
            x = self.recon_head(x)
            result["recon"] = x

        return result
