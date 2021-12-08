import torch.nn as nn
from torch import load
from collections import OrderedDict


class _EncoderDecoderModel(nn.Module):
    def __init__(self):
        super().__init__()

    def _load_encoder(self, pretrained_weights):
        if pretrained_weights is not None:  # Load pre-trained weights
            saved_weights = load(pretrained_weights)['model']
            original_weights = self.state_dict()
            for key in saved_weights.keys():
                if key in original_weights.keys():
                    original_weights[key] = saved_weights[key]
            self.load_state_dict(original_weights)
        else:
            print('No pre-training.')

    def forward(self, x):
        pass


class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier, aux_classifier=None, recon_head=None,
                 lane_classifier=None, channel_reducer=None, scnn_layer=None):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier  # Name aux is already take for COCO pre-trained models
        self.lane_classifier = lane_classifier
        self.channel_reducer = channel_reducer  # Reduce ResNet feature channels to 128 as did in RESA
        self.scnn_layer = scnn_layer
        self.recon_head = recon_head

    def forward(self, x):
        # input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)
        result = OrderedDict()
        x = features['out']

        # For lane detection
        if self.channel_reducer is not None:
            x = self.channel_reducer(x)
        if self.scnn_layer is not None:
            x = self.scnn_layer(x)

        # Semantic segmentation
        x = self.classifier(x)
        # x = F.interpolate(x, size=(513, 513), mode='bilinear', align_corners=True)
        result['out'] = x

        # For lane detection
        if self.lane_classifier is not None:
            result['lane'] = self.lane_classifier(x.softmax(dim=1))

        # For COCO pre-trained models
        if self.aux_classifier is not None:
            x = features['aux']
            x = self.aux_classifier(x)
            # x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
            result['aux'] = x

        return result
