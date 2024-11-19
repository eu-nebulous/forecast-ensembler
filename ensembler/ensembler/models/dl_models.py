"""Script for Ensembler torch network"""

import torch
import torch.nn as nn
from dl_helpers import (attention_module, base_conv_layer, base_linear_module,
                        calculate_matching_padding, masked_softmax)


class FCNN(nn.Module):
    """Torch ensemble network"""

    def __init__(self, forecasters_num=6, n_inputs=[11, 32, 64], series_len=21):
        super().__init__()
        self.forecasters_num = forecasters_num
        n_inputs[0] = n_inputs[0] * series_len
        layers = [
            base_linear_module(in_f, out_f)
            for in_f, out_f in zip(n_inputs[:-1], n_inputs[1:])
        ] + [nn.Linear(n_inputs[-1], forecasters_num)]
        self.fcnn = nn.Sequential(*layers)

    def get_mask_and_preds(self, prediction):
        """Get binary mask for predictions which
        are not present and predicted values"""
        return (
            prediction[:, self.forecasters_num :],
            prediction[:, : self.forecasters_num],
        )

    def forward(self, batch):
        """model forward function"""
        x, preds_to_ensemble = batch
        x = torch.flatten(x, start_dim=1)
        x = self.fcnn(x)
        mask, preds = self.get_mask_and_preds(preds_to_ensemble)
        x = masked_softmax(x, mask)
        return torch.sum(preds * x, 1)


class EnsemblerRegressorModel(nn.Module):
    """Torch ensemble network"""

    def __init__(
        self,
        forecasters_num=6,
        n_inputs=[11, 32, 64],
        n_outputs=[32, 64, 128],
        kernel_size=[5, 5, 3],
        stride=[1, 1, 1],
        dilation=[1, 1, 1],
        series_len=21,
        with_att=False,
    ):
        super().__init__()
        self.forecasters_num = forecasters_num
        self.n_layers = len(dilation)
        self.with_att = with_att
        self.series_len = series_len
        self.cnn_layers = nn.ModuleList(
            self.create_cnn_layers(n_inputs, n_outputs, kernel_size, stride, dilation)
        )
        self.att_layers = attention_module(embed_dim=n_outputs[-1])
        self.activation_layers = nn.GELU()
        self.last_layer = nn.Linear(
            int(series_len * n_outputs[-1]) + self.forecasters_num * 2,
            self.forecasters_num,
        )

    def create_cnn_layers(self, n_inputs, n_outputs, kernel_size, stride, dilation):
        """Creates convolutional layers with padding
        so that the output has the same length as input"""
        return [
            base_conv_layer(
                in_f,
                out_f,
                k,
                stride=s,
                padding=calculate_matching_padding(
                    self.series_len, self.series_len, k, d, s
                ),
                dilation=d,
            )
            for in_f, out_f, k, s, d in zip(
                n_inputs, n_outputs, kernel_size, stride, dilation
            )
        ]

    def get_mask_and_preds(self, prediction):
        """Get binary mask for predictions which
        are not present and predicted values"""
        return (
            prediction[:, self.forecasters_num :],
            prediction[:, : self.forecasters_num],
        )

    def forward(self, batch):
        """model forward function"""
        x, preds_to_ensemble = batch
        for i in range(self.n_layers):
            x = x.permute(0, 2, 1)
            x = self.cnn_layers[i](x)
            x = x.permute(0, 2, 1)
            if i == self.n_layers - 1:
                if self.with_att:
                    x, _ = self.att_layers(x, x, x)
                    x = self.activation_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x, preds_to_ensemble), 1)
        x = self.last_layer(x)
        mask, preds = self.get_mask_and_preds(preds_to_ensemble)
        x = masked_softmax(x, mask)
        return torch.sum(preds * x, 1)


class BaseCNNNetworkMask(nn.Module):
    """Dense CNN 1D network with softmax at the end"""

    def __init__(self, num_feat=6, hist_len=5, pred_len=1, extra_feat=3):
        super(BaseCNNNetworkMask, self).__init__()
        self.extra_feat = extra_feat
        self.num_feat = num_feat
        self.hist_len = hist_len
        self.pred_len = pred_len
        n_inputs = [
            self.num_feat,
            16,
        ]
        self.forecasters_num = 6
        n_outputs = [16, 32]
        kernel_size = [4, 4]
        stride = [1, 1]
        padding = [1, 2]
        dilation = [1, 2]
        self.n_layers = len(dilation)
        layers = [
            base_conv_layer(in_f, out_f, k, stride=s, padding=p, dilation=d)
            for in_f, out_f, k, s, p, d in zip(
                n_inputs, n_outputs, kernel_size, stride, padding, dilation
            )
        ]
        self.cnn_layers = nn.ModuleList(layers)
        self.att_layers = nn.ModuleList(
            [[attention_module(embed_dim=e_dim) for e_dim in n_outputs][-1]]
        )
        self.activation_layers = nn.ModuleList(
            [nn.GELU() for _ in range(self.n_layers)]
        )
        self.last_layer = nn.Sequential(
            nn.Linear(268, 128),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, self.forecasters_num),
        )

    def get_mask_and_preds(self, prediction):
        """Get binary mask for predictions which
        are not present and predicted values"""
        return (
            prediction[:, self.forecasters_num :],
            prediction[:, : self.forecasters_num],
        )

    def forward(self, batch):
        x, preds_to_ensemble = batch
        org_x = preds_to_ensemble
        for i in range(self.n_layers):
            x = x.permute(0, 2, 1)
            x = self.cnn_layers[i](x)
            x = x.permute(0, 2, 1)
            if i >= self.n_layers:
                x, _ = self.att_layers[0](x, x, x)
                x = self.activation_layers[i](x)

        x = torch.flatten(x, start_dim=1)
        x = torch.cat([x, org_x.squeeze(1)], 1)
        x = self.last_layer(x)
        mask, preds = self.get_mask_and_preds(preds_to_ensemble)
        x = masked_softmax(x, mask)
        return torch.sum(preds * x, 1)
