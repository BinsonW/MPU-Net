from copy import deepcopy
import numpy as np
import torch
from nnunet.network_architecture.custom_modules.helperModules import Identity
from nnunet.network_architecture.generic_UNet import Upsample
from torch import nn


class BasicResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, props, stride=None):
        """
        :param in_planes:
        :param out_planes:
        :param props:
        :param override_stride:
        """
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 3
        self.kernel_size = kernel_size
        props["conv_op_kwargs"]["stride"] = 1
        self.stride = stride
        self.props = props
        self.out_planes = out_planes
        self.in_planes = in_planes
        if stride is not None:
            kwargs_conv1 = deepcopy(props["conv_op_kwargs"])
            kwargs_conv1["stride"] = stride
        else:
            kwargs_conv1 = props["conv_op_kwargs"]
        self.conv1 = props["conv_op"](in_planes,
                                      out_planes,
                                      kernel_size,
                                      padding=[(i - 1) // 2
                                               for i in kernel_size],
                                      **kwargs_conv1)
        self.norm1 = props["norm_op"](out_planes, **props["norm_op_kwargs"])
        self.nonlin1 = props["nonlin"](**props["nonlin_kwargs"])
        # No dropout
        if props["dropout_op_kwargs"]["p"] != 0:
            self.dropout = props["dropout_op"](**props["dropout_op_kwargs"])
        else:
            self.dropout = Identity()
        self.conv2 = props["conv_op"](out_planes,
                                      out_planes,
                                      kernel_size,
                                      padding=[(i - 1) // 2
                                               for i in kernel_size],
                                      **props["conv_op_kwargs"])
        self.norm2 = props["norm_op"](out_planes, **props["norm_op_kwargs"])
        self.nonlin2 = props["nonlin"](**props["nonlin_kwargs"])
        if (self.stride is not None and any(
            (i != 1 for i in self.stride))) or (in_planes != out_planes):
            stride_here = stride if stride is not None else 1
            self.downsample_skip = nn.Sequential(
                props["conv_op"](in_planes,
                                 out_planes,
                                 1,
                                 stride_here,
                                 bias=False),
                props["norm_op"](out_planes, **props["norm_op_kwargs"]),
            )
        else:
            self.downsample_skip = lambda x: x

    def forward(self, x):
        residual = x
        out = self.dropout(self.conv1(x))
        out = self.nonlin1(self.norm1(out))
        out = self.norm2(self.conv2(out))
        residual = self.downsample_skip(residual)
        out += residual
        return self.nonlin2(out)


class ResidualLayer(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        network_props,
        num_blocks,
        first_stride=None,
        block=BasicResidualBlock,
    ):
        super().__init__()
        network_props = deepcopy(network_props)
        self.convs = nn.Sequential(
            block(
                input_channels,
                output_channels,
                kernel_size,
                network_props,
                first_stride,
            ), *[
                block(output_channels, output_channels, kernel_size,
                      network_props) for _ in range(num_blocks - 1)
            ])

    def forward(self, x):
        return self.convs(x)


class encoder_groupconv(nn.Module):
    def __init__(
        self,
        input_channels,
        base_num_features,
        num_blocks_per_stage,
        feat_map_mul_on_downscale,
        pool_op_kernel_sizes,
        conv_kernel_sizes,
        props,
        default_return_skips=True,
        max_num_features=512,
        block=BasicResidualBlock,
    ):
        super(encoder_groupconv, self).__init__()
        self.default_return_skips = default_return_skips
        self.props = props
        self.stages = []
        self.stage_output_features = []
        self.stage_pool_kernel_size = []
        self.stage_conv_op_kernel_size = []
        num_stages = len(conv_kernel_sizes)
        if not isinstance(num_blocks_per_stage, (list, tuple)):
            num_blocks_per_stage = [num_blocks_per_stage] * num_stages
        else:
            assert len(num_blocks_per_stage) == num_stages
        self.num_blocks_per_stage = num_blocks_per_stage
        self.initial_conv = props["conv_op"](input_channels,
                                             base_num_features,
                                             3,
                                             padding=1,
                                             **props["conv_op_kwargs"])
        self.initial_norm = props["norm_op"](base_num_features,
                                             **props["norm_op_kwargs"])
        self.initial_nonlin = props["nonlin"](**props["nonlin_kwargs"])
        current_input_features = base_num_features
        for stage in range(num_stages):
            current_output_features = min(
                base_num_features * feat_map_mul_on_downscale**stage,
                max_num_features)
            # current_output_features = base_num_features * feat_map_mul_on_downscale ** stage
            current_kernel_size = conv_kernel_sizes[stage]
            if stage != 0:
                current_pool_kernel_size = pool_op_kernel_sizes[stage - 1]
            else:
                current_pool_kernel_size = None
            current_stage = ResidualLayer(
                current_input_features,
                current_output_features,
                current_kernel_size,
                props,
                self.num_blocks_per_stage[stage],
                current_pool_kernel_size,
                block,
            )
            self.stages.append(current_stage)
            self.stage_output_features.append(current_output_features)
            self.stage_conv_op_kernel_size.append(current_kernel_size)
            if current_pool_kernel_size is not None:
                self.stage_pool_kernel_size.append(current_pool_kernel_size)
            current_input_features = current_output_features
        self.stages = nn.ModuleList(self.stages)

    def forward(self, x, return_skips=None):
        """
        :param x:
        :param return_skips: if none then self.default_return_skips is used
        :return:
        """
        skips = []
        x = self.initial_nonlin(self.initial_norm(self.initial_conv(x)))
        for s in self.stages:
            x = s(x)
            if self.default_return_skips:
                skips.append(x)
        if return_skips is None:
            return_skips = self.default_return_skips
        if return_skips:
            return skips
        else:
            return x


class encoder_plain(nn.Module):
    def __init__(
        self,
        previous,
        input_channels,
        base_num_features,
        num_blocks_per_stage,
        feat_map_mul_on_downscale,
        pool_op_kernel_sizes,
        conv_kernel_sizes,
        props,
        default_return_skips=True,
        max_num_features=512,
        block=BasicResidualBlock,
    ):
        """
        :param input_channels:
        :param base_num_features:
        :param num_blocks_per_stage:
        :param feat_map_mul_on_downscale:
        :param pool_op_kernel_sizes:
        :param conv_kernel_sizes:
        :param props:
        """
        super(encoder_plain, self).__init__()
        self.default_return_skips = default_return_skips
        self.props = props
        self.stages = []
        self.stage_output_features = []
        self.stage_pool_kernel_size = []
        self.stage_conv_op_kernel_size = []
        num_stages = len(conv_kernel_sizes)
        if not isinstance(num_blocks_per_stage, (list, tuple)):
            num_blocks_per_stage = [num_blocks_per_stage] * num_stages
        else:
            assert len(num_blocks_per_stage) == num_stages
        self.num_blocks_per_stage = num_blocks_per_stage
        self.initial_conv = props["conv_op"](input_channels,
                                             base_num_features,
                                             3,
                                             padding=1,
                                             **props["conv_op_kwargs"])
        self.initial_norm = props["norm_op"](base_num_features,
                                             **props["norm_op_kwargs"])
        self.initial_nonlin = props["nonlin"](**props["nonlin_kwargs"])
        in_features = [16, 32, 64, 128, 256, 512, 640]
        out_features = [16, 32, 64, 128, 256, 320, 320]
        out_features = previous.stage_output_features
        in_features = []
        in_features.append(base_num_features)
        for i in range(len(out_features) - 1):
            in_features.append(out_features[i])
        for stage in range(num_stages):
            current_kernel_size = conv_kernel_sizes[stage]
            if stage != 0:
                current_pool_kernel_size = pool_op_kernel_sizes[stage - 1]
            else:
                current_pool_kernel_size = None
            current_stage = ResidualLayer(
                in_features[stage],
                out_features[stage],
                current_kernel_size,
                props,
                self.num_blocks_per_stage[stage],
                current_pool_kernel_size,
                block,
            )
            self.stages.append(current_stage)
            self.stage_output_features.append(
                previous.stage_output_features[stage])
            self.stage_conv_op_kernel_size.append(current_kernel_size)
            if current_pool_kernel_size is not None:
                self.stage_pool_kernel_size.append(current_pool_kernel_size)
        self.stages = nn.ModuleList(self.stages)

    def forward(self, x, skips_dw, return_skips=None):
        """
        :param x:
        :param return_skips: if none then self.default_return_skips is used
        :return:
        """
        skips = []
        x = self.initial_nonlin(self.initial_norm(self.initial_conv(x)))
        for i, s in enumerate(self.stages):
            x = s(x)
            if self.default_return_skips:
                skips.append(x)
        if return_skips is None:
            return_skips = self.default_return_skips
        if return_skips:
            return skips
        else:
            return x


class decoder_groupconv(nn.Module):
    def __init__(
        self,
        previous,
        num_classes,
        num_blocks_per_stage=None,
        network_props=None,
        deep_supervision=False,
        upscale_logits=False,
        block=BasicResidualBlock,
    ):
        super(decoder_groupconv, self).__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        previous_stages = previous.stages
        previous_stage_output_features = previous.stage_output_features
        previous_stage_pool_kernel_size = previous.stage_pool_kernel_size
        previous_stage_conv_op_kernel_size = previous.stage_conv_op_kernel_size
        if network_props is None:
            self.props = previous.props
        else:
            self.props = network_props
        if self.props["conv_op"] == nn.Conv2d:
            transpconv = nn.ConvTranspose2d
            upsample_mode = "bilinear"
        elif self.props["conv_op"] == nn.Conv3d:
            transpconv = nn.ConvTranspose3d
            upsample_mode = "trilinear"
        else:
            raise ValueError(
                "unknown convolution dimensionality, conv op: %s" %
                str(self.props["conv_op"]))
        if num_blocks_per_stage is None:
            num_blocks_per_stage = previous.num_blocks_per_stage[:-1][::-1]
        assert len(num_blocks_per_stage) == len(
            previous.num_blocks_per_stage) - 1
        self.stage_pool_kernel_size = previous_stage_pool_kernel_size
        self.stage_output_features = previous_stage_output_features
        self.stage_conv_op_kernel_size = previous_stage_conv_op_kernel_size
        num_stages = len(previous_stages) - 1
        self.tus = []
        self.stages = []
        self.deep_supervision_outputs = []
        cum_upsample = np.cumprod(np.vstack(
            [np.array(i) for i in self.stage_pool_kernel_size]),
                                  axis=0).astype(float)
        for i, s in enumerate(np.arange(num_stages)[::-1]):
            features_below = previous_stage_output_features[s + 1]
            features_skip = previous_stage_output_features[s]
            self.tus.append(
                transpconv(
                    features_below,
                    features_skip,
                    previous_stage_pool_kernel_size[s],
                    previous_stage_pool_kernel_size[s],
                    bias=False,
                ))
            self.stages.append(
                ResidualLayer(
                    2 * features_skip,
                    features_skip,
                    previous_stage_conv_op_kernel_size[s],
                    self.props,
                    num_blocks_per_stage[i],
                    None,
                    block,
                ))
            if deep_supervision:
                seg_layer1 = self.props["conv_op"](
                    features_skip * 2, features_skip, 1, 1, 0, 1, 1,
                    False)  # kernel_size = 1, 1 x 1 conv
                seg_layer2 = self.props["conv_op"](
                    features_skip, num_classes, 1, 1, 0, 1, 1,
                    False)  # kernel_size = 1, 1 x 1 conv
                if upscale_logits:
                    upsample = Upsample(scale_factor=float(cum_upsample[s]),
                                        mode=upsample_mode)
                    self.deep_supervision_outputs.append(
                        nn.Sequential(seg_layer1, seg_layer2, upsample))
                else:
                    self.deep_supervision_outputs.append(
                        nn.Sequential(seg_layer1, seg_layer2))
        self.segmentation_output = nn.Sequential(
            self.props["conv_op"](features_skip * 2, features_skip, 1, 1, 0, 1,
                                  1, False),
            self.props["conv_op"](features_skip, num_classes, 1, 1, 0, 1, 1,
                                  False),
        )
        self.tus = nn.ModuleList(self.tus)
        self.stages = nn.ModuleList(self.stages)
        self.deep_supervision_outputs = nn.ModuleList(
            self.deep_supervision_outputs)

    def forward(self, skips_enc_dw, skips_dec_plain):
        skips_enc_dw = skips_enc_dw[::-1]
        seg_outputs = []
        x = skips_enc_dw[0]
        for i in range(len(self.tus)):
            x = self.tus[i](x)
            x = torch.cat((x, skips_enc_dw[i + 1]), dim=1)
            x = self.stages[i](x)
            out = torch.cat((x, skips_dec_plain[i]), dim=1)
            if self.deep_supervision:
                seg_outputs.append(self.deep_supervision_outputs[i](out))
            if (not self.deep_supervision) and (i == len(self.tus) - 1):
                segmentation = self.deep_supervision_outputs[i](out)
        if self.deep_supervision:
            return seg_outputs[::-1]
        else:
            return segmentation


class decoder_plain(nn.Module):
    def __init__(
        self,
        previous,
        num_classes,
        num_blocks_per_stage=None,
        network_props=None,
        deep_supervision=False,
        upscale_logits=False,
        block=BasicResidualBlock,
    ):
        super(decoder_plain, self).__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        previous_stages = previous.stages
        previous_stage_output_features = previous.stage_output_features
        previous_stage_pool_kernel_size = previous.stage_pool_kernel_size
        previous_stage_conv_op_kernel_size = previous.stage_conv_op_kernel_size
        if network_props is None:
            self.props = previous.props
        else:
            self.props = network_props
        if self.props["conv_op"] == nn.Conv2d:
            transpconv = nn.ConvTranspose2d
            upsample_mode = "bilinear"
        elif self.props["conv_op"] == nn.Conv3d:
            transpconv = nn.ConvTranspose3d
            upsample_mode = "trilinear"
        else:
            raise ValueError(
                "unknown convolution dimensionality, conv op: %s" %
                str(self.props["conv_op"]))
        if num_blocks_per_stage is None:
            num_blocks_per_stage = previous.num_blocks_per_stage[:-1][::-1]
        assert len(num_blocks_per_stage) == len(
            previous.num_blocks_per_stage) - 1
        self.stage_pool_kernel_size = previous_stage_pool_kernel_size
        self.stage_output_features = previous_stage_output_features
        self.stage_conv_op_kernel_size = previous_stage_conv_op_kernel_size
        num_stages = len(previous_stages) - 1
        self.tus = []
        self.stages = []
        self.deep_supervision_outputs = []
        cum_upsample = np.cumprod(np.vstack(
            [np.array(i) for i in self.stage_pool_kernel_size]),
                                  axis=1).astype(float)
        for i, s in enumerate(np.arange(num_stages)[::-1]):
            features_below = previous_stage_output_features[s + 1]
            features_skip = previous_stage_output_features[s]
            self.tus.append(
                transpconv(
                    features_below,
                    features_skip,
                    previous_stage_pool_kernel_size[s],
                    previous_stage_pool_kernel_size[s],
                    bias=False,
                ))
            self.stages.append(
                ResidualLayer(
                    2 * features_skip,
                    features_skip,
                    previous_stage_conv_op_kernel_size[s],
                    self.props,
                    num_blocks_per_stage[i],
                    None,
                    block,
                ))
            if deep_supervision and s != 0:
                seg_layer = self.props["conv_op"](features_skip, num_classes,
                                                  1, 1, 0, 1, 1, False)
                if upscale_logits:
                    upsample = Upsample(scale_factor=float(cum_upsample[s]),
                                        mode=upsample_mode)
                    self.deep_supervision_outputs.append(
                        nn.Sequential(seg_layer, upsample))
                else:
                    self.deep_supervision_outputs.append(seg_layer)
        self.segmentation_output = self.props["conv_op"](features_skip,
                                                         num_classes, 1, 1, 0,
                                                         1, 1, False)
        self.tus = nn.ModuleList(self.tus)
        self.stages = nn.ModuleList(self.stages)
        self.deep_supervision_outputs = nn.ModuleList(
            self.deep_supervision_outputs)

    def forward(self, skips):
        skips = skips[::-1]
        skips_out = []
        x = skips[0]
        for i in range(len(self.tus)):
            x = self.tus[i](x)
            x = torch.cat((x, skips[i + 1]), dim=1)
            x = self.stages[i](x)
            skips_out.append(x)
        return skips_out


class MPUNet_single_multi_modality(nn.Module):
    def __init__(
        self,
        input_channels: int = 4,
        base_num_features:
        int = 16,  # number of output features of the initial conv
        num_blocks_per_stage_encoder: int = 2,
        feat_map_mul_on_downscale: int = 2,
        pool_op_kernel_sizes=[2, 2, 2, 2, 2],
        conv_kernel_sizes=[3, 3, 3, 3, 3],
        props: dict = None,
        num_classes: int = 3,
        num_blocks_per_stage_decoder=None,
        deep_supervision=True,
        upscale_logits=False,  # output upsampling, match the size of input
        max_features=512,
        initializer=None,
        block=BasicResidualBlock,
    ):
        super(MPUNet_single_multi_modality, self).__init__()
        self.num_classes = num_classes
        self.do_ds = deep_supervision
        self._props_plain = self._props_dw = {
            "conv_op": nn.Conv3d,
            "conv_op_kwargs":
            {},  # except for input channels, output channel, padding, kernel size
            "norm_op": nn.BatchNorm3d,
            "norm_op_kwargs": {},  # except for num features
            "nonlin": nn.LeakyReLU,
            "nonlin_kwargs": {},  # all arg
            "dropout_op": nn.Dropout3d,
            "dropout_op_kwargs": {
                "p": 0
            },
        }
        self._props_dw["conv_op_kwargs"] = {"groups": input_channels}
        self.conv_op = self._props_plain["conv_op"]
        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes,
                                                        0,
                                                        dtype=np.int64)
        self.encoder_groupconv = encoder_groupconv(
            input_channels,
            base_num_features,
            num_blocks_per_stage_encoder,
            feat_map_mul_on_downscale,
            pool_op_kernel_sizes,
            conv_kernel_sizes,
            props=self._props_dw,
            default_return_skips=True,
            max_num_features=max_features,
            block=block,
        )
        self.encoder_plain = encoder_plain(
            self.encoder_groupconv,
            input_channels,
            base_num_features,
            num_blocks_per_stage_encoder,
            feat_map_mul_on_downscale,
            pool_op_kernel_sizes,
            conv_kernel_sizes,
            props=self._props_plain,
            default_return_skips=True,
            max_num_features=max_features,
            block=block,
        )
        self.decoder_groupconv = decoder_groupconv(
            self.encoder_groupconv,
            num_classes,
            num_blocks_per_stage_decoder,
            props,
            deep_supervision,
            upscale_logits,
            block=block,
        )
        self.decoder_plain = decoder_plain(
            self.encoder_plain,
            num_classes,
            num_blocks_per_stage_decoder,
            props,
            deep_supervision,
            upscale_logits=False,
            block=block,
        )
        if initializer is not None:
            self.apply(initializer)

    def forward(self, x):
        skips_enc_dw = self.encoder_groupconv(x)
        skips_enc_plain = self.encoder_plain(x, skips_enc_dw)
        skips_dec_plain = self.decoder_plain(skips_enc_plain)
        out = self.decoder_groupconv(skips_enc_dw, skips_dec_plain)
        return out

    def do_ds_false(self):
        self.decoder_groupconv.deep_supervision = False


if __name__ == "__main__":
    a = MPUNet_single_multi_modality()
    img = torch.zeros((1, 4, 128, 128, 128), device=torch.device("cuda:0"))
    a.to(torch.device("cuda:0"))
    out = a(img)
    b = 1
