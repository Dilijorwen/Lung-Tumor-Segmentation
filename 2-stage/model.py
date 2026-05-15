import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
    ) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = False,
    ) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels,
                in_channels // 2,
                kernel_size=2,
                stride=2,
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_x != 0 or diff_y != 0:
            x = F.pad(
                x,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ],
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        bilinear: bool = False,
    ) -> None:
        super().__init__()
        factor = 2 if bilinear else 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        self.outc = OutConv(base_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def _as_none(value):
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return value


def _as_int_tuple(value, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",") if part.strip()]
    return tuple(int(item) for item in value)


def normalize_smp_architecture(config: dict) -> str:
    aliases = {
        "smpunet": "unet",
        "smp_unet": "unet",
        "library_unet": "unet",
        "unet": "unet",
        "smpattentionunet": "unet_attention",
        "smp_attention_unet": "unet_attention",
        "attentionunet": "unet_attention",
        "attention_unet": "unet_attention",
        "unet_attention": "unet_attention",
        "unetattention": "unet_attention",
        "smpunetplusplus": "unet++",
        "smp_unetplusplus": "unet++",
        "smp_unet_plus_plus": "unet++",
        "unetplusplus": "unet++",
        "unet_plus_plus": "unet++",
        "unet++": "unet++",
    }

    def normalize_value(raw_value):
        value = str(raw_value).strip().lower().replace("-", "_").replace(" ", "_")
        return aliases.get(value)

    architecture = normalize_value(config["architecture"]) if "architecture" in config else None
    name = normalize_value(config.get("name", "SMPUnet"))

    if architecture is not None:
        return architecture
    if name is not None:
        return name

    raw_value = config.get("architecture", config.get("name", "SMPUnet"))
    raise ValueError(f"Unsupported SMP architecture: {raw_value!r}")


def model_output_name(config: dict) -> str:
    name = str(config.get("name", "")).strip().lower()
    if name in {"unet2d", "legacy_unet2d", "custom_unet"}:
        return "unet2d"
    return normalize_smp_architecture(config)


def _build_smp_unet(config: dict) -> nn.Module:
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            "segmentation_models_pytorch is required for SMP U-Net models. "
            "Install it with: pip install segmentation-models-pytorch"
        ) from exc

    architecture = normalize_smp_architecture(config)
    encoder_depth = int(config.get("encoder_depth", 5))
    decoder_channels = _as_int_tuple(
        config.get("decoder_channels"),
        default=(256, 128, 64, 32, 16),
    )
    if len(decoder_channels) != encoder_depth:
        raise ValueError(
            "decoder_channels length must match encoder_depth: "
            f"got {len(decoder_channels)} channels for depth {encoder_depth}"
        )

    decoder_attention_type = _as_none(config.get("decoder_attention_type", None))
    if architecture == "unet_attention" and decoder_attention_type is None:
        decoder_attention_type = "scse"

    model_cls = smp.UnetPlusPlus if architecture == "unet++" else smp.Unet
    return model_cls(
        encoder_name=str(config.get("encoder_name", "resnet34")),
        encoder_depth=encoder_depth,
        encoder_weights=_as_none(config.get("encoder_weights", None)),
        decoder_use_batchnorm=bool(config.get("decoder_use_batchnorm", True)),
        decoder_channels=decoder_channels,
        decoder_attention_type=decoder_attention_type,
        in_channels=int(config.get("in_channels", 1)),
        classes=int(config.get("out_channels", config.get("classes", 1))),
        activation=None,
    )


def _build_legacy_unet(config: dict) -> UNet2D:
    return UNet2D(
        in_channels=int(config.get("in_channels", 1)),
        out_channels=int(config.get("out_channels", 1)),
        base_channels=int(config.get("base_channels", 32)),
        bilinear=bool(config.get("bilinear", False)),
    )


def build_model(config: dict) -> nn.Module:
    name = str(config.get("name", "SMPUnet")).lower()
    library = str(config.get("library", "")).lower()

    if name in {"unet2d", "legacy_unet2d", "custom_unet"}:
        return _build_legacy_unet(config)

    if library in {"segmentation_models_pytorch", "smp"} or name in {
        "smpunet",
        "smp_unet",
        "smpunetplusplus",
        "smp_unetplusplus",
        "smpattentionunet",
        "smp_attention_unet",
        "library_unet",
        "library_unetplusplus",
        "library_attention_unet",
        "unet",
        "unet_attention",
        "attention_unet",
        "unet++",
        "unetplusplus",
        "unet_plus_plus",
    }:
        return _build_smp_unet(config)

    raise ValueError(
        "Unsupported model config. Use name=SMPUnet for library U-Net, "
        "name=SMPUnetPlusPlus for library U-Net++, "
        "name=SMPAttentionUnet for attention U-Net, or name=UNet2D for legacy checkpoints."
    )
