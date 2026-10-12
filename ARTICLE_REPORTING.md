# Note on transformer pretraining

The runs are labelled `imagenet1k`, but the Swin and ViT backbones were loaded with timm's default pretrained weights, which come from larger datasets:

| Backbone | Pretrained weights actually used (timm default) |
|---|---|
| Swin-B (`swin_base_patch4_window7_224`) | ImageNet-22k pretraining, fine-tuned on ImageNet-1k (`ms_in22k_ft_in1k`) |
| ViT-B (`vit_base_patch16_224`) | ImageNet-21k pretraining, fine-tuned on ImageNet-1k (`augreg2_in21k_ft_in1k`) |
| DeiT-B (`deit_base_patch16_224`) | ImageNet-1k only (`fb_in1k`) |
| AlexNet, VGG16, GoogLeNet | ImageNet-1k only (torchvision `IMAGENET1K_V1`) |

This applies to the previous study as well. Its code loads the same timm defaults and names the Swin 22k→1k weight file in [model_utils.py:51](model_utils.py#L51).

The article should give the pretraining source for each backbone instead of "ImageNet-1k" for all of them. Comparisons between CNNs and Swin/ViT combine differences in architecture with differences in the amount of pretraining data.
