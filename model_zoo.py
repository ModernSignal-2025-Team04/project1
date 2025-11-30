from unet import UNet2D, NestedUNet
from nnunet import nnUNet
from transunet import TransUNet

def get_model(model_name, in_ch, num_classes):
    if model_name == 'UNet2D':
        return UNet2D(in_ch=in_ch, num_classes=num_classes)
    elif model_name == 'NestedUNet':
        return NestedUNet(deepsupervision=False, in_channel=in_ch, out_channel=num_classes)
    elif model_name == 'nnUNet':
        return nnUNet(in_channels=in_ch, num_classes=num_classes)
    elif model_name == 'TransUNet':
        params = {
            'img_size': 256,
            'class_num': num_classes,
            'zero_head': False,
            'vis': False
        }
        return TransUNet(params)
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    
if __name__ == "__main__":
    # Example usage
    import torch
    from torchsummary import summary
    # model = get_model('UNet2D', in_ch=1, num_classes=1).cuda()
    # model = get_model('nnUNet', in_ch=1, num_classes=1).cuda()
    model = get_model('TransUNet', in_ch=1, num_classes=1).cuda()
    print(model(torch.randn(1, 1, 256, 256).cuda()).shape)
    # summary(model, (1, 256, 256))