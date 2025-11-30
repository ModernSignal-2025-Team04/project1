# eval_task3.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from dataset_faz import FazSegDataset
from unet import UNet2D
from metrics import dice_coeff, hd95, assd


# CKPT = "checkpoints/task3_unet_augmix.pth"   # Task3 的模型权重
CKPT = "checkpoints/task3_mean_teacher_student_best.pth"   # Task3 的模型权重


def evaluate_domain(model, domain_id, device):
    dataset = FazSegDataset(domain_id, split="test", aug="none")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    dice_list = []
    hd_list = []
    assd_list = []

    model.eval()
    for imgs, masks in tqdm(loader, desc=f"Domain {domain_id}"):
        imgs = imgs.to(device)
        masks = masks.to(device)

        with torch.no_grad():
            logits = model(imgs)
            preds = (torch.sigmoid(logits) > 0.5).float()

        dice_list.append(dice_coeff(preds.cpu(), masks.cpu()))

        pred_np = preds.cpu().numpy()[0, 0].astype(bool)
        mask_np = masks.cpu().numpy()[0, 0].astype(bool)

        hd_list.append(hd95(pred_np, mask_np))
        assd_list.append(assd(pred_np, mask_np))

    return (
        sum(dice_list) / len(dice_list),
        sum(hd_list) / len(hd_list),
        sum(assd_list) / len(assd_list),
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    model = UNet2D().to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))

    results = {}

    for dom in range(1, 6):
        dice, hd, assd_val = evaluate_domain(model, dom, device)
        results[dom] = (dice, hd, assd_val)

    print("\n===== Task3 Evaluation =====")
    for dom, (dice, hd, assd_val) in results.items():
        print(f"Domain{dom}: Dice={dice:.4f}, HD95={hd:.2f}, ASSD={assd_val:.3f}")


if __name__ == "__main__":
    main()
