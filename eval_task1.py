# eval_task1.py
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from dataset_faz import FazSegDataset
from model_zoo import get_model
from metrics import dice_coeff, hd95, assd


# ------------------------------
# 选择要测试的模型文件
# ------------------------------
MODEL_NAME = "UNet2D"
CKPT = f"checkpoints/{MODEL_NAME}_faz_aug_none.pth"   # 用训练出来的最好那一个


def evaluate_domain(model, domain_id, device):
    """
    在某个 domain 的 test 上计算 Dice、HD95、ASSD（平均）
    """
    dataset = FazSegDataset(domain_id=domain_id, split="test", aug="none")
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

        # Dice (torch)
        dice = dice_coeff(preds.cpu(), masks.cpu())
        dice_list.append(dice)

        # HD95 / ASSD (numpy)
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
    print("Using device:", device)

    # 加载模型
    model = get_model(MODEL_NAME, in_ch=1, num_classes=1).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))

    results = {}

    # 在 domain1~5 的 test 上评估
    for dom in range(1, 6):
        dice, hd, assd_val = evaluate_domain(model, dom, device)
        results[f"domain{dom}"] = (dice, hd, assd_val)

    print("\n===== Task1 Evaluation =====")
    for dom, (dice, hd, assd_val) in results.items():
        print(f"{dom}: Dice={dice:.4f}, HD95={hd:.3f}, ASSD={assd_val:.3f}")


if __name__ == "__main__":
    main()
