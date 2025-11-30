import os
import sys
import imageio.v2 as imageio
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset_faz import FazSegDataset
from cut_utils import ResnetGenerator 
from cut_utils import tensor2im


def generate_images():
    """Generate images using CUT model
    """
    SAVE_DIR = os.path.join(os.path.realpath(os.path.dirname(sys.argv[0])), f'cut_generated')
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model_path = os.path.join(os.path.realpath(os.path.dirname(sys.argv[0])), f'cut_results/CUT7.pt')
    model_dict = torch.load(model_path)
    G = ResnetGenerator().to(device)
    optimizer_G = optim.Adam(G.parameters(), lr=2e-4)
    G.load_state_dict(model_dict['model_state_dict'])
    optimizer_G.load_state_dict(model_dict['optimizer_state_dict'])

    # load data
    dataset = FazSegDataset(1, split="train", aug="none")
    src_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # inference
    G.eval()
    with torch.no_grad():
        for idx, (X, _) in tqdm(enumerate(src_loader)):
            X = X.to(device)
            fake_Y = G(X, None)
            im_fake_Y = tensor2im(fake_Y)
            save_path = os.path.join(SAVE_DIR, f"domain1_to_3_cut_{idx}.png")
            imageio.imwrite(save_path, im_fake_Y.squeeze())
    print("Saved to:", SAVE_DIR)

if __name__ == "__main__":
    generate_images()