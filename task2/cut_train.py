import os
import sys
import time
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset_faz import FazSegDataset
from cut_utils import ResnetGenerator, PatchGAN, ProjectionHead, PatchNCELoss
from cut_utils import initialize_model_dict, epoch_time, set_requires_grad

def train(
    G, H, D, optimizer_G, optimizer_H, optimizer_D,
    criterion_GAN, criterion_NCE,
    layers_nce,
    src_loader, tgt_loader, device, epoch, log_interval
):
    """Train network for one epoch
    """
    G.train()
    H.train()
    D.train()
    # print(D)
    loss = 0.
    pbar = tqdm(enumerate(zip(src_loader, tgt_loader)), position=0, desc="Training", total=min(len(src_loader), len(tgt_loader)))
    for step, ((X, _), (Y, _)) in pbar:
        B, C, h, w = X.size()
        patch_size = 16
        n_patch = B * (h // patch_size) * (w // patch_size)
        value_real = 1.
        value_fake = 0.
        label_real = torch.full(
            (n_patch, 1, patch_size, patch_size),
            value_real, dtype=torch.float, device=device
        )
        label_fake = torch.full(
            (n_patch, 1, patch_size, patch_size),
            value_fake, dtype=torch.float, device=device
        )

        # Update Discriminator D
        set_requires_grad(D, True)
        real_X = X.to(device)
        fake_Y, encoded_real_X = G(real_X, layers_nce)
        real_Y = Y.to(device)
        score_real = D(real_Y)
        score_fake = D(fake_Y.detach())

        # Least square GAN loss
        # V(D) = 0.5 * E_x[(D(x) - 1)^2] + 0.5 * E_z[(D(G(z)))^2]
        loss_GAN_D = 0.5 * criterion_GAN(score_real, label_real) ** 2
        loss_GAN_D += 0.5 * criterion_GAN(score_fake, label_fake) ** 2
        loss_D = loss_GAN_D
        # update parameters in D
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # Update Generator G
        set_requires_grad(D, False)

        encoded_fake_Y = G(fake_Y, layers_nce, encode_only=True)
        fake_X, encoded_real_Y = G(real_Y, layers_nce)
        encoded_fake_X = G(fake_X, layers_nce, encode_only=True)
        # Projections for NCE calculation
        projections_real_X, i_X = H(encoded_real_X)
        projections_fake_Y, _ = H(encoded_fake_Y, idx_patch=i_X)
        projections_real_Y, i_Y  = H(encoded_real_Y)
        projections_fake_X, _ = H(encoded_fake_X, idx_patch=i_Y)

        # Least square GAN loss
        # V(G) = 0.5 * E_z[(D(G(x)) - 1)^2]
        score_fake = D(fake_Y)
        loss_GAN_G = 0.5 * criterion_GAN(score_fake, label_real)
        # Weighting for NCE loss
        lambda_X = 1
        lambda_Y = 1
        # NCE Loss: L_NCE(G, H, X)
        NCE_X = [
            criterion(proj_X, proj_Y, B).mean()
            for proj_X, proj_Y, criterion in zip(
                projections_real_X, projections_fake_Y, criterion_NCE
            )
        ]
        # Identity Loss: L_NCE(G, H, Y)
        NCE_Y = [
            criterion(proj_Y, proj_X, B).mean()
            for proj_Y, proj_X, criterion in zip(
                projections_real_Y, projections_fake_X, criterion_NCE
            )
        ]
        # Total loss for G: L_GAN + L_NCE
        n_layer = len(layers_nce)
        NCE = (lambda_X * sum(NCE_X) + lambda_Y * sum(NCE_Y)) / n_layer
        loss_G = loss_GAN_G + NCE
        # update parameters in G and H
        optimizer_G.zero_grad()
        optimizer_H.zero_grad()
        loss_G.backward()
        optimizer_G.step()
        optimizer_H.step()

        loss = loss_D + loss_G
        pbar.set_postfix(loss=loss.item())
        pbar.set_description(f'Epoch [{epoch}]')
        # log progress
        # if (step+1) % log_interval == 0:
        #     print(
        #         'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #             epoch, (step+1) * B, min(len(src_loader.dataset), len(tgt_loader.dataset)),
        #             100. * (step+1) / min(len(src_loader), len(tgt_loader)), loss.item()
        #         )
        #     )
        # break
        # im_fake_B = visualize(G)
        # writer.add_image('Fake Dog', im_fake_B)
    return loss.item(), loss_G.item(), loss_D.item()


# Parameter settings
train_from_scratch = True
n_epoch = 7
learning_rate = 2e-4
log_interval = 1
if not train_from_scratch:
    # Load model checkpoint
    model_path = os.path.join(os.path.realpath(os.path.dirname(sys.argv[0])), f'cut_results/CUT{n_epoch}.pt')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model_dict = torch.load(model_path)
else:
    model_dict = initialize_model_dict(n_epoch)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and data loaders
print('Initialize model...', end='')
G = ResnetGenerator().to(device)
layers_nce = [0, 2, 3, 4, 8]
X = torch.rand((1, 1, 256, 256)).to(device)
encoded_X = G(X, layers_nce, encode_only=True)
H = ProjectionHead().to(device)
_, _ = H(encoded_X)
D = PatchGAN().to(device)
optimizer_G = optim.Adam(G.parameters(), learning_rate)
optimizer_H = optim.Adam(H.parameters(), learning_rate)
optimizer_D = optim.Adam(D.parameters(), learning_rate)
if not train_from_scratch:
    G.load_state_dict(model_dict['model_state_dict'])
    optimizer_G.load_state_dict(model_dict['optimizer_state_dict'])
# datasets
src = FazSegDataset(1, "train", aug="none")  # domain1
tgt = FazSegDataset(3, "train", aug="none")  # domain3
src_loader = DataLoader(src, batch_size=4, shuffle=True)
tgt_loader = DataLoader(tgt, batch_size=4, shuffle=True)
# initialize loss functions
criterion_GAN = nn.MSELoss()
criterion_NCE = [PatchNCELoss().to(device) for layer in layers_nce]
print('Done')

# Training Loop
start_epoch = 1 
model_name = f'CUT{n_epoch}.pt'
pbar = tqdm(range(start_epoch, n_epoch+1), position=1, desc="Overall Training")
for epoch in pbar:
    start_time = time.time()
    train_loss, train_loss_G, train_loss_D = train(
        G, H, D, optimizer_G, optimizer_H, optimizer_D,
        criterion_GAN, criterion_NCE, layers_nce,
        src_loader, tgt_loader, device, epoch, log_interval
    )
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    pbar.set_description(f'Epoch [{epoch}/{n_epoch}], Time: {epoch_mins}m {epoch_secs}s')
    pbar.set_postfix(loss = train_loss)
    # log results to model dictionary
    model_dict['train_loss']['G'].append(train_loss)
    model_dict['train_loss']['D'].append(train_loss_G)
    model_dict['train_loss']['total'].append(train_loss_D)
    model_dict['metrics']['last']['loss'] = train_loss
    model_dict['metrics']['last']['epoch'] = epoch
    if epoch == 1 or train_loss < model_dict['metrics']['best']['loss']:
        model_dict['model_state_dict'] = G.state_dict()
        model_dict['optimizer_state_dict'] = optimizer_G.state_dict()
        model_dict['metrics']['best']['epoch'] = epoch
        model_dict['metrics']['best']['loss'] = train_loss
        torch.save(model_dict, model_name)