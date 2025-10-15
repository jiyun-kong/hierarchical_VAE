# python train.py --epochs 1 --batch_size 16 --dataset_path /scratch2/hayeonjeong/nvae/dataset/train
import argparse
import logging
import os
import time
import yaml
import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch
import importlib
from torch.utils.data import DataLoader

from nvae.dataset import ImageFolderDataset
from nvae.utils import add_sn
# from nvae.vae_celeba_kernel3 import NVAE

from fid.fid_score import calculate_fid_given_paths

def generate_fake_images(model, z_dim, save_dir, num_images=5000, device='cuda'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i in range(num_images):
            z = torch.randn((1, z_dim, 2, 2)).to(device)  # NVAE 구조에 맞게!
            gen_img, _ = model.decoder(z)
            img = gen_img[0].cpu().permute(1,2,0).numpy()
            img = np.clip(img, 0, 1)
            plt.imsave(f"{save_dir}/gen_{i:05d}.png", img)

def calculate_fid(real_dir, fake_dir, batch_size=64, device='cuda', dims=2048):
    fid_value = calculate_fid_given_paths([real_dir, fake_dir], batch_size, device, dims)
    return fid_value

class WarmupKLLoss:

    def __init__(self, init_weights, steps,
                 M_N=0.005,
                 eta_M_N=1e-5,
                 M_N_decay_step=3000):
        """
        WarmupKLLoss : 여러 계층의 KL loss를 각각 따로, 그리고 점진적으로(warmup) 적용

        각 계층별로 KL loss의 가중치를 처음엔 작게, 일정 step마다 선형적으로 1까지 증가
        모든 계층의 warmup이 끝나면, 전체 KL loss에 곱하는 M_N 값도 점진적으로 줄여서(감쇠)
        학습 후반에 KL loss가 너무 커지지 않도록]
        """
        self.init_weights = init_weights # 각 계층 KL loss의 초기 가중치
        self.M_N = M_N # 각 계층 KL loss가 1까지 증가하는 데 걸리는 step 수
        self.eta_M_N = eta_M_N # M_N의 최소값
        self.M_N_decay_step = M_N_decay_step # M_N이 감쇠되는 step 수

        self.speeds = [(1. - w) / s for w, s in zip(init_weights, steps)]
        self.steps = np.cumsum(steps)
        self.stage = 0
        self._ready_start_step = 0
        self._ready_for_M_N = False
        self._M_N_decay_speed = (self.M_N - self.eta_M_N) / self.M_N_decay_step

    def _get_stage(self, step):
        while True:

            if self.stage > len(self.steps) - 1:
                break

            if step <= self.steps[self.stage]:
                return self.stage
            else:
                self.stage += 1

        return self.stage

    # 현재 step에 맞는 각 계층별 KL loss 가중치를 계산해서 전체 KL loss를 반환
    # 계층적 VAE는 여러 계층의 latent z를 사용하므로, 각 계층별로 KL loss를 따로 조절해줘야 학습이 안정적
    
    def get_loss(self, step, losses):
        loss = 0.
        stage = self._get_stage(step)

        for i, l in enumerate(losses):
            # Update weights
            if i == stage:
                speed = self.speeds[stage]
                t = step if stage == 0 else step - self.steps[stage - 1]
                w = min(self.init_weights[i] + speed * t, 1.)
            elif i < stage:
                w = 1.
            else:
                w = self.init_weights[i]

            if self._ready_for_M_N == False and i == len(losses) - 1 and w == 1.:
                self._ready_for_M_N = True
                self._ready_start_step = step
            l = losses[i] * w
            loss += l

        if self._ready_for_M_N:
            M_N = max(self.M_N - self._M_N_decay_speed *
                      (step - self._ready_start_step), self.eta_M_N)
        else:
            M_N = self.M_N

        return M_N * loss

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_recon_vs_gt(x, x_recon, save_path):
    # x, x_recon: (C, H, W) tensor
    x = x.cpu().permute(1,2,0).numpy()
    x_recon = x_recon.cpu().permute(1,2,0).detach().numpy()
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(np.clip(x, 0, 1))
    axs[0].set_title("Original")
    axs[0].axis('off')
    axs[1].imshow(np.clip(x_recon, 0, 1))
    axs[1].set_title("Reconstruction")
    axs[1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Trainer for NVAE with config file.")
    parser.add_argument('--config', type=str, required=True, help='Path to yaml config file')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    batch_size = config.get('batch_size', 128)
    z_dim = config.get('z_dim', 512)
    train_dataset_path = config.get('train_dataset_path', '/scratch2/hayeonjeong/nvae/dataset/train')
    valid_dataset_path = config.get('valid_dataset_path', '/scratch2/hayeonjeong/nvae/dataset/val')
    # test_dataset_path = config.get('test_dataset_path', '/scratch2/hayeonjeong/nvae/dataset/test')
    epochs = config.get('epochs', 100)
    n_cpu = config.get('n_cpu', 8)
    lr = config.get('lr', 0.01)
    exp_name = config.get('exp_name', None)  # exp_name을 None으로 초기화
    wandb_project = config.get('wandb_project', 'hierarchical-vae')
    pretrained_weights = config.get('pretrained_weights', None)

    model_variant = config.get('model_variant', 'kernel3')
    vae_mod = importlib.import_module(f"nvae.vae_celeba_{model_variant}")
    NVAE = vae_mod.NVAE
    model = NVAE(z_dim=z_dim, img_dim=(64, 64))

    
    # exp_name이 없으면 자동 생성
    if exp_name is None:
        dataset_name = os.path.basename(train_dataset_path)
        exp_name = f"{model_variant}_batch{batch_size}_zdim{z_dim}_{dataset_name}"
    
    # 저장 빈도 설정
    save_image_every = config.get('save_image_every', 500)
    save_checkpoint_every = config.get('save_checkpoint_every', 5)

    # Make experiment directories
    os.makedirs(exp_name, exist_ok=True)
    os.makedirs(f"{exp_name}/output", exist_ok=True)
    os.makedirs(f"{exp_name}/checkpoints", exist_ok=True)
    os.makedirs(f"{exp_name}/logs", exist_ok=True)

    # 로그 파일 준비
    log_file = open(f"{exp_name}/logs/train.log", "w")
    print(f"Experiment: {exp_name}", file=log_file)
    print(f"Experiment: {exp_name}")

    # wandb init
    wandb.init(
        entity="personal_nvae",
        project="hierarchical-vae",
        name=exp_name, config=config)

    # Data
    train_ds = ImageFolderDataset(train_dataset_path, img_dim=64)
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=n_cpu)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    

    # apply Spectral Normalization
    model.apply(add_sn)
    model.to(device)

    if pretrained_weights:
        model.load_state_dict(torch.load(pretrained_weights, map_location=device), strict=False)

    warmup_kl = WarmupKLLoss(init_weights=[1., 1. / 2, 1. / 8],
                             steps=[4500, 3000, 1500],
                             M_N=batch_size / len(train_ds),
                             eta_M_N=5e-6,
                             M_N_decay_step=36000)
    print('M_N=', warmup_kl.M_N, 'ETA_M_N=', warmup_kl.eta_M_N)

    optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-4)

    step = 0
    for epoch in range(epochs):
        model.train()

        for i, image in enumerate(train_dataloader):
            optimizer.zero_grad()

            image = image.to(device)
            image_recon, recon_loss, kl_losses = model(image)
            kl_loss = warmup_kl.get_loss(step, kl_losses)  # WarmupKLLoss 사용
            loss = recon_loss + kl_loss
            bpd = loss.item() / (np.log(2) * np.prod(image.shape[1:]))

            # 로그 파일에 기록
            log_str = f"Epoch {epoch}/{epochs}, Step {step}, Loss: {loss.item():.6f}, Recon: {recon_loss.item():.6f}, KL: {kl_loss.item():.6f}, BPD: {bpd:.6f}"
            print(log_str, file=log_file)
            log_file.flush()  # 즉시 파일에 쓰기

            # wandb log
            wandb.log({
                "recon_loss": recon_loss.item(),
                "kl_loss": kl_loss.item() if hasattr(kl_loss, 'item') else float(kl_loss),
                "total_loss": loss.item(),
                "bpd": bpd,
                "epoch": epoch,
                "step": step
            })

            loss.backward()
            optimizer.step()
            step += 1

            # 500 step마다 이미지 저장 (빈도 줄임)
            if step % save_image_every == 0:
                x = image[0]
                x_recon = image_recon[0]
                save_path = f"{exp_name}/output/epoch{epoch}_step{step}.png"
                save_recon_vs_gt(x, x_recon, save_path)
                wandb.log({"recon_vs_gt": wandb.Image(save_path)})

            
            if step != 0 and step % 100 == 0:
                with torch.no_grad():
                    z = torch.randn((1, z_dim, 2, 2)).to(device)
                    gen_img, _ = model.decoder(z)
                    gen_img = gen_img.permute(0, 2, 3, 1)
                    gen_img = gen_img[0].cpu().numpy() * 255
                    gen_img = gen_img.astype(np.uint8)
                    gen_img = gen_img / 255.0  # wandb.Image는 0~1 또는 0~255 모두 지원

                    wandb.log({
                        "random_sample": wandb.Image(gen_img, caption=f"epoch {epoch}, loss {loss.item():.6f}")
                    })

        scheduler.step()

        # 10 에포크마다 체크포인트 저장 (빈도 줄임)
        if (epoch + 1) % save_checkpoint_every == 0:
            torch.save(model.state_dict(), f"{exp_name}/checkpoints/ckpt_epoch{epoch}.pth")

            # FID 계산
            fake_dir = f"{exp_name}/fid_fake"
            generate_fake_images(model, z_dim, fake_dir, num_images=5000, device=device)
            real_dir = valid_dataset_path
            fid_value = calculate_fid(real_dir, fake_dir, batch_size=64, device=device, dims=2048)

            # log 파일 기록
            print(f"Epoch {epoch+1}: FID (random sampling): {fid_value}", file=log_file)
            log_file.flush()

            # wandb 기록
            wandb.log({"FID": fid_value, "epoch": epoch+1})

    log_file.close()  # 로그 파일 닫기
    wandb.finish()

if __name__ == '__main__':
    main()
