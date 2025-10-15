import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil

# NVAE 모델 import
from nvae.vae_celeba import NVAE
from nvae.dataset import ImageFolderDataset
from nvae.utils import add_sn
from fid.fid_score import calculate_fid_given_paths

def generate_fake_images(model, z_dim, save_dir, num_images=5000, device='cuda'):
    """Fake 이미지 생성 (train_z_dim.py에서 가져옴)"""
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
    """FID 계산 (train_z_dim.py에서 가져옴)"""
    fid_value = calculate_fid_given_paths([real_dir, fake_dir], batch_size, device, dims)
    return fid_value

def load_checkpoint(model, checkpoint_path):
    """Checkpoint 로드"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    return checkpoint

def calculate_bpd(model, dataloader, device):
    """Bits per dimension 계산 (train_z_dim.py의 loss 계산 로직 활용)"""
    model.eval()
    total_loss = 0
    total_pixels = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating BPD"):
            x = batch.to(device)
            
            # NVAE forward pass (train_z_dim.py와 동일)
            image_recon, recon_loss, kl_losses = model(x)
            
            # KL loss 계산 (단순 합 - WarmupKLLoss 없이)
            kl_loss = sum(kl_losses)
            total_loss_batch = recon_loss + kl_loss
            
            total_loss += total_loss_batch.item()
            total_pixels += x.numel()
    
    # BPD = loss / (log(2) * num_pixels) (train_z_dim.py와 동일)
    bpd = total_loss / (np.log(2) * total_pixels)
    return bpd

def evaluate_checkpoint(model, dataloader, device, z_dim, valid_dataset_path, exp_dir):
    """단일 checkpoint 평가 (BPD + FID)"""
    model.eval()
    
    # BPD 계산
    bpd = calculate_bpd(model, dataloader, device)
    
    # FID 계산
    print(f"Calculating FID for {exp_dir}...")
    
    # Fake 이미지 생성
    fake_dir = f"{exp_dir}/fid_fake_temp"
    generate_fake_images(model, z_dim, fake_dir, num_images=1000, device=device)
    
    # FID 계산
    fid = calculate_fid(valid_dataset_path, fake_dir, batch_size=64, device=device, dims=2048)
    
    # 임시 폴더 정리
    shutil.rmtree(fake_dir, ignore_errors=True)
    
    return {
        'bpd': bpd,
        'fid': fid
    }

def find_best_checkpoint(checkpoint_results):
    """Best checkpoint 찾기 (BPD와 FID 모두 고려)"""
    best_checkpoint_bpd = None
    best_score_bpd = float('inf')
    best_checkpoint_fid = None
    best_score_fid = float('inf')
    
    for checkpoint_name, results in checkpoint_results.items():
        # BPD 기준 (낮을수록 좋음)
        bpd_score = results['bpd']
        if bpd_score < best_score_bpd:
            best_score_bpd = bpd_score
            best_checkpoint_bpd = checkpoint_name
        
        # FID 기준 (낮을수록 좋음)
        fid_score = results['fid']
        if fid_score < best_score_fid:
            best_score_fid = fid_score
            best_checkpoint_fid = checkpoint_name
    
    return {
        'best_bpd': (best_checkpoint_bpd, best_score_bpd),
        'best_fid': (best_checkpoint_fid, best_score_fid)
    }

def main():
    # 설정
    exp_dir = "baseline_batch256_zdim128_train"
    checkpoint_dir = f"{exp_dir}/checkpoints"
    valid_dataset_path = '/scratch2/hayeonjeong/nvae/dataset/val'
    z_dim = 128
    batch_size = 64
    num_workers = 4
    
    print(f"=== Validation 시작 ===")
    print(f"Experiment: {exp_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Validation dataset: {valid_dataset_path}")
    print(f"z_dim: {z_dim}")
    print()
    
    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Validation 데이터셋 준비
    val_ds = ImageFolderDataset(valid_dataset_path, img_dim=64)
    val_dataloader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    # NVAE 모델 초기화
    model = NVAE(z_dim=z_dim, img_dim=(64, 64))
    model.apply(add_sn)  # Spectral Normalization 적용
    model.to(device)
    
    # Checkpoint 파일들 찾기
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_files = list(checkpoint_path.glob("ckpt_epoch*.pth"))
    checkpoint_files.sort()
    
    print(f"Found {len(checkpoint_files)} checkpoints")
    print()
    
    # 각 checkpoint 평가
    checkpoint_results = {}
    
    for checkpoint_file in tqdm(checkpoint_files, desc="Evaluating checkpoints"):
        try:
            # Checkpoint 로드
            load_checkpoint(model, checkpoint_file)
            
            # 평가 수행
            results = evaluate_checkpoint(model, val_dataloader, device, z_dim, valid_dataset_path, exp_dir)
            
            checkpoint_name = checkpoint_file.stem
            checkpoint_results[checkpoint_name] = results
            
            print(f"  {checkpoint_name}: BPD={results['bpd']:.4f}, FID={results['fid']:.2f}")
            
        except Exception as e:
            print(f"Error evaluating {checkpoint_file}: {e}")
            continue
    
    # Best checkpoint 찾기
    if checkpoint_results:
        best_results = find_best_checkpoint(checkpoint_results)
        print()
        print("=== 최고 성능 Checkpoint ===")
        print(f"Best BPD: {best_results['best_bpd'][0]} (BPD: {best_results['best_bpd'][1]:.4f})")
        print(f"Best FID: {best_results['best_fid'][0]} (FID: {best_results['best_fid'][1]:.2f}")
        
        # 결과 저장
        output_file = f"{exp_dir}/validation_results.json"
        with open(output_file, 'w') as f:
            json.dump({
                'all_results': checkpoint_results,
                'best_results': best_results
            }, f, indent=2)
        
        print(f"\n결과가 {output_file}에 저장되었습니다.")
        
        # 최종 validation score 출력
        print("\n=== 최종 Validation Score ===")
        print(f"Best BPD Score: {best_results['best_bpd'][1]:.4f}")
        print(f"Best FID Score: {best_results['best_fid'][1]:.2f}")
        
    else:
        print("평가할 수 있는 checkpoint가 없습니다.")

if __name__ == '__main__':
    main() 