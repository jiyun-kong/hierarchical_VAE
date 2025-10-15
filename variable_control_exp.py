# latent space의 특정 채널 값 직접 조작하면서
# 이미지가 어떻게 변하는지 실시간으로 시각화하는 실험용 도구
# 키보드로 latent 값을 조절, 변화하는 이미지를 바로 확인할 수 있음

import torch

from nvae.utils import add_sn
from nvae.vae_celeba import NVAE
import numpy as np
import matplotlib.pyplot as plt
import cv2

def set_bn(model, num_samples=1, iters=100):
    model.train()
    for i in range(iters):
        if i % 10 == 0:
            print('setting BN statistics iter %d out of %d' % (i + 1, iters))
        z = torch.randn((num_samples, z_dim, 2, 2)).to(device)
        model.decoder(z)
    model.eval()

if __name__ == '__main__':
    device = "cuda:0"
    z_dim = 512
    model = NVAE(z_dim=z_dim, img_dim=64)
    model.apply(add_sn)
    model.to(device)

    model.load_state_dict(torch.load("../checkpoints/ae_ckpt_169_0.689621.pth", map_location=device), strict=False)

    z = torch.randn((1, 512, 2, 2)).to(device) # latent vector (batch, channel:feature map 개수 (latent channel), h, w)
    x = 1 # latent map의 위치 인덱스
    y = 1 # latent map의 위치 인덱스
    m = 1 # s 리스트의 인덱스
    s = [482, 14, 204, 255, 356, 397, 404, 437] # 조작할 latent 채널 인덱스 리스트
    alpha = 0.1 # step 크기
    freeze_level = 0 # 디코더에서 freeze할 계층

    # m = 14, 204, 255, 356, 397, 404, 437, 482 x = 1, y = 1 : 실험에서 쓸 인덱스 예시

    zs = model.decoder.zs # 디코더 내부 latent 변수 리스트

    while True:

        key = cv2.waitKey(200)

        with torch.no_grad():
            # 이미지 생성
            gen_imgs, losses = model.decoder(z, mode='fix', freeze_level=freeze_level)

            # 이미지를 numpy로 변환, 0~255로 스케일링, BGR로 변환, 2배 확대, 텍스트 추가
            gen_imgs = gen_imgs.permute(0, 2, 3, 1)
            for gen_img in gen_imgs:
                gen_img = gen_img.cpu().numpy() * 255
                gen_img = gen_img.clip(0, 255).astype(np.uint8)

                # plt.imshow(gen_img)
                # plt.savefig(f"output/ae_ckpt_%d_%.6f.png" % (epoch, total_loss))
                # plt.show()
                gen_img = cv2.cvtColor(gen_img, cv2.COLOR_RGB2BGR)
                gen_img = cv2.resize(gen_img, (int(gen_img.shape[0] * 2), int(gen_img.shape[1] * 2)), cv2.INTER_AREA)
                cv2.putText(gen_img, str(s[m]) + ',' + str(zs[-1][0, s[m], y, x].item()), org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=(255, 255, 255), thickness=1)
                cv2.imshow("hidden:", gen_img)


        # 키 입력에 따라 latent 변수 조작
        # w: 현재 채널의 값을 증가
        # s: 현재 채널의 값을 감소
        # a/d: 조작할 채널 인덱스 변경
        # q: 프로그램 종료

        # 예) z[0, 204, 1, 1] += 0.1
        # 1번 샘플의 204번째 채널의 (1, 1) 위치에 있는 값을 0.1만큼 증가시킴
        
        if key == ord('w'):
            zs[-1][:, s[m], y, x] += alpha
            # zs[-1] = torch.randn(1, 64, 16, 16)
        elif key == ord('s'):
            zs[-1][:, s[m], y, x] -= alpha
        elif key == ord('a'):
            m = (m - 1) % len(s)
        elif key == ord('d'):
            m = (m + 1) % len(s)
        elif key == ord('q'):
            exit(0)
