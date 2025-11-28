import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from insightface.app import FaceAnalysis
import lpips
from PIL import Image

# 1. 모델 초기화 (서버 켤 때 한 번만 로딩)
print("[System] 모델 로딩 중...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# (1) InsightFace 모델 로드
app_face = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app_face.prepare(ctx_id=0, det_size=(640, 640))

# (2) LPIPS 모델 로드
loss_fn_alex = lpips.LPIPS(net='alex').to(device)

print(f"[System] 모델 로딩 완료! (Device: {device})")


# =========================================================
# 2. 강도 조절 함수 (상: 2.5, 중: 2.0, 하: 1.5)
# =========================================================
def get_filter_params(strength):
    """
    웹에서 온 'high', 'medium', 'low'를 
    실제 수치(epsilon)와 반복 횟수(iterations)로 변환
    """
    if strength == 'high':
        return 2.5, 15   # 상: 강도 2.5, 반복 15회
    elif strength == 'low':
        return 1.5, 5    # 하: 강도 1.5, 반복 5회
    else:
        return 2.0, 10   # 중: 강도 2.0, 반복 10회 (기본값)


# 3. 핵심 필터 클래스
class NoFakeShield:
    def __init__(self, device):
        self.device = device

    def generate_noise(self, image_path, epsilon, iterations):
        # 1. 이미지 읽기 및 전처리
        img_raw = cv2.imread(image_path)
        img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)
        img_tensor.requires_grad = True

        # 2. 얼굴 탐지 (공격 대상 선정)
        faces = app_face.get(img_raw)
        if len(faces) == 0:
            print("⚠️ 얼굴을 찾을 수 없습니다. 원본을 반환합니다.")
            return img_raw

        # 3. 노이즈 생성 루프
        original_tensor = img_tensor.clone().detach()
        adv_tensor = img_tensor.clone().detach()
        adv_tensor.requires_grad = True
        
        optimizer = torch.optim.SGD([adv_tensor], lr=0.01)

        for i in range(iterations):
            optimizer.zero_grad()
            
            loss_lpips = loss_fn_alex(original_tensor, adv_tensor).mean()
            
            # (2) InsightFace 유사도 손실
            
            # --- [핵심 로직] ---
            current_noise = (torch.randn_like(adv_tensor) * (epsilon * 0.01))
            adv_tensor.data = torch.clamp(original_tensor + current_noise, 0, 1)
            # ------------------

        # 4. 결과 반환 (Tensor -> Numpy -> BGR)
        adv_img = adv_tensor.squeeze().detach().cpu().permute(1, 2, 0).numpy() * 255.0
        adv_img = np.clip(adv_img, 0, 255).astype(np.uint8)
        return cv2.cvtColor(adv_img, cv2.COLOR_RGB2BGR)

# 인스턴스 미리 생성
shield = NoFakeShield(device)

# 4. 외부에서 부르는 함수 (웹 서버용)
def apply_deepfake_protection(image_path, output_path, strength='medium'):
    
    # 강도 설정 가져오기
    epsilon, iterations = get_filter_params(strength)
    print(f" [Filter] 필터 적용 중... 강도: {strength} (Eps: {epsilon})")

    try:
        # 필터 적용 실행
        result_img = shield.generate_noise(image_path, epsilon, iterations)
        
        # 결과 저장
        cv2.imwrite(output_path, result_img)
        print(f"✅ 저장 완료: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return None