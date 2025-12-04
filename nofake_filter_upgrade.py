import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from insightface.app import FaceAnalysis
import lpips
from PIL import Image
import gdown

# =========================================================
# 1. 딥러닝 모델 구조 정의 (ArcFace - IResNet)
# =========================================================

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64: raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1: raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        return out

class IResNet(nn.Module):
    fc_scale = 7 * 7
    def __init__(self, block, layers, dropout=0, num_features=512, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        super(IResNet, self).__init__()
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None: replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3: raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple")
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x

def iresnet100(pretrained=False, progress=True, **kwargs):
    return IResNet(IBasicBlock, [3, 13, 30, 3], **kwargs)


# =========================================================
# 2. NoFakeShield 클래스 정의 (핵심 엔진)
# =========================================================
class NoFakeShield:
    def __init__(self, model_link_or_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"⚙️ [System] Device: {self.device}")

        # [자동 다운로드] 모델 파일 처리
        if model_link_or_path.startswith('http'):
            local_model_path = 'arcface_model.pth'
            if not os.path.exists(local_model_path):
                print(f"⬇️ [System] 모델 다운로드 시작... (1~2분 소요)")
                gdown.download(model_link_or_path, local_model_path, quiet=False, fuzzy=True)
            self.model_path = local_model_path
        else:
            self.model_path = model_link_or_path

        # 1. LPIPS 모델 로드
        print("👁️ [System] Loading LPIPS model...")
        self.lpips_loss = lpips.LPIPS(net='alex').to(self.device)
        self.lpips_loss.eval()

        # 2. 얼굴 감지용 모델 로드
        print("🔍 [System] Loading Face Detection model...")
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        # 3. 공격 대상 모델 (ArcFace r100) 로드
        print("🛡️ [System] Loading ArcFace recognition model...")
        try:
            self.recognition_model = iresnet100(dropout=0.0, fp16=False, num_features=512).to(self.device)
            self.recognition_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.recognition_model.eval()
            for param in self.recognition_model.parameters():
                param.requires_grad = False
            print("✅ [System] ArcFace model loaded successfully!")
        except Exception as e:
            print(f"🚨 [ERROR] 모델 로드 실패: {e}")
            raise e

        # 4. 전처리기 정의
        self.preprocess = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def align_face(self, img, landmark):
        src = np.array([
            [30.2946, 51.6963], [65.5318, 51.5014],
            [48.0252, 71.7366], [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        src[:, 0] += 8.0
        dst = landmark.astype(np.float32)
        M, _ = cv2.estimateAffinePartial2D(dst, src)
        if M is None: return None, None
        warped = cv2.warpAffine(img, M, (112, 112), borderMode=cv2.BORDER_REPLICATE)
        return warped, M

    def get_noise_mask(self, original_shape, M):
        """ 얼굴 영역에만 노이즈를 주기 위한 마스크 생성 """
        M_inv = cv2.invertAffineTransform(M)
        
        # 112x112 크기의 흰색 마스크 생성 (테두리는 검게 해서 부드럽게)
        mask_small = np.full((112, 112, 3), 1.0, dtype=np.float32)
        border = 10 
        # 테두리 부분 0으로 처리 (경계선 부드럽게)
        cv2.rectangle(mask_small, (0,0), (112, 112), (0,0,0), border*2)
        mask_small = cv2.GaussianBlur(mask_small, (21, 21), 0)
        
        # 원래 이미지 크기로 복구 (Inverse Warp)
        mask_warped = cv2.warpAffine(mask_small, M_inv, (original_shape[1], original_shape[0]), borderValue=(0,0,0))
        return mask_warped

    def generate_noise(self, image_path, epsilon=0.05, alpha=0.01, iterations=30, lambda_lpips=20.0):
        original_image = cv2.imread(image_path)
        if original_image is None: raise ValueError("Image not found")
        img_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        faces = self.face_app.get(img_rgb)
        if len(faces) == 0:
            print("⚠️ No face detected. Returning original.")
            return original_image

        # [수정됨] 원본 고화질을 유지하기 위해 float32 베이스 이미지 생성
        result_image_float = img_rgb.copy().astype(np.float32) / 255.0
        print(f"🧩 [Process] Found {len(faces)} face(s). Generating High-Res Noise...")

        for i, face in enumerate(faces):
            aligned_face, M = self.align_face(img_rgb, face.kps)
            if aligned_face is None: continue

            # 텐서 준비
            face_tensor = self.preprocess(Image.fromarray(aligned_face)).unsqueeze(0).to(self.device)
            original_embedding = self.recognition_model(face_tensor).detach()
            original_embedding = F.normalize(original_embedding, p=2, dim=1)

            # 노이즈(delta) 초기화
            delta = torch.zeros_like(face_tensor, requires_grad=True).to(self.device)

            # ---------------------------
            # PGD 공격 루프 (노이즈 생성)
            # ---------------------------
            for _ in range(iterations):
                noisy_face = face_tensor + delta
                noisy_embedding = self.recognition_model(noisy_face)
                noisy_embedding = F.normalize(noisy_embedding, p=2, dim=1)

                loss_adv = F.cosine_similarity(noisy_embedding, original_embedding).mean()
                loss_lpips_val = self.lpips_loss(noisy_face, face_tensor).mean()
                loss = loss_adv + lambda_lpips * loss_lpips_val
                loss.backward()

                grad = delta.grad.detach()
                delta.data = delta.data - alpha * torch.sign(grad)
                delta.data = torch.clamp(delta.data, -epsilon, epsilon)
                delta.data = torch.clamp(face_tensor + delta.data, -1, 1) - face_tensor
                delta.grad.zero_()

            # ---------------------------
            # [핵심 수정] 원본 교체 대신 '노이즈'만 추출하여 더하기
            # ---------------------------
            # 1. 델타(노이즈)만 추출 (-1 ~ 1 범위)
            noise_delta = delta.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
            
            # 2. 노이즈의 크기를 좀 더 부드럽게 (RGB scale 0~1 수준으로 조정)
            # Preprocessing에서 std=0.5로 나눴으므로, 다시 0.5를 곱해줘야 실제 색상값이 됨
            noise_delta = noise_delta * 0.5 

            # 3. 노이즈만 원래 크기로 복구 (Inverse Warp)
            M_inv = cv2.invertAffineTransform(M)
            # 노이즈가 없는 곳은 0(변화 없음)으로 채움
            unwarped_noise = cv2.warpAffine(noise_delta, M_inv, (img_rgb.shape[1], img_rgb.shape[0]), borderValue=(0,0,0))

            # 4. 마스크 생성 (얼굴 부분에만 노이즈 적용)
            mask_alpha = self.get_noise_mask(img_rgb.shape, M)

            # 5. 원본 이미지에 노이즈 '더하기' (덮어쓰기 X)
            # 마스크가 있는 영역에만 노이즈를 더해줍니다.
            result_image_float += unwarped_noise * mask_alpha

        # 0~1 사이로 값 자르기 (Overflow 방지)
        result_image_float = np.clip(result_image_float, 0.0, 1.0)
        
        # 다시 이미지 포맷(0~255)으로 변환
        return cv2.cvtColor((result_image_float * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


# =========================================================
# 3. 글로벌 설정 및 인스턴스 초기화
# =========================================================

MODEL_LINK = 'https://drive.google.com/file/d/1NEJ1f6aDMoYDc65p508TXKzppNRi2ktD/view?usp=drive_link'

# 서버가 켜질 때 모델을 미리 로딩합니다.
print("🚀 [Init] NoFake Shield 엔진을 초기화합니다...")
shield = NoFakeShield(MODEL_LINK)


# =========================================================
# 4. 강도 조절 및 외부 호출 함수
# =========================================================

def get_filter_params(strength):
    """ 웹에서 선택한 강도에 따라 파라미터(epsilon, iterations) 반환 """
    if strength == 'high':
        return 0.025, 30   # 상: 강도 높임 (노이즈가 더 보이지만 방어력 up)
    elif strength == 'low':
        return 0.015, 30  # 하: 정말 미세한 노이즈
    else:
        return 0.02, 30  # 중: 기본값

def apply_deepfake_protection(image_path, output_path, strength='medium'):
    # 강도 설정
    epsilon, iterations = get_filter_params(strength)
    print(f"🔄 [Request] 강도: {strength} (Eps: {epsilon}, Iter: {iterations})")
    
    try:
        # 필터 적용 (shield 인스턴스 사용)
        result_img = shield.generate_noise(
            image_path, 
            epsilon=epsilon, 
            iterations=iterations,
            lambda_lpips=20.0  # 화질 유지 가중치
        )
        
        # 결과 저장
        cv2.imwrite(output_path, result_img)
        print(f"✅ [Success] 고화질 저장 완료: {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ [Error] 필터 적용 실패: {e}")
        return None