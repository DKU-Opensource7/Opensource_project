// ============================================================
// [1] Colab 실행 후 나온 ngrok 주소를 여기에 넣으세요!
// (뒤에 /upload는 붙이지 말고 주소만! 예: https://abcd.ngrok-free.app)
// ============================================================
const NGROK_URL = "https://chia-forestlike-dubitably.ngrok-free.dev"; 


document.addEventListener('DOMContentLoaded', () => {
    // ============================================================
    // [2] 약관 동의 및 스크롤 잠금 (기존 기능 유지)
    // ============================================================
    const body = document.body;
    const masterCheckbox = document.getElementById('terms-agree-all');
    const subCheckboxes = document.querySelectorAll('.required-agree'); // 클래스 이름 주의
    const uploadSection = document.getElementById('upload-section');
    const applyButton = document.getElementById('btn-apply');

    // (1) '전체 동의' 클릭 시
    if(masterCheckbox) {
        masterCheckbox.addEventListener('change', function() {
            subCheckboxes.forEach(cb => {
                cb.checked = this.checked;
            });
            toggleScrollLock(this.checked);
        });
    }

    // (2) 하위 체크박스 클릭 시
    subCheckboxes.forEach(cb => {
        cb.addEventListener('change', function() {
            const allChecked = Array.from(subCheckboxes).every(c => c.checked);
            if(masterCheckbox) masterCheckbox.checked = allChecked;
            toggleScrollLock(allChecked);
        });
    });

    function toggleScrollLock(isUnlocked) {
        if (isUnlocked) {
            body.classList.remove('scroll-locked');
            if(uploadSection) uploadSection.scrollIntoView({ behavior: 'smooth' });
        } else {
            body.classList.add('scroll-locked');
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
        if (applyButton) {
            applyButton.disabled = !isUnlocked;
        }
    }


    // ============================================================
    // [3] 이미지 업로드 미리보기 기능 (추가됨)
    // ============================================================
    const fileInput = document.getElementById('image-upload');
    const imgBefore = document.getElementById('image-before');
    const placeholderBefore = document.getElementById('placeholder-before');

    if(fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    imgBefore.src = e.target.result;
                    // 원본 이미지 표시되면 안내 문구 숨기기
                    if(placeholderBefore) placeholderBefore.style.display = 'none'; 
                }
                reader.readAsDataURL(file);
            }
        });
    }


    // ============================================================
    // [4] 서버 통신 및 필터 적용 기능 (핵심 기능 추가됨)
    // ============================================================
    const btnApply = document.getElementById('btn-apply');
    const imgAfter = document.getElementById('image-after');
    const placeholderAfter = document.getElementById('placeholder-after');
    const btnDownload = document.getElementById('btn-download');

    if(btnApply) {
        btnApply.addEventListener('click', async () => {
            // 1. 파일 선택 여부 확인
            if (!fileInput || fileInput.files.length === 0) {
                alert("⚠️ 먼저 원본 이미지를 업로드해주세요.");
                return;
            }

            // 2. 강도(Strength) 값 가져오기 (HTML의 라디오 버튼 name="filter-strength")
            const strengthRadio = document.querySelector('input[name="filter-strength"]:checked');
            const strengthVal = strengthRadio ? strengthRadio.value : 'medium';

            // 3. 데이터 포장하기 (FormData)
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);     // 파이썬이 'file'로 받음
            formData.append('strength', strengthVal);        // 파이썬이 'strength'로 받음

            // 4. 로딩 표시 보여주기
            if(placeholderAfter) {
                placeholderAfter.style.display = 'flex';
                placeholderAfter.innerHTML = "<span class='placeholder-text'>AI 변환 중...⏳<br>(약 1분 소요)</span>";
            }

            try {
                // 5. 서버로 전송 (POST)
                const response = await fetch(NGROK_URL + "/upload", {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    // 6. 성공! 이미지 받아서 보여주기
                    const blob = await response.blob();
                    const imgUrl = URL.createObjectURL(blob);
                    
                    imgAfter.src = imgUrl;
                    if(placeholderAfter) placeholderAfter.style.display = 'none'; // 안내 문구 숨김
                    
                    // 다운로드 버튼 기능 연결 및 표시
                    if(btnDownload) {
                        btnDownload.style.display = 'inline-block';
                        btnDownload.onclick = () => {
                            const link = document.createElement('a');
                            link.href = imgUrl;
                            link.download = "NoFake_Result.jpg";
                            link.click();
                        };
                    }
                } else {
                    alert("서버 오류 발생! (ngrok 주소를 확인하세요)");
                    if(placeholderAfter) placeholderAfter.innerHTML = "<span class='placeholder-text'>오류 발생 ❌</span>";
                }
            } catch (error) {
                console.error(error);
                alert("서버 연결 실패!\n1. Colab이 켜져 있나요?\n2. script.js 맨 윗줄 주소가 맞나요?");
                if(placeholderAfter) placeholderAfter.innerHTML = "<span class='placeholder-text'>연결 실패 ⚠️</span>";
            }
        });
    }
});
