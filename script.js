// 법률 동의 스크롤락 부분
document.addEventListener('DOMContentLoaded', () => {
    const body = document.body;
    const agreeCheckbox = document.getElementById('terms-agree');
    const targetSection = document.getElementById('upload-section');

    // . 체크박스 클릭 이벤트를 감지
    agreeCheckbox.addEventListener('change', function() {
        // 체크박스가 체크되었을 때만 동작
        if (this.checked) {
            // 스크롤 잠금 해제
            body.classList.remove('scroll-locked');
            
            // 스크롤 이동시킵니다.
            if (targetSection) {
                targetSection.scrollIntoView({
                    behavior: 'smooth' // 부드럽게 스크롤
                });
            }
        } 
		//
        else {
            body.classList.add('scroll-locked');
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }

    });

});
