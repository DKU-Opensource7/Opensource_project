# Opensource_project
Dankook Uni. Opensource_Nofake

필터 프로그램 코드에 대한 설명입니다.
#  NoFake: 딥페이크 방지 필터
이 코드는 고성능 연산이 필요하므로 **Google Colab의 GPU 환경**에서 실행되도록 설계되었습니다.

---

##  파일 구성 및 역할

| 파일명 | 역할 및 설명 |
| **`nofake_filter.py`** | **[핵심 엔진]** 딥페이크 방지 알고리즘(InsightFace, LPIPS 등)이 구현된 파이썬 파일입니다. 강도 조절(상/중/하) 로직이 포함되어 있습니다. |
| **`NoFake_Server.ipynb`** | **[실행 스위치]** Google Colab에서 이 파일을 열어 실행하면, AI 서버가 가동됩니다. (Flask + ngrok 사용) |
| **`requirements.txt`** | **[환경 설정]** 서버 구동에 필요한 라이브러리 목록입니다. |

---

##  서버 구동 가이드 (웹 개발팀 필독)

웹페이지와 연동하기 위해 아래 순서대로 서버를 실행해주세요.

1. Colab에서 파일 열기
	1. 저장소에 있는 **`NoFake_Server.ipynb`** 파일을 클릭합니다.
	2. 화면 상단의 **"Open in Colab"** 버튼을 누르거나, 파일을 다운로드하여 Google Colab에 업로드합니다.


2. ngrok 토큰 입력 (세 번째 셀) 서버를 외부와 연결하기 위해 본인의 ngrok 토큰이 필요합니다. (ngrok 홈페이지에서 로그인 후 복사)

# [수정 전]
NGROK_TOKEN = "여기에_토큰을_넣으세요"

# [수정 후 예시]
NGROK_TOKEN = "2And9s8f7..."

3. 서버 가동
	1. Colab 상단 메뉴의 **[런타임] > [모두 실행]**을 클릭합니다.

	2. 실행이 완료되면 맨 마지막에 나오는 Public URL을 복사합니다.

4. 주의사항
	1. 주소 변경: 서버를 껐다 켤 때마다 ngrok 주소가 바뀝니다. 바뀐 주소를 사용해야 합니다.
	2. 세션 유지: Google Colab은 브라우저 탭을 닫거나 90분 이상 조작이 없으면 연결이 끊깁니다. 시연 중에는 Colab 탭을 켜두세요
