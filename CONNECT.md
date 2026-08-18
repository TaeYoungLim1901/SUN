# 대회 서버 붙기 — 사람이 할 일 3분

팀 노트북: `https://kasiai-kubeflow.kaist.ac.kr/notebook/astronomy-ai-2026-t87/nb-astronomy-ai-2026-t87/lab`

이 주소는 **KAIST Keycloak SSO 뒤에 있다.** 에이전트는 로그인 자격증명이 없어서 직접 못 들어간다
(확인: 위 URL 이 `kasiai-auth.kaist.ac.kr/realms/KAIST/...` 로 302 리다이렉트).
**사람이 브라우저에서 한 번 로그인해서 세션 쿠키만 넘겨주면**, 그 다음부터는 전부 에이전트가 한다.

---

## 계획 A — 세션 쿠키 (기본)

### 사람이 하는 것 (한 번, 30초)

1. **크롬**에서 위 lab 주소를 열고 SSO 로그인. JupyterLab 이 보이면 성공.
2. `F12` → **Network** 탭 → `Ctrl+R` 로 새로고침.
3. 목록 맨 위 요청(이름이 `lab` 인 것)에서 **우클릭 → Copy → Copy as cURL (bash)**.
4. 채팅에 "복사했어" 라고만 쳐라.

그 뒤 에이전트가 클립보드를 읽어 쿠키를 뽑고 `~/.sun_cookie` 에 저장한다:

```
python tools/remote.py auth      # 클립보드의 cURL 에서 쿠키 추출 → 저장 → 연결 확인
```

> 쿠키가 devtools 목록에 `oauth2_proxy_kubeflow_0`, `_1` 처럼 **쪼개져** 있을 수 있다 (oauth2-proxy 는
> 4KB 넘는 쿠키를 나눈다). 그래서 개별 쿠키를 복사하지 말고 **Copy as cURL** 로 통째로 가져온다.

### 그 뒤 (에이전트 전용)

```
python tools/remote.py check                 # GPU 이름·디스크·주피터 버전
python tools/remote.py ls                    # /home/jovyan 목록
python tools/remote.py put local.py work/x.py
python tools/remote.py get work/log.txt .
python tools/remote.py run -f train.py       # 원격 커널에서 실행, 출력 스트리밍
```

**학습은 `run` 으로 붙잡아 두지 마라.** 백그라운드로 던지고 로그만 회수한다:

```
python tools/remote.py run -c "import subprocess; subprocess.Popen('nohup python work/train.py > work/train.log 2>&1 &', shell=True)"
python tools/remote.py get work/train.log .
```

### 쿠키가 만료되면

증상: `쿠키 만료` 메시지 + exit 3. 대처: 위 1–4 를 다시 하면 끝. (SSO 세션 수명은 미확인 —
끊기는 주기를 알게 되면 여기 적어라.)

⚠ `RULES.md` §1: **같은 계정 다중 기기 동시 접속 시 기존 세션이 만료된다.** 쿠키를 넘겨주고 나서
사람이 다른 기기에서 또 로그인하면 이 경로가 죽는다. 접속은 한 기기로 일원화할 것.

---

## 계획 B — 노트북 안에서 되부르기 (계획 A 가 막히면)

쿠키가 IP 나 User-Agent 에 묶여 있거나, 웹소켓이 프록시에서 막히면 A 가 죽는다. 그때는 방향을 뒤집는다.
서버는 `!pip install` 이 허용되므로 **바깥으로 나가는 인터넷이 열려 있다.**

사람이 하는 것: JupyterLab 에서 새 셀 하나에 에이전트가 준 **부트스트랩 셀**을 붙여넣고 실행. 그게 끝.
그 셀은 개인 깃 저장소를 clone 하고, `work/inbox/` 를 폴링해 새 스크립트를 실행하고 결과를 push 한다.
그 뒤 에이전트는 로컬에서 커밋만 하면 서버가 알아서 집어간다.

제약: **개발 편의용일 뿐이다.** 최종 `submission/code.ipynb` 는 외부 clone 없이 **그 파일 하나로**
돌아가야 한다 (`RULES.md` §2).

---

## 계획 C — 최후 (A·B 둘 다 죽음)

에이전트가 셀 단위로 코드를 써 주고 사람이 붙여넣는다. 느리고 사람이 많이 든다. 여기까지 오지 마라.
