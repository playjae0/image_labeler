# Fresh Start

이 저장소는 초기화되었으며, Streamlit 기반 이미지 불량 라벨링 앱을 새로 시작할 준비가 완료되었습니다.

## Runtime Compatibility Check

- OS: Ubuntu 22.04 LTS ✅
- Node.js: >= 20.0 (현재 코드에서 필수 아님, 설치되어 있어도 충돌 없음) ✅
- npm: >= 10.0 (현재 코드에서 필수 아님, 설치되어 있어도 충돌 없음) ✅

현재 코드베이스는 Python + Streamlit 앱이며, Node/npm 빌드 체인을 사용하지 않습니다.
따라서 Node/npm 버전 조건은 "문제 없음(비침투)" 상태입니다.

## Python dependencies

아래 의존성을 설치한 뒤 실행하세요.

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```
