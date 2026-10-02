# HermesYume

> [Dreamer](https://github.com/EESIZ/clawdreamer)를 openclaw용으로 만들어 놨는데, 같은 걸 [Hermes Agent](https://github.com/NousResearch/hermes-agent)에도 붙여보자는 프로젝트
  이름은 Hermes + Yume(夢, 꿈)의 합성어

> AI 에이전트는 잠을 자지 않잠. 꿈도 꾸지 않음. → 이게 사실 AI 에이전트의 가장 큰 문제

> 꿈을 꾼다는건, 자면서 기억들을 압축하고 정리하는 과정을 우연히 의식이 깨어나는 바람에 보게 되는 과정이라고 함
  Dreamer에서 했던 그 가설 그대로, Hermes 에이전트에게도 매일 밤 잠을 재워보자는 것

요약 : 꿈이라는 과정은 = 인간 기억 DB 압축과정이라는 가설 → Hermes 에이전트에게도 유사한 기능을 추가해보자는 프로젝트

**HermesYume은 Hermes 에이전트에게 '꿈'을 선물한다.**

openclaw를 쓴다면 [Dreamer](https://github.com/EESIZ/clawdreamer) 쪽을 쓰면 된다.

[English](README.md) · [설계 노트](DESIGN.md)

## v2: 갈아엎은 이유

v1은 꿈 꾼 결과를 Hermes의 `MEMORY.md`/`USER.md`에 다시 써 넣었다. 두 파일 합쳐서 3,500자 남짓. 한도가 있으니 "항상성"이라고 이름 붙여서, 넘치면 약한 것부터 내보냈다.

막상 붙여보니 이게 문제였다:

- 두 파일은 이미 꽉 차 있었다. 자동으로 하나 쓰려면 뭔가 하나를 밀어내야 하고, 밀려난 게 하필 **사용자가 정해 둔 규칙**이었다. 첫 실행에서 바로
- 에이전트도 같은 파일을 낮에 고친다. 밤에 내가 고치고, 낮에 에이전트가 고치고... 둘이 같은 파일 두고 싸우는 구조

그리고 Dreamer 쪽 로그를 몇 달치 뜯어봤는데, 더 아팠다:

- 같은 파일 7개만 153일 동안 반복 처리하고 있었고, 한 번도 안 읽힌 파일이 37개 넘게 있었다
- 회상 95번 중 95번, 질문이랑 상관없는 기억 3개가 붙었다. "필요할 때만 꺼낸다"는 건 사실이 아니었음
- 판정 JSON이 100토큰에서 잘려서 2,815건이 조용히 "unrelated"로 처리됐다. 그래서 같은 얘기가 의역만 바뀌어 3,216번 새로 생김
- 중요도의 89%가 0.8, 종류는 99%가 fact. 신호 역할을 못 함
- 망각은 매일 밤 누적으로 깎는 방식이라, 모든 기억이 7~9번 실행 만에 사라졌다. 정작 0.1짜리 잡담은 영원히 남고

그래서 v2는 방향을 뒤집었다:

- **`MEMORY.md`/`USER.md`는 "항상 켜져 있는 작은 핵심"으로 그냥 둔다.** 야간 작업은 이 두 파일에 한 글자도 안 쓴다
- **장기기억은 LanceDB에 사실 단위로** 쌓는다 (`$HERMES_HOME/hermesyume/lancedb/`). 출처, 이력, 강도가 행마다 붙는다
- **대화할 때는 Hermes 메모리 provider가 관련 있는 것만** 꺼내 붙인다. 코사인 0.40 이상(OpenAI 임베딩 기준), 최대 5개, 1,000자까지
- **잊는 건 삭제가 아니라 상태 변화.** 흐려진 기억은 `dormant`가 돼서 자동 회상에서만 빠진다. 검색하면 나오고, 다시 언급되면 깨어난다

## 작동 원리

```
[Telegram / CLI / TUI]
        │
Hermes (자기 venv, 그대로)
  ├─ MEMORY.md / USER.md  ← 지금처럼 세션 시작할 때 시스템 프롬프트에
  └─ hermesyume provider  ($HERMES_HOME/plugins/hermesyume, 표준 라이브러리만)
        prefetch → serving/recall.sqlite (읽기 전용) → <memory-context> (최대 5개)
        훅       → live.db (회상 기록, 기억해/잊어 inbox)
                                     │
state.db (읽기 전용) ─────────────────┤
workspace/memory/*.md (읽기 전용) ────┤
                                     ▼
   yume dream  (systemd user timer, 매일 밤, 별도 venv, 잠금 하나)
     NREM: 정착한 대화 → 창 → LLM이 0~N개 주장 추출 → 게이트
     REM:  LanceDB에 upsert (중복 / 상태 변경 / 다른 측면 / 무관),
           회상 반영, 강도·상태 전이, 가드, 계획 하나 → 커밋 한 번
     → serving/recall.sqlite 원자적 교체 → Dream Log
```

### Phase 1: NREM -- "오늘 무슨 일이 있었지?"

- `state.db`는 **읽기 전용**으로만 연다. `content`만 읽고, provider가 기억을 끼워 넣은 API 사본(`api_content`)은 절대 안 읽는다. 안 그러면 회상한 기억이 다시 "새 기억"으로 학습되는 고리가 생김
- user/assistant 발화만, 30분 이상 조용하거나 끝난 세션만. cron이랑 합성 평가 세션은 뺀다
- 마크다운 노트(`md_sources`)도 읽는다. 파일은 안 옮기고, 어디까지 읽었는지 바이트 오프셋으로 기억
- 대화를 교환 단위로 이어 붙여 8,000자 이하 창을 만들고, 창마다 절대 날짜 머리말을 붙인다
- LLM은 **0개 이상**의 주장을 낸다. 종류는 13개(rule, profile, preference, reference, procedure, decision, lesson, project, fact, state, schedule, event, opinion). "남길 거 없음"도 정상 답변
- 그다음 결정론적 게이트: 절대 날짜 없이 "오늘/내일"만 쓴 것, 파일·세션 나열, "대화를 나눴다" 같은 메타 서술, 기억 시스템 자체에 대한 서술, 비밀값, 주입 패턴은 버린다. 버린 건 사유랑 같이 Dream Log에 남김

### Phase 2: REM -- "이건 내가 아는 것과 맞아?"

- 새 주장마다 가까운 기존 기억(벡터 top-k + 같은 주제)을 찾고, LLM한테 관계를 열거형으로 묻는다: `duplicate` / `state_change` / `different_aspects` / `unrelated`
  - 파싱 실패하면 `unknown`으로 두고 다음 밤에 다시 묻는다. **절대 unrelated로 위장하지 않음.** Dreamer에서 제일 크게 데인 부분
- **중복**: 새 행 안 만들고 기존 행에 근거만 추가. **사용자가 한 말만** 강화로 친다. 에이전트가 자기 말 되풀이한 걸로는 안 세짐
- **상태 변경**: 옛 행은 `superseded`로 남기고(지우지 않음) 체인으로 잇는다. 신구는 처리한 시각이 아니라 **일이 일어난 시각**으로 판정
- **다른 측면**: 합친 문장에 두 입력의 숫자·날짜·이름이 전부 살아 있을 때만 하나로 합친다. 하나라도 빠지면 둘 다 두고 연결만
- 고정(pin)되거나 보호된 기억을 사용자 근거 없이 바꾸려는 연산은 적용 안 하고 기록만 한다

### 망각: 깎지 않고 그때그때 계산

```
t_ref = max(생성 시각, 마지막 사용자 근거, 마지막 실제 사용)
hl    = 반감기[종류] · (1 + 0.5·ln(1 + 실제로 쓰인 횟수))
s     = 중요도 · 2^(−(지금 − t_ref)/hl)
```

입력이 같으면 결과도 같다. 하루 빼먹어도, 두 번 돌려도 결과가 안 바뀜. 0.10 아래로 떨어지고 21일 넘으면 `dormant`.
보호는 근거로 얻는다: 사용자가 분명히 말했거나(또는 다른 세션에서 두 번 말했거나), 핵심 파일에 있거나, "기억해"라고 한 규칙·프로필은 감쇠하지 않는다. 그리고 **붙여지기만 한 건 시계를 안 되돌린다.** 실제로 답에 쓰였을 때만 되돌림

### 회상 (provider)

- Hermes 프로세스 안에서 표준 라이브러리만 쓴다. Hermes venv에 설치하는 패키지 0개, 상시 데몬 0개
- 야간 작업이 만들어 둔 읽기 전용 SQLite 사본에서 검색한다. OpenAI 임베딩이면 2단계(256차원으로 거르고 1536차원으로 재정렬), 로컬 `hash` 임베딩이면 한 번에 정확히. 그다음 상대 컷, 중복 제거. 이미 핵심 파일에 있는 건 안 붙이고, 같은 세션에서 이미 붙인 것도 안 붙인다
- 임베딩 API가 죽으면 FTS 키워드 검색으로 대신한다
- 도구는 `yume_search`, `yume_remember`, `yume_forget` 세 개. **사용자가 먼저 말했을 때만** 쓰라고 적어 뒀다. 에이전트가 "이거 기억해 둘까요?" 하고 묻기 시작하면, 그 대화가 또 기억이 되고... 무한 루프. 그래서 에이전트는 자기 꿈 얘기를 먼저 꺼내지 않는다
- `inject: false`면 shadow 모드: 계산은 다 하고 기록만 하고 아무것도 안 붙인다. 이걸로 며칠 돌려 보고 임계값을 보정한 뒤 켠다

### Phase 3: Dream Log -- "오늘 밤 무슨 꿈을 꿨지?"

매 실행마다 `dream-log/YYYY-MM-DD_HHMMSS.md`가 생긴다. 뭘 읽었고 뭘 왜 뺐는지, 뽑은 주장 전부, 게이트에서 버린 것과 사유, 새로 생긴 기억·강화·대체(전→후)·통합(입력 2개→결과) 전문, 만료·휴면·부활, 회상 통계, 판정별 코사인 분포, 토큰이랑 비용까지.
잊은 건 id만 남기고, 비밀값은 어디에도 안 남긴다.

## 돌아가는 Hermes 옆에서 돌려도 괜찮은가

v1 때보다 훨씬 신경 썼다:

- `state.db`는 `mode=ro` + `query_only`. `MEMORY.md`/`USER.md`는 읽기만 하고 잠금 파일도 안 만든다. 두 파일에 쓰는 코드는 사람이 직접 치는 `yume core-restore`랑 `yume core-proposal apply` 둘뿐
- `yume dream --dry-run`은 LLM은 진짜로 부르지만, 쓰는 건 계획 파일이랑 `_dry` Dream Log뿐이다. 돌리고 나서 Hermes 홈 전체 해시가 그대로인지 테스트로 확인함
- 실행은 계획을 먼저 디스크에 쓰고 한 번에 커밋한다. 중간에 죽으면 다음 실행이 계획을 재생. `yume restore --run <id>`로 그 실행 전으로 되돌릴 수 있다
- 비밀값은 LLM에 보내기 전에 가리고, 저장 전에 거르고, 밤마다 다시 검사한다
- 운영 알림(실행 실패, 401, 모델 불일치, 2밤 연속 정체, 창 격리, 비밀값 발견, 회상 이상)은 기본이 `alerts.log` 파일뿐. 텔레그램으로 받고 싶으면 야간 작업이 봇 API로 **caption 없는 `.txt` 파일**만 보낸다. 대화 메시지가 아니라서 에이전트가 볼 일도, 인용할 일도 없다

## 빠른 시작

### 한 방에 설치 (Hermes가 돌아가는 머신에서)

```bash
curl -fsSL https://raw.githubusercontent.com/EESIZ/HermesYume/main/install.sh | bash -s -- --timer
```

Hermes 돌리는 그 사용자로 실행하면 된다. 하는 일:

- `$HERMES_HOME/state.db`가 있는지 본다 (`HERMES_HOME` 기본값은 `~/.hermes`, 프로필 쓰면 `--hermes-home DIR`)
- `~/HermesYume`에 코드를 받거나 최신으로 당긴다
- 야간 작업용 venv를 `~/.local/share/hermesyume/venv`에 만든다. [uv](https://docs.astral.sh/uv/) 있으면 uv로, 없으면 `python3 -m venv` + pip (Python 3.11이나 3.12 필요)
- provider를 `$HERMES_HOME/plugins/hermesyume`에 복사한다. **복사만 하고 켜지는 않음**
- `yume init` (이미 있는 `config.json`은 그대로 둠), `yume doctor`
- `--timer`: systemd user 타이머 등록 (매일 04:40, 서울 시간). user systemd가 없으면 crontab에 한 줄 넣는데, 그 전에 기존 crontab을 `$HERMES_HOME/hermesyume/crontab.bak`에 백업해 둔다

안 하는 일: `hermes config set memory.provider` 실행, `config.yaml`이나 `$HERMES_HOME/.env` 수정, Hermes 재시작. 켜는 건 직접 하라고 마지막에 명령을 그대로 찍어 준다 ([켜기](#켜기)). 다시 돌리면 업데이트만 된다.

API 키는 보통 따로 안 넣어도 된다. Hermes가 이미 쓰고 있는 `.env`에서 `DEEPSEEK_API_KEY`나 `OPENAI_API_KEY`를 이름으로 찾아 쓴다 ([제공자](#제공자-provider)). 둘 다 없으면 그 파일에 하나 넣으면 되고, 뭘 찾았는지는 `yume doctor`가 알려준다.

### 수동 설치

```bash
# 1. 야간 작업용 별도 venv (Hermes venv는 절대 안 건드림). 태그된 커밋을 설치
deploy/setup_venv.sh --ref v2.0.0
Y=~/.local/share/hermesyume/venv/bin/yume

# 2. 데이터 폴더, 설정, 빈 저장소 만들고 점검 (1토큰 호출 포함)
export HERMES_HOME=~/.hermes          # 프로필 쓰면 프로필 디렉토리
$Y init && $Y doctor

# 3. 지금 있는 기억 옮기기: 먼저 보고, 그다음 적용
$Y migrate --estimate
$Y migrate --dry-run                  # dream-log/*_dry.md 읽어보기
$Y migrate --approve-migration

# 4. provider 설치, shadow 모드로 시작
deploy/install_provider.sh --hermes-home "$HERMES_HOME"
$Y config set inject false
hermes config set memory.provider hermesyume

# 5. 매일 밤 타이머 (systemd user unit)
cp deploy/hermesyume-dream.{service,timer} ~/.config/systemd/user/
systemctl --user edit hermesyume-dream.service   # HERMES_HOME이 ~/.hermes가 아니면 여기서 지정
systemctl --user daemon-reload && systemctl --user enable --now hermesyume-dream.timer

# 6. shadow로 며칠 돌린 뒤
$Y calibrate                          # recall_min_cos 추천 (0.40 아래로는 안 내림)
$Y config set inject true
```

처음 며칠은 Dream Log를 직접 읽어보는 걸 추천. 내가 그렇게 해서 구멍을 찾았다.

### 켜기

설치만 해서는 Hermes 쪽은 아무것도 안 바뀐다. 켜는 건 세 단계고, 첫 단계는 무조건 shadow 모드:

1. `$Y config set inject false` -- 메시지마다 회상 계산은 다 하고 `live.db`에 기록만 한다. 붙이는 건 없음
2. `hermes config set memory.provider hermesyume` -- 다음 메시지부터 provider가 돈다
3. 며칠 밤 돌려 보고 `$Y status --recall`이랑 Dream Log를 본 다음, `$Y calibrate` → `$Y config set inject true`

`calibrate`는 0.40 아래로는 추천을 안 한다. OpenAI 임베딩이면 그게 맞고, `hash` 임베딩이면 shadow 숫자가 딴소리 하지 않는 한 실측 기본값 0.30을 그냥 두면 된다.

당장 멈추려면 `$Y config set enabled false` (provider가 파일을 다시 읽어서 재시작 필요 없음). 완전히 빼려면 `memory.provider` 줄을 지우면 된다. 데이터는 남는다. 코드를 업데이트했으면 Hermes 게이트웨이는 직접 재시작 (파이썬이 provider 모듈을 캐시함).

## 제공자 (provider)

키는 `$HERMES_HOME/.env`(Hermes가 원래 쓰는 그 파일)에서 **이름으로만** 읽고, 없으면 환경변수를 본다. 파일을 source 하지도 않고 값을 로그에 남기지도 않는다. `yume doctor`는 뭐가 골라졌는지랑 키를 어디서 찾았는지만 보여준다. 키 값은 절대 안 찍음.

**LLM** (추출이랑 관계 판정, 밤에만 씀) -- `llm_provider`:

| | 언제 | 모델 | |
|---|---|---|---|
| `deepseek` | `auto`일 때 `DEEPSEEK_API_KEY`가 있으면 | `deepseek-v4-flash` (`deepseek_model`) | 제일 쌈. thinking은 끄고, JSON 모드로 부른다 (서버가 JSON 모드를 거부하면 그냥 파싱으로 버팀). `DEEPSEEK_BASE_URL`로 주소를 바꿀 수 있고, DeepSeek 키는 DeepSeek 주소 말고는 어디에도 안 보낸다 |
| `openai` | `auto`인데 DeepSeek 키가 없으면 | `gpt-4.1-mini` (`extract_model` / `judge_model`) | OpenAI 호환이면 다 됨 (`llm_base_url`, `OPENAI_BASE_URL`) |

`extract_model` / `judge_model`은 OpenAI 모델 이름이다. DeepSeek일 때는 `deepseek-*`로 시작하는 이름을 넣었을 때만 쓰인다. Dream Log 비용 줄이 맞으려면 `llm_price_in_per_mtok` / `llm_price_out_per_mtok`을 쓰는 쪽 단가로 바꿔 두면 된다.

**임베딩** (기억마다 저장하고, 회상할 때 메시지마다 하나) -- `embed_provider`:

| | 언제 | 모델 | 장단점 |
|---|---|---|---|
| `openai` | `auto`일 때 `OPENAI_API_KEY`가 있으면 | `text-embedding-3-small`, 1536차원 | 의미로 찾는다. 단어가 하나도 안 겹쳐도 같은 얘기면 잡음. 메시지마다 싼 HTTP 호출 한 번 |
| `hash` | `auto`인데 OpenAI 키가 없으면 (DeepSeek엔 임베딩 API가 없음) | `hash/ngram-v1`, 1024차원, 표준 라이브러리 | 공짜, 로컬, 키도 네트워크도 필요 없음. 야간 작업이랑 provider가 같은 파일을 써서 벡터가 비트 단위로 같다. 대신 **단어가 겹치는지만 본다** (단어 + 글자 2/3-gram). "Postgres 16" vs "17", 조사만 다른 한국어 문장은 잡지만, 단어가 하나도 안 겹치는 질문은 못 잡음. 임계값도 따로 (아래) |

고르는 건 `yume init` 때 한 번이고, `config.json`에 `"embed_provider": "openai"`나 `"hash"`로 박아 둔다. 그래서 나중에 키를 넣거나 빼도 저장된 벡터가 바뀌지 않는다. 벡터마다 모델 id(`openai/text-embedding-3-small@1536`, `hash/ngram-v1@1024`)가 붙어 있고, 설정한 모델이랑 저장된 모델이 다르면 야간 작업이 아예 안 돈다. 섞일 일은 없음. `yume reembed`는 같은 제공자 안에서 모델 바꿀 때만 되고, `hash` ↔ API 전환은 아직 안 만들었다 (바꾸려면 데이터 폴더를 새로 시작).

`hash` 임계값은 한국어·영어 합성 문장 쌍으로 관련/무관 코사인 분포를 재서 정했다 (`config.json`에 직접 넣은 값이 있으면 그게 우선):

| 키 | OpenAI | hash | 이유 (hash) |
|---|---|---|---|
| `recall_min_cos` | 0.40 | 0.30 | 무관한 질문→기억 쌍은 0.19%만 넘고, 관련 쌍은 43%가 넘는다 (API 없이 가는 대가) |
| `pinned_min_cos` / `search_min_cos` | 0.33 / 0.30 | 0.25 / 0.20 | |
| `injected_strong_cos` | 0.50 | 0.40 | |
| `candidate_cos` | 0.72 | 0.40 | 상태 변경의 93%가 판정 LLM까지 간다 |
| `sweep_cos` | 0.82 | 0.55 | |
| `auto_dup_cos` | 0.95 | 0.90 | 숫자는 그대로인 상태 변경이 0.78까지 나와서, 거의 같은 문장도 판정 LLM을 거치게 |
| `suppress_cos` / `core_match_cos` / `mmr_cos` | 0.90 / 0.90 / 0.92 | 0.80 / 0.75 / 0.85 | |

`recall_min_cos` 하한은 신경망 임베딩이면 0.40, `hash`면 0.25.

## 마이그레이션

`yume migrate`는 아무것도 잃지 않게 만들었다:

| 단계 | 내용 |
|---|---|
| M0 | 모든 원천을 읽기 전용으로 훑고 sha256 기록 (`migration/inventory.json`) |
| M1 | 핵심 파일 항목 하나당 행 하나, 글자 그대로. 값 없이 머리글만 있는 항목은 목록에 따로 |
| M2 | `USER.md`에서 프로필·규칙으로 분류된 항목은 자동으로 pin. 승인 절차 없음, 빼고 싶으면 `yume unpin` |
| M3 | `MEMORY.md`의 옛 "Session: …" 조각은 원문 그대로 휴면(레거시) 행으로 보존하고, 따로 추출도 한다 |
| M4 | 마크다운 노트 백로그 (`md_sources`) |
| M5 | 옛 Dreamer 덤프(`migration/dreamer_memories.json`) → 휴면 레거시 행 |
| M6 | `state.db` 백필 (처음부터 새로 시작하려면 `--statedb-start now`) |
| M7 | 전체 REM, export, 보정 힌트, 마이그레이션 Dream Log, `proposals/MEMORY.md.proposed` |

`--only core,dump,memory_md,md,statedb`로 단계를 고르고, `--estimate`로 창 수·호출 수·비용을 먼저 본다. 예상 호출이 `--max-llm-calls`를 넘으면 실행을 거부한다.
`core_map`이 핵심 파일 항목을 하나도 빠짐없이 설명해야 통과. `MEMORY.md` 정리 제안은 그냥 파일이다. 직접 확인하고 `yume core-proposal apply`로 적용

## 명령

| 명령 | |
|---|---|
| `yume dream [--dry-run] [--offline] [--now +70d] [--settle-minutes N]` | 야간 실행 (`--json`이면 통계 출력) |
| `yume status [--recall]` | 최근 실행, 백로그, 회상 상태 |
| `yume search <q> [--include-inactive]` / `yume inspect --query/--id/--label` | 기억 들여다보기 |
| `yume forget <id>` / `yume unpin <id>` / `yume pin list` | 관리 |
| `yume restore --run <id> [--reprocess] [--unforget]` | 그 실행과 이후 실행을 함께 되돌리기 (잊은 기억은 `--unforget` 없으면 다시 잊음) |
| `yume core-check` / `yume core-restore <id>` / `yume core-proposal apply` | 핵심 파일 (뒤의 둘만 파일에 씀) |
| `yume migrate`, `yume calibrate`, `yume export`, `yume reembed` | 위 참고 |
| `yume alert-flush` | 쌓인 운영 알림 보내기 (`alert_telegram` 켰을 때) |
| `yume doctor [--offline]`, `yume init`, `yume config set/get/show` | 준비 |
| `yume debug plant/event` | 샌드박스 전용 시험 도구 (라이브 홈에서는 거부) |

`--offline`은 규칙 기반 가짜 LLM과 해시 임베딩으로 네트워크 없이 돌립니다(샌드박스 점검용). 라이브 홈에서는 `--dry-run`과 함께만 쓸 수 있습니다.

*라이브 홈*은 `~/.hermes`, 게이트웨이가 한 번이라도 돈 홈(`gateway_state.json`, `gateway.pid`, `gateway.lock`), 그리고 `$HERMESYUME_PROTECT_HOMES`(`:`로 구분)나 `protect_homes` 키에 적은 홈이다. 샌드박스 홈은 보호된 홈의 `workspace_dir`, `$HERMESYUME_PROTECT_WORKSPACES`, `protect_workspaces`에 절차 문서를 안 쓴다.

## 설정

`$HERMES_HOME/hermesyume/config.json` 하나. 평평한 키, 비밀값 없음, 야간 작업이랑 provider가 같이 읽는다. 전체 키와 기본값은 `config.json.example`에 있다. 자주 만질 것만:

| 키 | 기본값 | |
|---|---|---|
| `enabled` / `inject` | `true` / `true` | 전체 스위치 / shadow 모드 |
| `llm_provider` / `embed_provider` | `auto` / `auto` | [제공자](#제공자-provider) 참고. `init`이 고른 `embed_provider`를 박아 둠 |
| `extract_model` / `judge_model` | `gpt-4.1-mini` | OpenAI용. nano는 써보니 못 쓰겠더라 |
| `deepseek_model` | `deepseek-v4-flash` | DeepSeek용 |
| `include_sources` | `["telegram","cli","tui"]` | 배울 세션 종류 |
| `workspace_dir` / `md_sources` | | 에이전트 작업 폴더(절차 문서가 `docs/yume/`에 쌓임) / 마크다운 노트 폴더. 새로 설치하면 비어 있으니 `yume config set`으로 넣기 |
| `hermes_runtime_dir` | | `tools/threat_patterns.py`를 가져올 Hermes 체크아웃. 비우면 `$HERMES_RUNTIME_DIR`, 그다음 import 가능한 `hermes_cli` 위치, 둘 다 없으면 동봉 사본 |
| `exclude_first_message_regex` / `deny_cwd_globs` / `strip_line_regex` | 비어 있음 | 첫 메시지가 맞는 세션 빼기(합성 평가 같은 것) / 그 cwd에서 시작한 세션 빼기 / 추출 전에 봇 상태 줄 지우기 |
| `protect_homes` / `protect_workspaces` | `[]` | 라이브로 칠 홈·워크스페이스 추가 ([명령](#명령) 참고) |
| `recall_min_cos` / `recall_k` / `recall_budget_chars` | `0.40` (hash `0.30`) / `5` / `1000` | |
| `max_windows_per_run` / `max_llm_calls` | `60` / `400` | 실행당 상한 (넘친 건 다음 밤에 이어서) |
| `alert_telegram` | `false` | 알림을 `.txt` 문서로 보내기 |

키는 `$HERMES_HOME/.env`에서 **이름으로만** 읽는다 (`DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, 알림 켜면 `TELEGRAM_BOT_TOKEN`, `YUME_ALERT_CHAT_ID`). 파일을 source 하지 않고, 값은 로그에 안 남긴다.

## 디렉토리 구조

```
$HERMES_HOME/hermesyume/            (폴더 0700, 파일 0600)
  config.json
  lancedb/                          memories, memory_history, suppress  (정본)
  ledger.db                         워터마크, 창, 실행 기록, 핵심 파일 관찰, 감사 기록
  live.db                           provider → 야간 작업: 회상 이벤트, inbox, health
  serving/recall.sqlite             provider용 읽기 전용 사본 (원자적 교체)
  runs/<run_id>/plan.json           재생 가능한 커밋 계획
  dream-log/                        매일 밤 리포트
  alerts.log                        운영 알림
  migration/  proposals/  backups/
```

## 한계

- 인젝션 검사는 여전히 패턴 기반이다. 진짜 방어선은 도구 출력을 아예 입력으로 안 쓰는 것
- 품질은 LLM 따라간다. 작은 모델은 쓸데없는 걸 뽑거나 관계를 잘못 판정한다. 게이트랑 감쇠가 쓰레기가 쌓이는 건 막아 주지만, 모델을 똑똑하게 만들어 주진 않음
- "실제로 쓰였다"는 판정은 답변에 나온 토큰 비율로 하는 휴리스틱이다. 오판해도 수명이 조금 늘 뿐, 지워지진 않는다
- 의미로 찾는 회상에는 임베딩 API(OpenAI 호환)가 필요하다. 없으면 `hash` 임베딩이라 겹치는 단어로만 찾는다. 돌다가 API가 죽으면 키워드 검색으로 버틴다

"저번 주에 설정 바꿨다고 했잖아요" 같은 일은 더 이상 없다. -- 이번엔 진짜로, "아마도"

## 테스트

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements-dream.txt pytest -e .
.venv/bin/python -m pytest -q tests/dream            # 야간 작업 쪽: 네트워크 없이, 가짜 LLM·임베딩
cd <hermes-runtime> && venv/bin/python -B -m unittest discover -s <repo>/tests/provider -t <repo>   # provider 쪽: Hermes의 파이썬으로
```

실제 설치본이랑 비교하는 테스트 셋은 따로 켜야 돈다: `HERMESYUME_TEST_LIVE_HOME=<홈>`(state.db 컬럼 비교와 라이브 홈 가드, 읽기만 함), `HERMES_RUNTIME_DIR=<체크아웃>`(위협 패턴 비교). 안 주면 건너뛴다.

## 요구 사항

- Python 3.11이나 3.12 (야간 venv: LanceDB, pyarrow, numpy, openai, pyyaml). [uv](https://docs.astral.sh/uv/)는 있으면 쓴다 (`deploy/setup_venv.sh`는 uv 필수)
- 메모리 provider 플러그인을 지원하는 Hermes Agent (provider 자체는 표준 라이브러리만)
- LLM 키 하나: `DEEPSEEK_API_KEY`나 `OPENAI_API_KEY` (OpenAI 호환이면 다 됨). 임베딩은 `OPENAI_API_KEY`, 없으면 `hash`로 키 없이

## Donations
If you find it useful...
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/V7V21XAPRC)

## 라이선스

MIT
