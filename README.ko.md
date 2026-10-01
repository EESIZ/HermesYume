# HermesYume

> [Dreamer](https://github.com/EESIZ/clawdreamer)를 openclaw용으로 만들어 놨는데, 같은 걸 [Hermes Agent](https://github.com/NousResearch/hermes-agent)에도 붙여보자는 프로젝트
  이름은 Hermes + Yume(夢, 꿈)의 합성어

> AI 에이전트는 잠을 자지 않잠. 꿈도 꾸지 않음. → 이게 사실 AI 에이전트의 가장 큰 문제

> 꿈을 꾼다는건, 자면서 기억들을 압축하고 정리하는 과정을 우연히 의식이 깨어나는 바람에 보게 되는 과정이라고 함
  Dreamer에서 했던 그 가설 그대로, Hermes 에이전트에게도 매일 밤 잠을 재워보자는 것

요약 : 꿈이라는 과정은 = 인간 기억 DB 압축과정이라는 가설 → Hermes 에이전트에게도 유사한 기능을 추가해보자는 프로젝트

**HermesYume은 Hermes 에이전트에게 '꿈'을 선물한다.**

openclaw를 쓴다면 [Dreamer](https://github.com/EESIZ/clawdreamer) 쪽을 쓰면 된다.

[English](README.md)

## 왜 Hermes에는 따로 만들어야 했나

Hermes는 기억하는 방식 자체가 openclaw랑 다르다. 기본 메모리는 일부러 작게 만들어져 있다:

| 파일 | 내용 | 기본 한도 |
|------|------|-----------|
| `~/.hermes/memories/MEMORY.md` | 에이전트 자기 노트 (환경, 컨벤션, 교훈) | 2,200자 |
| `~/.hermes/memories/USER.md` | 사용자 프로필 (선호, 말투) | 1,375자 |

두 파일은 매 세션 시작할 때 시스템 프롬프트에 통째로 들어간다. 대화 원본은 전부 `~/.hermes/state.db`(SQLite + FTS5)에 쌓이고, 필요하면 `session_search`로 꺼내 본다.

뇌로 치면 해마(`state.db`, 날것의 에피소드)도 있고 작은 신피질(`MEMORY.md`/`USER.md`, 항상 켜져 있는 지식)도 이미 있다. 근데 **잠**이 없다.
하루치 대화에서 중요한 걸 골라 옮기고, 바뀐 사실을 합치고, 파일이 꽉 찼을 때 뭘 잊을지 정하는 과정이 따로 없다. 지금은 에이전트가 대화 도중에, 다른 일 하면서 알아서 적어야 한다.

HermesYume이 그 꿈을 맡는다.

## 작동 원리

Dreamer랑 똑같이 매일 밤 3단계를 거친다. 다른 점은 입력이 마크다운 파일이 아니라 `state.db`고, 출력이 LanceDB가 아니라 `MEMORY.md`/`USER.md`라는 것.

```
~/.hermes/state.db  (sessions + messages, 읽기 전용)
        │
        ▼
   ┌─────────┐
   │  NREM   │  끝난 세션 → 대화 단위 분할 → 임베딩 → 클러스터링 → LLM이 오래 갈 사실만 추출
   └────┬────┘  ("memory" / "user"로 분류)
        ▼
   ┌─────────┐
   │   REM   │  새 사실 vs 가장 비슷한 기존 항목:
   └────┬────┘    duplicate → 강화 · state_change → 병합 (최신 우선, "(prev: …)")
        │         different_aspects → 통합 · unrelated → 추가
        │       항상성: 예산 초과 시 → 긴 항목 압축 → 가장 약한 항목 망각
        ▼
~/.hermes/memories/MEMORY.md, USER.md  (Hermes와 같은 락으로 원자적 기록)
        │
        ▼
   Dream Log  (~/.hermesyume/dream-log/YYYY-MM-DD_HHMM.md)
```

### Phase 1: NREM -- "오늘 무슨 일이 있었지?"

NREM 수면 동안 해마는 하루의 사건들을 재생하고, 중요한 패턴만 골라 신피질로 전달한다. HermesYume도 같은 일을 한다:

- 지난 실행 이후 활동이 있었고, 마지막 활동이 30분 이상 지난 세션만 `state.db`에서 로드 (아직 대화 중인 세션을 도중에 꿈꾸면 곤란하니까). DB는 **읽기 전용**으로만 연다
- `user`/`assistant` 발화만 사용. 도구 출력(웹 페이지, 파일)은 기본 제외 -- 어디서 긁어온 페이지에 박힌 프롬프트 인젝션이 영구 기억이 되면 안 되니까. `cron` 세션도 기본 제외
- 압축된 세션은 압축 요약이 아니라 원본 메시지를 사용
- 대화를 교환 단위(사용자 발화 + 응답)로 분할
- 임베딩 유사도로 관련 대화를 클러스터링
- LLM이 클러스터마다 *오래 갈* 사실만 추출. 예산이 몇천 자밖에 안 돼서 "남길 거 없음"도 정상 답변

날것의 대화가 들어가서, 몇 줄짜리 사실이 나온다.

### Phase 2: REM -- "이건 내가 아는 것과 맞아?"

REM 수면은 새 기억과 기존 기억을 통합하는 시간이다 -- 모순을 해결하고, 연결을 강화한다. HermesYume의 REM 단계:

- **새** 사실과 **기존** 항목 사이만 비교 (예상 복잡도 : O(N*M)) ← 이건 Dreamer 때랑 똑같음. 여전히 이정도가 내 한계인듯
- 관계 분류: `duplicate` / `state_change` / `different_aspects` / `unrelated`
  - 임베딩 유사도만 보고 중복이라고 버리지 **않는다**. "Postgres 16"이랑 "Postgres 17"은 임베딩상 거의 똑같아서, 중복인지 상태 변경인지는 분류기한테 물어봐야 안다. Dreamer에도 있던 구멍인데 여기서 막았다
- **중복**: 기존 항목을 강화 (중요도 ↑)
- **상태 변경**: 하나로 병합 ("DB는 Postgres 17" + (prev: "Postgres 16"))
- **다른 측면**: 하나로 통합. 단, 합쳐서 오히려 길어지면 그냥 추가
- **시냅스 항상성**: Hermes 메모리는 크기가 정해져 있으니까, 뭔가를 기억하려면 뭔가는 잊어야 한다
  - 파일이 한도의 85%(`FILL_RATIO`)를 넘으면 → 먼저 LLM으로 긴 항목을 줄이고 → 그래도 넘치면 *감쇠된 중요도*가 제일 낮은 것부터 내보낸다
  - 중요도는 같은 얘기가 다시 나오면 올라가고, 안 나오면 하루하루 조금씩 떨어진다
  - 에이전트가 직접 적어 둔 항목은 처음 발견될 때 중요도 0.7로 등록
  - 잊은 항목은 `memory-archive/forgotten.jsonl`에 남는다. 완전히 지우는 게 아니라 접근이 어려워질 뿐 (원본 대화도 `state.db`에 그대로 있음)
  - 남겨 둔 15%는 낮 동안 에이전트가 직접 `memory add` 할 자리
- 기록 전에 모든 항목을 프롬프트 인젝션·유출·비밀키 패턴으로 검사. Hermes의 `tools.threat_patterns`를 import할 수 있으면 그걸 쓰고, 아니면 내장된 일부 패턴을 쓴다

"저번 주에 설정 바꿨다고 했잖아요" 같은 일은 더 이상 없다. -- "아마도"

### Phase 3: Dream Log -- "오늘 밤 무슨 꿈을 꿨지?"

매 사이클마다 마크다운 리포트가 생성된다: 뭘 뽑았고, 뭘 추가/병합/통합/압축했고, 뭘 잊었고, 뭘 막았는지. 실행 전후 파일 크기까지. 에이전트의 기억 관리에 대한 투명한 기록.

## 돌아가는 Hermes를 건드려도 괜찮은가

에이전트가 쓰고 있는 파일을 밤에 몰래 고치는 거라, 이 부분은 신경 써서 만들었다:

- `state.db`는 `mode=ro`로만 연다. 절대 안 쓴다
- 메모리 파일은 Hermes랑 같은 방식으로 쓴다: `MEMORY.md.lock`에 배타적 `flock` → 임시 파일에 쓰고 → `os.replace`. 결과물이 `§` 포맷으로 정확히 왕복돼서 Hermes의 drift guard도 통과한다 (Hermes의 `MemoryStore`로 직접 읽어서 확인함)
- 계획은 스냅샷 기준으로 세우지만, 적용은 *지금* 파일에 한다. 그 사이 에이전트가 바꾼 항목은 건드리지 않고 넘어간다. 하드 한도를 넘게 되면 HermesYume이 추가한 것부터 되돌린다
- 쓰기 전에 매번 `~/.hermesyume/memory-archive/<timestamp>/`에 원본 백업
- Hermes는 세션 시작 시점의 메모리를 고정해서 쓰기 때문에, 바뀐 내용은 **다음 세션**부터 반영된다

## 빠른 시작

```bash
# 1. 의존성 설치 (pyyaml 하나뿐이고 이것도 선택. 핵심 코드는 표준 라이브러리)
pip install -r requirements.txt

# 2. 환경 설정
cp .env.example .env
# .env에 OpenAI API 키 입력 (또는 ollama 사용)

# 3. 점검 (state.db, 메모리 파일, 한도, 키)
python doctor.py

# 4. 리허설 (계획만 세우고 dream log만 남김, Hermes 메모리는 안 건드림)
python hermesyume.py --dry-run --verbose

# 5. 실행
python hermesyume.py --verbose
```

처음 며칠은 `--dry-run`으로 돌리면서 dream log를 직접 읽어보는 걸 추천.

Dreamer랑 달리 `session-flush`(새벽 2시 `/new` 자동 전송)가 필요 없다. Hermes는 메시지를 받는 즉시 `state.db`에 저장하니까.

### Hermes 프로필

`HERMES_HOME`을 프로필 디렉토리로 지정하고, 프로필마다 `HERMESYUME_HOME`을 따로 둔다:

```bash
HERMES_HOME=~/.hermes/profiles/work HERMESYUME_HOME=~/.hermesyume-work python hermesyume.py
```

## 크론잡 실행

```bash
# 예시: 매일 새벽 3시 실행
0 3 * * * /path/to/hermesyume/examples/run-hermesyume.sh
```

또는 `examples/` 디렉토리의 systemd timer를 사용.

## 설정

한도와 활성화 여부는 `$HERMES_HOME/config.yaml`에서 읽어온다 (`memory.memory_char_limit`, `memory.user_char_limit`, `memory.memory_enabled`, `memory.user_profile_enabled`).

나머지는 환경변수로 오버라이드 가능:

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `HERMES_HOME` | `~/.hermes` | Hermes 홈 (또는 프로필 디렉토리) |
| `HERMESYUME_HOME` | `~/.hermesyume` | 커서, 메타데이터, 로그, 아카이브 |
| `HERMESYUME_EMBEDDING_PROVIDER` | `openai` | `openai`, `ollama`, `sentence-transformers` |
| `HERMESYUME_LLM_PROVIDER` | `openai` | `openai` (OpenAI 호환 URL 전부), `ollama`, `minimax` |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | 예: OpenRouter |
| `HERMESYUME_OPENAI_LLM_MODEL` | `gpt-4.1-nano` | |
| `HERMESYUME_FILL_RATIO` | `0.85` | Hermes 한도 대비 채울 비율 |
| `HERMESYUME_DECAY_RATE` | `0.01` | 강화 없이 하루 지날 때마다 깎이는 중요도 |
| `HERMESYUME_FORGET_THRESHOLD` | `0` | 0보다 크면 예산이 남아도 희미해진 항목을 잊음 |
| `HERMESYUME_KEEP_PREV_STATE` | `true` | 상태 변경 병합 시 짧은 "(prev: …)" 유지 |
| `HERMESYUME_EXCLUDE_SOURCES` | `cron` | 제외할 세션 source (쉼표 구분) |
| `HERMESYUME_INCLUDE_TOOL_MESSAGES` | `false` | 도구 출력도 사용 (위험 증가) |
| `HERMESYUME_SESSION_SETTLE_SECONDS` | `1800` | 이보다 최근에 활동한 세션은 건너뜀 |
| `HERMESYUME_MAX_NEW_FACTS` | `12` | 사이클당 최대 신규 사실 수 |
| `HERMESYUME_ENTRY_MAX_CHARS` | `220` | 항목 하나의 최대 길이 |

### 에러 알림

HermesYume도 AI 에이전트가 모르게 백그라운드에서 돌아가는 프로세스다. 장애 발생 시 에이전트가 아닌 **운영자**에게 직접 알림을 보낸다.

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `HERMESYUME_ALERT_PROVIDER` | (비활성) | `telegram`, `slack`, `webhook` |
| `HERMESYUME_ALERT_TELEGRAM_BOT_TOKEN` | | 텔레그램 봇 토큰 |
| `HERMESYUME_ALERT_TELEGRAM_CHAT_ID` | | 알림 받을 텔레그램 채팅 ID |
| `HERMESYUME_ALERT_SLACK_WEBHOOK_URL` | | Slack incoming webhook URL |
| `HERMESYUME_ALERT_WEBHOOK_URL` | | 일반 webhook (POST JSON) |

### 디렉토리 구조

```
$HERMESYUME_HOME/
  state.json            # 세션 커서 (어디까지 꿈꿨는지)
  meta.json             # 항목별 중요도 / 강화 시점 / 임베딩 캐시
  dream-log/            # 출력: 매일 밤 정리 리포트
  memory-archive/
    <timestamp>/        # 쓰기 직전 MEMORY.md / USER.md 백업
    forgotten.jsonl     # 잊은 항목 전부 (점수 포함)
  episodes/             # 선택: 추가로 꿈꿀 마크다운 노트 (YYYY-MM-DD*.md)
```

## 한계

- 인젝션 검사는 패턴 기반이다. 멀쩡해 보이는 문장으로 바꿔 쓴 악성 지시는 통과할 수 있다. 진짜 방어선은 도구 출력을 아예 안 쓰는 것
- 품질은 LLM 따라간다. 작은 로컬 모델은 쓸데없는 걸 뽑거나 관계를 잘못 분류할 수 있다
- 중요도는 "다시 언급됐나 + 시간"으로만 매기는 휴리스틱이다. Hermes가 기록을 안 남겨서, 에이전트가 어떤 기억을 실제로 *써먹었는지*는 알 수 없다
- 외부 메모리 프로바이더(Honcho, Mem0 등)는 안 건드린다. 내장 파일만 다룬다

## 테스트

```bash
python -m unittest discover tests
```

네트워크 없이 가짜 Hermes 홈에서 전체 사이클을 돌린다. 상태 변경 병합, user/agent 분류, 인젝션 차단, 예산 초과 시 망각, 포맷 왕복, 동시 수정 처리를 확인.

## 요구 사항

- Python 3.10+
- 임베딩 제공자 (택 1):
  - OpenAI API 키 (`text-embedding-3-small`)
  - [Ollama](https://ollama.com) 로컬 실행 (`nomic-embed-text`)
  - `pip install sentence-transformers` (`all-MiniLM-L6-v2`)
- LLM 제공자 (택 1):
  - OpenAI API 키 또는 OpenAI 호환 엔드포인트 (`gpt-4.1-nano`)
  - [Ollama](https://ollama.com) 로컬 실행 (`qwen2.5:3b` 등)
  - MiniMax API 키

## Donations
If you find it useful...
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/V7V21XAPRC)

## 라이선스

MIT
