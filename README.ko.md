# HermesYume

**Hermes + Yume(夢, 꿈)**: [Hermes Agent](https://github.com/NousResearch/hermes-agent)를 위한 수면 중 기억 정리기.

> AI 에이전트는 잠을 자지 않는다. 꿈도 꾸지 않는다. → 이게 사실 AI 에이전트의 가장 큰 문제

OpenClaw용으로 만들었던 [Dreamer (clawdreamer)](https://github.com/EESIZ/clawdreamer)를 Hermes로 옮긴 버전이다.
아이디어는 그대로다. 꿈은 (가설상) 뇌가 하루치 기억을 압축하고 정리하는 과정이니, 에이전트도 매일 밤 같은 과정을 거치게 하자.

[English](README.md)

## 왜 Hermes에 필요한가

Hermes의 기본 메모리는 의도적으로 작고 선별적이다.

| 파일 | 내용 | 기본 한도 |
|------|------|-----------|
| `~/.hermes/memories/MEMORY.md` | 에이전트 자신의 노트 (환경, 컨벤션, 교훈) | 2,200자 |
| `~/.hermes/memories/USER.md` | 사용자 프로필 (선호, 말투) | 1,375자 |

두 파일 모두 매 세션 시작 시 시스템 프롬프트에 들어간다. 대화 전체는 `~/.hermes/state.db`(SQLite + FTS5)에 남고, 필요할 때 `session_search`로 꺼내 볼 수 있다.

즉 Hermes에는 해마(`state.db`, 원본 에피소드)와 작은 신피질(`MEMORY.md`/`USER.md`, 항상 켜진 지식)이 이미 있다. 빠진 것은 **수면**이다. 하루 중 중요한 내용을 한쪽에서 다른 쪽으로 옮기고, 낡은 사실을 병합하고, 파일이 가득 찼을 때 무엇을 잊을지 정하는 과정이 따로 없다. 지금은 에이전트가 대화 도중에, 다른 일을 하면서 직접 써야 한다.

HermesYume이 그 오프라인 과정을 맡는다.

## 작동 원리

```
~/.hermes/state.db  (sessions + messages, 읽기 전용)
        │
        ▼
   ┌─────────┐
   │  NREM   │  끝난 세션 → 대화 교환 단위 → 임베딩 → 클러스터 → LLM이 오래 갈 사실만 추출
   └────┬────┘  ("memory" 또는 "user"로 분류)
        ▼
   ┌─────────┐
   │   REM   │  각 사실을 가장 가까운 기존 항목과 비교:
   └────┬────┘    duplicate → 강화 · state_change → 병합(최신 우선, "(prev: …)")
        │         different_aspects → 통합 · unrelated → 추가
        │       항상성: 예산 초과 시 → 긴 항목 압축 → 가장 약한 항목 망각
        ▼
~/.hermes/memories/MEMORY.md, USER.md  (Hermes와 같은 락으로 원자적으로 기록)
        │
        ▼
   Dream Log  (~/.hermesyume/dream-log/YYYY-MM-DD_HHMM.md)
```

### Phase 1: NREM: "오늘 무슨 일이 있었지?"

- 지난 실행 이후에 활동이 있었고, 마지막 활동이 30분 이상 지난 세션만 `state.db`에서 읽는다. 진행 중인 대화를 도중에 꿈꾸지 않기 위해서다. DB는 **읽기 전용**으로 연다.
- 기본값으로는 `user`/`assistant` 발화만 쓴다. 도구 출력(웹 페이지, 파일)은 신뢰할 수 없고, 가져온 페이지에 들어 있던 프롬프트 인젝션이 영구 기억이 되면 안 되기 때문이다. `cron` 세션도 기본으로 제외한다.
- 압축된 세션은 압축 요약이 아니라 원본 메시지를 사용한다.
- 대화를 교환 단위(사용자 발화 + 응답)로 나누고 임베딩해 비슷한 것끼리 묶은 뒤, LLM에게 *오래 유효할* 사실만 뽑게 한다. 예산이 수천 자뿐이므로 "없음"도 정상 답변이다.

### Phase 2: REM: "이건 내가 아는 것과 맞나?"

- 새 사실은 기존 항목하고만 비교한다(O(N·M)). LLM이 가장 가까운 항목과의 관계를 `duplicate` / `state_change` / `different_aspects` / `unrelated`로 분류한다.
  - 임베딩 유사도만으로 중복을 버리지 **않는다**. "Postgres 16"과 "Postgres 17"은 임베딩상 거의 같아서, 중복인지 상태 변경인지는 분류기만 구분할 수 있다. 원래 Dreamer에도 있던 문제를 여기서 고쳤다.
- **시냅스 항상성**: Hermes 메모리는 크기가 정해져 있어 잊는 과정이 필요하다. 파일이 한도의 `FILL_RATIO`(기본 85%)를 넘으면 먼저 LLM으로 긴 항목을 줄이고, 그래도 넘으면 *감쇠된 중요도*가 가장 낮은 항목부터 내보낸다. 중요도는 같은 사실이 다시 나오면 올라가고, 그 뒤로 날짜만큼 선형으로 떨어진다. 에이전트가 직접 쓴 항목은 처음 발견될 때 중요도 0.7로 등록한다. 내보낸 항목은 `memory-archive/forgotten.jsonl`에 남는다. 남겨 둔 15%는 낮 동안 에이전트가 직접 `memory add`할 공간이다.
- 모든 후보 항목은 기록 전에 프롬프트 인젝션·유출·비밀키 패턴 검사를 거친다. Hermes의 `tools.threat_patterns`를 import할 수 있으면 그것을 쓰고, 없으면 내장된 일부 패턴을 쓴다.

### Phase 3: Dream Log: "무슨 꿈을 꿨지?"

실행마다 마크다운 리포트가 남는다. 추출한 사실, 추가/병합/통합/압축/망각/차단된 항목, 실행 전후 크기가 들어 있다.

## 실행 중인 Hermes와의 안전성

- `state.db`는 `mode=ro`로 열며 절대 쓰지 않는다.
- 메모리 파일은 Hermes와 같은 방식으로 쓴다. `MEMORY.md.lock`에 배타적 `flock`을 걸고, 임시 파일에 쓴 뒤 `os.replace`한다. 출력은 `§` 포맷으로 정확히 왕복되므로 Hermes의 drift guard를 통과한다. Hermes의 `MemoryStore`로 결과물을 직접 읽어서 확인했다.
- 계획은 스냅샷 기준으로 세우지만 적용은 *현재* 파일에 한다. 그 사이 에이전트가 바꾼 항목에 대한 작업은 건너뛴다. 하드 한도를 넘게 되면 우리가 추가한 것부터 되돌린다.
- 매번 쓰기 전에 `~/.hermesyume/memory-archive/<timestamp>/`에 원본을 백업한다.
- Hermes는 세션 시작 시점의 메모리를 고정해서 쓰므로, 변경 사항은 **다음 세션**부터 반영된다.

## 빠른 시작

```bash
pip install -r requirements.txt      # pyyaml(선택)뿐, 핵심 코드는 표준 라이브러리
cp .env.example .env                 # OPENAI_API_KEY 설정 또는 ollama 사용
python doctor.py                     # state.db, 메모리 파일, 한도, 키 점검
python hermesyume.py --dry-run -v     # 계획만 세우고 dream log만 기록
python hermesyume.py -v               # 실제 실행
```

cron이나 `examples/`의 systemd 유닛으로 매일 밤 실행한다.

```bash
0 3 * * * /path/to/hermesyume/examples/run-hermesyume.sh
```

OpenClaw 버전과 달리 `session-flush` 단계가 필요 없다. Hermes는 메시지를 받는 즉시 `state.db`에 저장한다.

### Hermes 프로필

`HERMES_HOME`을 프로필 디렉터리로 지정하고, 프로필마다 `HERMESYUME_HOME`을 따로 둔다.

```bash
HERMES_HOME=~/.hermes/profiles/work HERMESYUME_HOME=~/.hermesyume-work python hermesyume.py
```

## 설정

한도와 활성화 여부는 `$HERMES_HOME/config.yaml`에서 읽는다(`memory.memory_char_limit`, `memory.user_char_limit`, `memory.memory_enabled`, `memory.user_profile_enabled`).

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `HERMES_HOME` | `~/.hermes` | Hermes 홈(또는 프로필 디렉터리) |
| `HERMESYUME_HOME` | `~/.hermesyume` | 커서, 메타데이터, 로그, 아카이브 |
| `HERMESYUME_EMBEDDING_PROVIDER` | `openai` | `openai`, `ollama`, `sentence-transformers` |
| `HERMESYUME_LLM_PROVIDER` | `openai` | `openai`(OpenAI 호환 URL 모두), `ollama`, `minimax` |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | 예: OpenRouter |
| `HERMESYUME_OPENAI_LLM_MODEL` | `gpt-4.1-nano` | |
| `HERMESYUME_FILL_RATIO` | `0.85` | Hermes 한도 대비 채울 비율 |
| `HERMESYUME_DECAY_RATE` | `0.01` | 강화 없이 하루 지날 때마다 줄어드는 중요도 |
| `HERMESYUME_FORGET_THRESHOLD` | `0` | 0보다 크면 예산이 남아도 희미해진 항목을 잊음 |
| `HERMESYUME_KEEP_PREV_STATE` | `true` | 상태 변경 병합 시 짧은 "(prev: …)" 유지 |
| `HERMESYUME_EXCLUDE_SOURCES` | `cron` | 제외할 세션 source (쉼표 구분) |
| `HERMESYUME_INCLUDE_TOOL_MESSAGES` | `false` | 도구 출력도 사용 (위험 증가) |
| `HERMESYUME_SESSION_SETTLE_SECONDS` | `1800` | 이보다 최근에 활동한 세션은 건너뜀 |
| `HERMESYUME_MAX_NEW_FACTS` | `12` | 실행당 최대 신규 사실 수 |
| `HERMESYUME_ENTRY_MAX_CHARS` | `220` | 항목 하나의 최대 길이 |
| `HERMESYUME_ALERT_PROVIDER` | (꺼짐) | `telegram`, `slack`, `webhook`. 에러는 에이전트가 아닌 운영자에게만 보낸다 |

선택 사항: `$HERMESYUME_HOME/episodes/`에 `YYYY-MM-DD*.md` 형식의 마크다운 노트를 넣으면 함께 처리한 뒤 아카이브한다.

## 한계

- 위협 검사는 패턴 기반이다. 무해해 보이는 문장으로 바꿔 쓴 악성 지시는 통과할 수 있다. 주된 방어선은 도구 출력을 제외하는 것이다.
- 품질은 LLM에 달려 있다. 작은 로컬 모델은 사소한 내용을 뽑거나 관계를 잘못 분류할 수 있다. 처음 며칠은 `--dry-run`으로 돌리면서 dream log를 직접 확인하는 것을 권한다.
- 중요도는 재언급과 시간에 기반한 휴리스틱이다. Hermes가 기록하지 않으므로, 에이전트가 어떤 항목을 실제로 *사용했는지*는 알 수 없다.
- 외부 Hermes 메모리 프로바이더(Honcho, Mem0 등)는 건드리지 않는다. 내장 파일만 다룬다.

## 테스트

```bash
python -m unittest discover tests
```

네트워크 없이 가상의 Hermes 홈에서 전체 사이클을 돌린다. 상태 변경 병합, user/agent 분류, 인젝션 차단, 예산 초과 시 망각, 포맷 왕복, 동시 수정 처리를 확인한다.

## Donations
If you find it useful...
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/V7V21XAPRC)

## License

MIT
