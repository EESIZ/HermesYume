"""Fake HERMES_HOME tree (stdlib only; usable from the Hermes venv).

Layout mirrors the live home: memories/MEMORY.md + USER.md in Hermes "\\n§\\n" format,
config.yaml (memory section), SOUL.md, skills/<name>/SKILL.md, .env (fake key), optional state.db,
plus a sibling workspace dir with memory/*.md and docs/. All content, counts and dates are synthetic
(22 USER entries incl. one header-only fragment, 13 episodic MEMORY entries = 35); only the shape
(a header-only fragment, episodic 'Session:' entries) follows §9 M1/M3.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

ENTRY_DELIMITER = "\n§\n"
FAKE_OPENAI_KEY = "sk-test-" + "0" * 32          # fake; matches the openai secret regex on purpose

# Input filters whose public defaults are empty (no deployment-specific values ship in config.py);
# the fixtures below (statedb.build_basic, MD_EPISODE, the md dir) exercise them, so tests opt in.
TEST_FILTERS = {
    "exclude_first_message_regex": r"^\[synthetic-eval\]",
    "deny_cwd_globs": ["/tmp/agent-probe*"],
    "md_exclude_globs": ["scratch-notes.md"],
    "strip_line_regex": [r"^🔁 \[BACKUP BOT\]", r"^💾 DISK"],
}

CONFIG_YAML = """\
model:
  default: deepseek-v4-pro
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200
  user_char_limit: 1375
  nudge_interval: 10
session_reset:
  mode: daily
  idle_minutes: 30
  at_hour: 4
"""

USER_ENTRIES: list[str] = [
    "**이름:** 테스트사용자",
    "**호칭:** 사장님",
    "**시간대:** Asia/Seoul",
    "**직업:** 프리랜서 디자이너",
    "**가계부 관리:** 지출 기록은 공용 가계부 DB를 단일 원장으로 사용한다.",
    "**일정 관리 원칙:** 반복 일정은 캘린더에, 일회성 할 일은 메모 앱에 등록한다.",
    "**Reading List:**",
    "**응답 스타일:** 결론 먼저, 짧게.",
    "**언어:** 한국어 기본, 코드 주석은 영어 허용.",
    "**알림 채널:** 텔레그램 DM.",
    "**금지:** 비밀값을 대화에 출력하지 않는다.",
    "**업무 시간:** 평일 09:00-18:00 KST.",
    "**주간 회의:** 월요일 10시.",
    "**택배 규칙:** 부재 중 택배는 경비실이 아니라 문 앞에 둔다.",
    "**크론 주의:** 야간 작업 결과는 아침에 한 번만 알린다.",
    "**데이터 원칙:** 공유 문서함이 단일 기준(SSOT).",
    "**운동 목표:** 주 3회 30분 이상 걷기.",
    "**확인 습관:** 날씨 질문은 도구로 먼저 확인.",
    "**메모 위치:** 일지는 workspace/memory에 남긴다.",
    "**호출 방식:** 짧은 질문은 바로 답한다.",
    "**표기:** 날짜는 YYYY-MM-DD.",
    "**검토:** 큰 변경은 먼저 계획을 보여준다.",
]

MEMORY_ENTRIES: list[str] = [
    f"Session: 2026-06-{(i % 2) + 27:02d} 대화 조각 {i}\nConversation Summary: 테스트용 레거시 세션 요약 {i}번."
    for i in range(13)
]

SKILLS = {"rate-lookup": "---\nname: rate-lookup\ndescription: Orion 요금표 조회 (rates_cli.py find)\n---\n본문\n"}

MD_EPISODE = """# Session: 2026-09-28 orion
Session Key: test
user: Orion 검수 당번은 7조가 맡는다.
user: Orion 장애 보고 절차는 알림 확인 → 로그 수집 → 원인 기록 → 회고 공유 순서다. 앞으로 이 순서를 지켜.
assistant: 알겠습니다.
🔁 [BACKUP BOT] 야간 백업 완료 3/3
"""


@dataclass
class FakeHome:
    root: Path               # HERMES_HOME
    workspace: Path          # workspace dir (md_sources = workspace/memory)

    @property
    def memories(self) -> Path: return self.root / "memories"
    @property
    def user_md(self) -> Path: return self.memories / "USER.md"
    @property
    def memory_md(self) -> Path: return self.memories / "MEMORY.md"
    @property
    def state_db(self) -> Path: return self.root / "state.db"
    @property
    def md_dir(self) -> Path: return self.workspace / "memory"


def write_core(path: str | os.PathLike, entries: list[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(ENTRY_DELIMITER.join(entries), encoding="utf-8")


def read_core(path: str | os.PathLike) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    return [e for e in (x.strip() for x in p.read_text(encoding="utf-8-sig").split(ENTRY_DELIMITER)) if e]


def make_hermes_home(base: str | os.PathLike, *, user_entries: list[str] | None = None,
                     memory_entries: list[str] | None = None, with_env: bool = True,
                     with_md: bool = True, env_text: str | None = None) -> FakeHome:
    """Create <base>/hermes (HERMES_HOME) and <base>/workspace. Never touches real homes."""
    base = Path(base)
    home = FakeHome(base / "hermes", base / "workspace")
    home.memories.mkdir(parents=True, exist_ok=True)
    write_core(home.user_md, USER_ENTRIES if user_entries is None else user_entries)
    write_core(home.memory_md, MEMORY_ENTRIES if memory_entries is None else memory_entries)
    (home.root / "config.yaml").write_text(CONFIG_YAML, encoding="utf-8")
    (home.root / "SOUL.md").write_text("# SOUL\n테스트 에이전트\n", encoding="utf-8")
    for name, body in SKILLS.items():
        d = home.root / "skills" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(body, encoding="utf-8")
    if with_env:
        env = env_text if env_text is not None else f"OPENAI_API_KEY={FAKE_OPENAI_KEY}\n"
        (home.root / ".env").write_text(env, encoding="utf-8")
        os.chmod(home.root / ".env", 0o600)
    home.workspace.mkdir(parents=True, exist_ok=True)
    (home.workspace / "docs").mkdir(exist_ok=True)
    if with_md:
        home.md_dir.mkdir(parents=True, exist_ok=True)
        (home.md_dir / "archive").mkdir(exist_ok=True)
        (home.md_dir / "2026-09-28-orion.md").write_text(MD_EPISODE, encoding="utf-8")
        (home.md_dir / "scratch-notes.md").write_text("## 임시 메모\n- 제외 대상 파일\n", encoding="utf-8")
    return home


def tree_hash(root: str | os.PathLike, *, exclude_dirs: tuple[str, ...] = ("dream-log", "runs")) -> dict[str, str]:
    """{relpath: sha256} of every regular file under root (lock files included), skipping any
    path with a component in exclude_dirs. For dry-run purity tests (T19/G10)."""
    root = Path(root)
    out: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        if any(part in exclude_dirs for part in rel_dir.parts):
            continue
        for fn in filenames:
            p = Path(dirpath) / fn
            if not p.is_file() or p.is_symlink():
                continue
            out[str(rel_dir / fn)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out
