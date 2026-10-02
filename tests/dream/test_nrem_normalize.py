"""normalize.py (N5): subject_key, refs verification, importance formula (§5.3, D7)."""

from __future__ import annotations

import os

import pytest

from hermesyume.normalize import (compute_importance, normalize_claim, ref_candidates, subject_key,
                                  verify_refs)
from hermesyume.types import KIND_BASE, KINDS, Claim


def imp(kind="fact", level=3, explicit_user=False, usc=1, assistant_only=False, source="dream"):
    return compute_importance(kind=kind, level=level, explicit_user=explicit_user,
                              user_session_count=usc, assistant_only=assistant_only, source=source)


# ── importance §5.3 ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", KINDS)
def test_baseline_equals_kind_base(kind):
    assert imp(kind) == pytest.approx(KIND_BASE[kind])     # level 3, one user session, nothing else


@pytest.mark.parametrize("level,exp", [(1, 0.34), (2, 0.42), (3, 0.50), (4, 0.58), (5, 0.66)])
def test_level_term(level, exp):
    assert imp("fact", level) == pytest.approx(exp)


def test_explicit_and_sessions_and_assistant_terms():
    assert imp("decision", explicit_user=True) == pytest.approx(0.65 + 0.12)
    assert imp("fact", usc=2) == pytest.approx(0.55)
    assert imp("fact", usc=4) == pytest.approx(0.65)
    assert imp("fact", usc=50) == pytest.approx(0.65)        # min(usc-1, 3)
    assert imp("fact", usc=0) == pytest.approx(0.50)         # D7: no extra −0.05 without user evidence
    assert imp("event", usc=0, assistant_only=True) == pytest.approx(0.30)
    assert imp("rule", usc=0, assistant_only=True) == pytest.approx(0.75)   # U2 penalty kept


def test_full_formula_combination():
    v = imp("project", level=4, explicit_user=True, usc=3, assistant_only=False)
    assert v == pytest.approx(0.60 + 0.08 + 0.12 + 0.10)


def test_clamp_bounds():
    assert imp("rule", level=5, explicit_user=True, usc=4) == 1.0
    assert imp("legacy", level=1, usc=0, assistant_only=True) == 0.05
    assert imp("opinion", level=1, usc=0, assistant_only=True) == pytest.approx(0.09)


def test_source_floors():
    assert imp("fact", source="core:user") == 0.85
    assert imp("event", level=1, source="core:memory") == 0.85
    assert imp("fact", source="tool:yume_remember") == 0.80
    assert imp("rule", explicit_user=True, source="tool:yume_remember") == pytest.approx(0.97)
    assert imp("rule", level=5, explicit_user=True, source="core:user") == 1.0
    assert imp("fact", source="md") == 0.50


def test_importance_deterministic():
    assert imp("lesson", 4, True, 2, False) == imp("lesson", 4, True, 2, False)


# ── subject_key ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("subject,key", [
    ("Orion 결제: 스테이징!", "orion결제스테이징"),
    ("  Ｏｒｉ on  ", "orion"),                    # NFKC full-width
    ("🔁 BACKUP · BOT", "backupbot"),                       # symbols and punctuation dropped
    ("**가계부 관리:**", "가계부관리"),
    ("", ""),
])
def test_subject_key(subject, key):
    assert subject_key(subject) == key


# ── refs ─────────────────────────────────────────────────────────────────────

def test_ref_candidates():
    c = ref_candidates("자세한 건 docs/yume/a.md에 있고 rates_cli.py find, rate-lookup 스킬. 10/10 마감.")
    assert c[:3] == ["docs/yume/a.md", "rates_cli.py", "10/10"]
    assert "rate-lookup" in c and "find" in c


def test_verify_refs(paths, cfg, fake_home):
    ws = fake_home.workspace
    (ws / "docs" / "yume").mkdir(parents=True)
    (ws / "docs" / "yume" / "incident-report.md").write_text("x", encoding="utf-8")
    text = ("절차는 docs/yume/incident-report.md, 일지는 memory/2026-09-28-orion.md, "
            "없는 파일 docs/yume/none.md, 상위 ../../etc/passwd, 절대 /etc/passwd, "
            f"홈 {fake_home.root / 'SOUL.md'}, 키 {fake_home.root / '.env'}, ~/.hermes/SOUL.md, "
            f"작업공간 절대 {ws / 'docs'}/. 요금은 rates_cli 로, 그리고 rate-lookup, nonexistent-skill.")
    refs = verify_refs(text, paths=paths, workspace_dir=cfg.workspace_dir)
    assert refs == ["docs/yume/incident-report.md", "memory/2026-09-28-orion.md",
                    str((fake_home.root / "SOUL.md").resolve()), "docs",
                    "skills/rate-lookup/SKILL.md"]


def test_verify_refs_relative_to_home_and_symlink_escape(paths, cfg, fake_home, tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("secret-ish", encoding="utf-8")
    os.symlink(outside, fake_home.workspace / "link.md")
    refs = verify_refs("SOUL.md 와 link.md 와 config.yaml", paths=paths, workspace_dir=cfg.workspace_dir)
    assert str((fake_home.root / "SOUL.md").resolve()) in refs
    assert str((fake_home.root / "config.yaml").resolve()) in refs
    assert not any("outside" in r or "link.md" in r for r in refs)   # symlink out of the tree refused


def test_verify_refs_without_workspace(paths, fake_home):
    refs = verify_refs("스킬 rate-lookup 참고", paths=paths, workspace_dir="")
    assert refs == ["skills/rate-lookup/SKILL.md"]


# ── normalize_claim ──────────────────────────────────────────────────────────

def _claim(**kw) -> Claim:
    base = dict(origin_key="w#0", source="dream", kind="rule", target="user",
                subject="Orion 요금 확인", text="Orion 요금 질문은 항상 rate-lookup 요금표부터 확인한다.",
                level=4, explicit_user=True, evidence_roles=["user"], user_evidence_count=1,
                user_session_count=1)
    base.update(kw)
    return Claim(**base)


def test_normalize_claim_sets_fields(paths, cfg):
    c = normalize_claim(_claim(), cfg=cfg, paths=paths)
    assert c.subject_key == "orion요금확인"
    assert c.refs == ["skills/rate-lookup/SKILL.md"]
    assert c.importance == 1.0                       # 0.85 + 0.08 + 0.12 clamped


def test_normalize_claim_assistant_only_penalty(paths, cfg):
    c = normalize_claim(_claim(evidence_roles=["assistant"], user_evidence_count=0,
                               user_session_count=0, explicit_user=False, level=3), cfg=cfg, paths=paths)
    assert c.importance == pytest.approx(0.75)


def test_normalize_claim_never_lowers_and_keeps_refs(paths, cfg):
    c = _claim(kind="fact", level=3, explicit_user=False, importance=0.9, refs=["docs/x.md"])
    normalize_claim(c, cfg=cfg, paths=paths)
    assert c.importance == 0.9
    assert c.refs[0] == "docs/x.md" and "skills/rate-lookup/SKILL.md" in c.refs


def test_normalize_claim_empty_subject(paths, cfg):
    c = normalize_claim(_claim(subject="  "), cfg=cfg, paths=paths)
    assert c.subject and c.subject_key


def test_normalize_claim_remember_floor(paths, cfg):
    c = normalize_claim(_claim(kind="fact", level=2, explicit_user=False, source="tool:yume_remember"),
                        cfg=cfg, paths=paths)
    assert c.importance == 0.80
