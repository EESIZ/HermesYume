"""vecutil: cosine helpers, numeric/negation guards (T7 numeric guard), fact preservation (T10)."""

import numpy as np

from hermesyume import vecutil as vu


def test_cos_and_topk():
    a = np.array([1, 0, 0], np.float32)
    b = np.array([1, 1, 0], np.float32)
    assert abs(vu.cos(a, b) - 0.7071) < 1e-3
    assert vu.cos(a, np.zeros(3)) == 0.0
    M = np.array([[0, 1, 0], [1, 0.1, 0], [1, 1, 0]], np.float32)
    top = vu.top_k(a, M, 2)
    assert [i for i, _ in top] == [1, 2]
    assert vu.top_k(a, M, 5, min_cos=0.8) == [(1, top[0][1])]
    assert vu.cos_matrix(M, M).shape == (3, 3)
    assert vu.top_k(a, np.zeros((0, 3), np.float32), 3) == []


def test_numeric_tokens_normalization():
    assert vu.numeric_tokens("포트는 8,081이다") == {"8081"}
    assert vu.numeric_tokens("마감은 2026-10-10이다") == {"2026-10-10"}
    assert vu.numeric_tokens("마감은 10월 10일") == {"10-10"}
    assert vu.numeric_tokens("2026년 10월 결산") == {"2026-10"}
    assert vu.numeric_tokens("2026년 10월 3일") == {"2026-10-03"}
    assert vu.numeric_tokens("비중 20.50%") == {"20.5"}
    assert vu.numeric_tokens("회의 09:00") == {"9:00"}
    assert vu.numeric_tokens("숫자 없음") == frozenset()


def test_auto_dup_numeric_and_negation_guard():
    a = "도서관 사물함 번호는 12번이다."
    b = "도서관 사물함 번호는 13번이다."
    assert not vu.auto_dup_eligible(a, b, "state", "state")                # T7: 12 vs 13
    assert vu.auto_dup_eligible(a, "도서관 사물함 12번을 배정받았다.", "state", "event")  # same family
    assert not vu.auto_dup_eligible("텔레그램을 보낸다", "텔레그램을 보내지 않는다", "rule", "rule")
    assert vu.has_negation("절대 하지 마라") and vu.has_negation("do not")
    assert not vu.auto_dup_eligible(a, a, "rule", "fact")                    # different family
    assert vu.same_family("rule", "procedure") and not vu.same_family("rule", "fact")


def test_strip_particle_and_fact_tokens():
    assert vu.strip_particle("포트는") == "포트"
    assert vu.strip_particle("규칙이다") == "규칙"
    assert vu.strip_particle("이다") == ""
    t = vu.fact_tokens("Orion 스테이징 서버 포트는 8081이다. 'rates_cli.py find' 사용")
    assert {"orion", "스테이징", "서버", "포트", "8081", "rates_cli.py find"} <= t
    assert "이다" not in t


def test_preservation_check():
    a = "Orion 데모 마감은 2026-10-10이다."
    b = "Orion 데모 발표자는 4조다."
    ok, missing = vu.preservation_check(a, b, "Orion 데모 마감은 2026-10-10이고 발표자는 4조다.")
    assert ok and missing == []
    ok, missing = vu.preservation_check(a, b, "Orion 데모는 4조가 발표한다.")
    assert not ok and "2026-10-10" in missing
    # date written differently still counts as preserved
    ok, _ = vu.preservation_check("마감은 10월 10일", "발표 4조", "마감 10월 10일, 발표는 4조")
    assert ok
