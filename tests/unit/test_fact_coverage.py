"""사실 숫자 포함률 (확장 브랜치): 쉼표·공백을 무시하고 한 자리 정수는 세지 않는다."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import fact_coverage_ext as fc  # noqa: E402


def test_commas_inside_numbers_do_not_matter():
    # 표에는 22000, 사실 문장에는 22,000이라고 적혀 있어도 같은 숫자다
    assert fc.coverage("운영서비스 22000 26000", ["운영서비스 22,000, 26,000"]) == 1.0
    assert fc.coverage("매출 1,160억원", ["매출은 1160억원"]) == 1.0


def test_single_digit_integers_are_ignored_but_two_digit_numbers_count():
    assert fc.fact_numbers("1분기 매출은 60,000백만원") == ["60000백만원"]
    assert fc.fact_numbers("2분기") == []
    assert fc.coverage("아무 숫자도 없음", ["1분기와 2분기"]) is None


def test_coverage_is_the_fraction_of_numeric_facts_fully_present_and_none_without_numbers():
    facts = ["매출은 1,160억원에서 1,280억원으로 늘었다.", "영업이익은 100억원에서 96억원으로 줄었다.", "두 지표가 모두 늘어난 것은 아니다."]
    assert fc.coverage("1,280억원 96억원 1,160억원", facts) == 0.5  # 첫째 사실만 숫자가 모두 있음
    assert fc.coverage("전혀 관계없는 글", facts) == 0.0
    assert fc.coverage("x", ["숫자가 없는 사실"]) is None
