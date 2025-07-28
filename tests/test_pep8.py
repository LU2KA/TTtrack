import pytest
from pylint.lint import Run

# inspired from BI-PYT homeworks
@pytest.fixture(scope="session")
def linter():
    args = [
        "./src",
        "app.py",
        "--disable=C0301,C0103,E1101",
        "-sn",
    ]
    results = Run(args, exit=False)
    return results.linter

def test_codestyle_score(linter):
    for msg in linter.reporter.messages:
        print(f"{msg.msg_id} ({msg.symbol}) line {msg.line}: {msg.msg}")

    score = linter.stats.global_note
    print(f"Pylint score = {score} Limit = 10")
    assert score >= 10, f"Pylint score {score} is below the limit 10"
