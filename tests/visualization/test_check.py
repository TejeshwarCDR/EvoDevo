from evodevo.visualization.check import (
    CheckResult,
    check_gifsicle_optional,
    check_imageio_gif_and_mp4,
    run_all,
)


def test_check_result_defaults_to_not_ok():
    r = CheckResult("some check")
    assert r.ok is False
    assert r.diagnosis is None


def test_check_result_passed_and_failed():
    r = CheckResult("x").passed()
    assert r.ok is True

    r2 = CheckResult("y").failed("go fix it")
    assert r2.ok is False
    assert r2.diagnosis == "go fix it"


def test_imageio_check_passes_in_this_environment():
    # imageio/imageio-ffmpeg are project dependencies (requirements.txt);
    # this should pass wherever the test suite itself runs.
    result = check_imageio_gif_and_mp4()
    assert result.ok, result.diagnosis


def test_gifsicle_check_never_fails_it_is_purely_informational():
    result = check_gifsicle_optional()
    assert result.ok  # always "passed" -- gifsicle is optional


def test_run_all_returns_four_checks_with_names():
    results = run_all()
    names = {r.name for r in results}
    assert len(results) == 4
    assert any("simulator" in n for n in names)
    assert any("render_mode" in n for n in names)
    assert any("imageio" in n for n in names)
    assert any("gifsicle" in n for n in names)
