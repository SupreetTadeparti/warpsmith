from warpsmith.sanitize import parse_report

LOG_WITH_ERROR = """\
========= COMPUTE-SANITIZER
========= Invalid __global__ write of size 4 bytes
=========     at matmul(const float *, const float *, float *, int, int, int)+0x2b0 in kernel.cu:17
=========     by thread (0,0,0) in block (0,0,0)
=========     Address 0x7f3a10000000 is out of bounds
=========
========= ERROR SUMMARY: 1 error
"""


def test_parse_error_report():
    count, report = parse_report(LOG_WITH_ERROR)
    assert count == 1
    assert report.splitlines()[0].strip() == "Invalid __global__ write of size 4 bytes"
    assert "kernel.cu:17" in report
    assert "COMPUTE-SANITIZER" not in report


def test_parse_clean_report():
    assert parse_report("========= COMPUTE-SANITIZER\n========= ERROR SUMMARY: 0 errors\n")[0] == 0


def test_parse_missing_summary():
    assert parse_report("")[0] is None
