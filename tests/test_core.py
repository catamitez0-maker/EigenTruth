"""EigenTruth 包级别冒烟测试。"""

from eigentruth import __version__


def test_version():
    assert __version__ == "0.1.0"
