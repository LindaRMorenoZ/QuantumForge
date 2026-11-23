from quantumforge import __version__


def test_package_version_is_defined():
    assert isinstance(__version__, str)
    assert __version__
