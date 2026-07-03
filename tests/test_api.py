from ragops import __version__
from ragops.app import create_app


def test_package_version_is_declared() -> None:
    assert __version__


def test_health_route_is_registered() -> None:
    app = create_app()

    paths = {route.path for route in app.routes}

    assert "/health" in paths
