from project_name import __version__
from project_name.cli import main


def test_example_fixture(example_fixture):
    assert example_fixture == 1


def test_version():
    assert __version__ == "0.1.0"


def test_cli(capsys):
    main()
    captured = capsys.readouterr()
    assert "Hi, I'm your cli!" in captured.out
