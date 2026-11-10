import pytest

from app.services.names import same_name


@pytest.mark.parametrize(("first", "second"), [
    ("Magdy Abdelrehim Ibrahim Ali", "MAGDY ABDELREHIM IBRAHIM ALI"),
    ("Ali Magdy", "Magdy Ibrahim Ali"),                                   # order and middle names ignored
    ("ABD ALRAZAK MAMOUN BAROUDI", "Abdalrazak Mamoun Baroudi"),          # space on one side only
    ("Abdalrazak Mamoun Baroudi", "ABD ALRAZAK MAMOUN BAROUDI"),
    ("MOHAMMED ABDUL RAHMAN", "Mohammed Abdulrahman"),
    ("Abdul Rahman Ali Hassan", "AbdulRahman Ali"),
])
def test_same_name(first: str, second: str) -> None:
    assert same_name(first, second)


@pytest.mark.parametrize(("first", "second"), [
    ("Sara Ali", "Magdy Abdelrehim Ibrahim Ali"),
    ("Ali", "Ali"),                                                       # one part is not enough
    ("ABD ALRAZAK", "Abdalrazak"),                                        # joined, still one part
    ("Abdalrazak Mamoun Baroudi", "Abdalrazak Mamoun Hassan"),
    ("Mohammed Abdul", "Mohammed Abdulrahman"),                          # a part of a word is not the word
])
def test_different_names(first: str, second: str) -> None:
    assert not same_name(first, second)


def test_spacing_can_be_kept_strict() -> None:
    assert not same_name("ABD ALRAZAK MAMOUN BAROUDI", "Abdalrazak Mamoun Baroudi", ignore_spacing=False)
