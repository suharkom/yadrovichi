from app.services import user_dict


def _reset(tmp_path, monkeypatch):
    monkeypatch.setattr(user_dict, "OVERRIDES_PATH", tmp_path / "replacements.json")
    monkeypatch.setattr(user_dict, "_cache", None)


def test_add_and_get(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    user_dict.add_replacement("проржа", "маржа")
    assert user_dict.user_overrides() == {"проржа": "маржа"}
    # итоговый словарь содержит и базу, и оверрайд
    merged = user_dict.get_replacements()
    assert merged["проржа"] == "маржа"


def test_override_applied_in_postprocessing(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    from app.services.text_postprocessing import apply_text_replacements

    user_dict.add_replacement("джинви", "GMV")
    assert apply_text_replacements("один рубль джинви") == "один рубль GMV"


def test_remove(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    user_dict.add_replacement("абв", "где")
    user_dict.remove_replacement("абв")
    assert user_dict.user_overrides() == {}


def test_empty_wrong_ignored(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    user_dict.add_replacement("  ", "нечто")
    assert user_dict.user_overrides() == {}
