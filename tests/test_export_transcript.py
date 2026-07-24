from app.services.export_transcript import to_txt, write_docx


def _tl():
    return [
        {"start": 0.0, "end": 2.0, "text": "привет", "display_name": "Преподаватель"},
        {"start": 65.0, "end": 67.0, "text": "да", "display_name": "Ученик 1"},
    ]


def test_txt_structure():
    txt = to_txt(_tl())
    assert txt.startswith("Расшифровка урока")
    assert "[00:00] Преподаватель: привет" in txt
    assert "[01:05] Ученик 1: да" in txt


def test_docx_written_and_readable(tmp_path):
    out = write_docx(_tl(), tmp_path / "t.docx")
    assert out.exists()
    from docx import Document

    doc = Document(str(out))
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "Расшифровка урока" in text
    assert "Преподаватель" in text and "привет" in text
