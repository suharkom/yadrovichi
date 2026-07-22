# yadrovichi

Локальный сервис: запись урока → текст, разбитый по ролям и таймкодам.
Кто говорил, когда и что сказал. Всё на своём железе, без облака.

## Что делаем

1. **Диаризация** — pyannote по всему файлу целиком → сегменты со спикерами
2. **Транскрибация** — faster-whisper large-v3-turbo (float16) → слова с таймкодами
3. **Склейка** — слова к спикерам по перекрытию, реплики, сглаживание коротких
   ложных переключений `A→B→A`
4. **Роли** — преподаватель (граф переходов + маркеры) + ученик 1, 2, 3…
5. **Постобработка** — словарь замен частых ошибок ASR
6. **Выдача** — FastAPI (`/docs`) + Gradio (`/ui`)

Диаризация и ASR на A4000 идут **параллельно** (обе модели живут в 16 ГБ),
полный RTF ближе к максимуму стадий, а не к их сумме.

## Железо

Целевой сервер — **RTX A4000** (Ampere, sm_86, 16 ГБ, драйвер 595, CUDA 13.2).
Куратор заявлял GTX 1080, но в машине A4000 — отсюда `float16`, а не `int8`
(int8 нужен только на Pascal). Замеры на 30-мин файле: ASR RTF ~0.03,
диаризация ~0.03, пайплайн ~0.06 при бюджете 0.4.

## Установка

Нужен **ffmpeg** в PATH.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
cp .env.example .env      # вписать HF_TOKEN, принять условия на pyannote/*
python -c "import torch; print(torch.cuda.get_device_capability(0))"   # (8, 6)
```

## Запуск

```bash
# Прогон пайплайна на файле → JSON с таймлайном, ролями и RTF
python -m scripts.smoke data/audio.mp3

# Сервис: /docs — API, /ui — интерфейс
uvicorn app.main:app --workers 1   # строго один воркер: видеокарта одна

# Юнит-тесты склейки и определения ролей (без GPU)
python -m pytest tests/ -q
```

## Структура

```
app/core/config.py           настройки из .env (модели, устройство, float16)
app/services/asr.py          faster-whisper, RTF по стадии
app/services/diarization.py  pyannote, сегменты со спикерами
app/services/alignment.py    привязка слов, реплики, сглаживание переключений
app/services/role_detection.py  преподаватель по графу + маркерам, нумерация
app/services/text_postprocessing.py  словарь замен ошибок ASR
app/services/mathnorm.py     формулы речью → символьный вид для LLM (не подключён)
app/services/pipeline.py     оркестратор: ASR ‖ диаризация → склейка → роли
app/main.py                  FastAPI: приём файла до 200 МБ, /docs
app/ui.py                    Gradio на /ui: таймлайн, цвет по роли, RTF, JSON
configs/teacher_markers.py   словарь маркеров преподавателя + веса
configs/text_replacements.py словарь замен ошибок ASR
scripts/smoke.py             прогон пайплайна на файле
tests/                       юнит-тесты alignment и role_detection
```

## Договорённости

- Кластеризация **одна на весь файл**, не почанково
- `min_speakers=2`, верхнюю границу не задавать
- Аудио и `.env` в гит не коммитить
- `compute_type=float16` (Ampere), не int8
- DER не заявляем как метрику: эталона с таймкодами нет
- Колаб только для отладки — цифры RTF берём с сервера (A4000)

## Открытые задачи

- **Роли:** определение преподавателя — взвешенная сумма (маркеры + граф +
  доля речи). На пограничном семинаре роль скачет между прогонами. Сделать
  граф основным сигналом, маркеры/долю речи — только на спорных случаях.
- **Потоковая выдача:** пайплайн батчевый; для отдачи по мере готовности
  превратить `pipeline.process` в генератор, подключить к StreamingResponse.
- **Формулы:** `mathnorm.py` готов, но не встроен в `text_postprocessing`.
- **WER:** на записи с субтитрами, оба текста через нормализацию.
