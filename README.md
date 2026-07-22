# yadrovichi

Локальный сервис: аудио или видео урока → текст, разбитый по ролям и таймкодам.
Кто говорил, когда и что сказал. Всё на своём железе, без облака.

## Что делаем

1. **Диаризация** — pyannote по всему файлу целиком → сегменты со спикерами
2. **Транскрибация** — faster-whisper large-v3-turbo (float16) → слова с таймкодами
3. **Склейка** — слова к спикерам по перекрытию, реплики, сглаживание коротких
   ложных переключений `A→B→A`
4. **Роли** — преподаватель (граф переходов) + ученик 1, 2, 3…
5. **Постобработка** — словарь замен частых ошибок ASR
6. **Выдача** — FastAPI + Gradio: батчем или потоком по мере готовности

Вход — и аудио, и **видео**: из видео ffmpeg вытягивает звуковую дорожку,
дальше распознавание то же.

## Результаты

| Файл | Длит. | Диаризация | ASR | Пайплайн | Обработка |
|---|---|---|---|---|---|
| Семинар (аудио) | 30 мин | 0.030 | 0.031 | 0.062 | ~2 мин |
| Урок STEM (видео) | 1 ч 31 мин | 0.034 | 0.084 | 0.127 | ~11,5 мин |

Бюджет RTF 0.4 — укладываемся втрое-вшестеро. pyannote 1,5 часа держит,
в память не упирается.

## Железо

Целевой сервер — **RTX A4000** (Ampere, sm_86, 16 ГБ, CUDA 13.x). Куратор
заявлял GTX 1080, но в машине A4000 — отсюда `float16`, а не `int8` (int8
нужен только на Pascal).

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
python -m scripts.smoke data/lesson.mp4

# Сервис: /docs — API, /ui — интерфейс. Один воркер: видеокарта одна.
PARALLEL_GPU_STAGES=false uvicorn app.main:app --workers 1 --port 8000

# Юнит-тесты склейки и определения ролей (без GPU)
python -m pytest tests/ -q
```

Ручки API:
- `POST /stream` — потоковая выдача NDJSON: meta → реплики по мере готовности → done
- `POST /transcribe` — батч, весь результат одним JSON
- `GET /ui` — Gradio: таймлайн заполняется вживую, цвет по роли, счётчик RTF

## Структура

```
app/core/config.py           настройки из .env (модели, устройство, лимит файла)
app/services/asr.py          faster-whisper, ленивая загрузка, RTF по стадии
app/services/diarization.py  pyannote (аудио через soundfile, в обход torchcodec)
app/services/alignment.py    привязка слов, реплики, сглаживание переключений
app/services/role_detection.py  преподаватель по графу + маркерам (батч-путь)
app/services/text_postprocessing.py  словарь замен ошибок ASR
app/services/mathnorm.py     формулы речью → символьный вид для LLM (не подключён)
app/services/pipeline.py     батч-оркестратор: диаризация → ASR → склейка → роли
app/services/streaming.py    потоковый пайплайн: граф-роли → потоковый ASR
app/main.py                  FastAPI: /stream, /transcribe, приём файла до лимита
app/ui.py                    Gradio на /ui: живой таймлайн, цвет по роли, JSON
configs/                     словари маркеров преподавателя и замен ошибок ASR
scripts/smoke.py             прогон пайплайна на файле
tests/                       юнит-тесты alignment и role_detection
```

## Договорённости

- Кластеризация **одна на весь файл**, не почанково
- `min_speakers=2`, верхнюю границу не задавать
- Аудио, видео и `.env` в гит не коммитить
- `compute_type=float16` (Ampere), не int8
- Стадии на этой карте гоняем **последовательно** (`PARALLEL_GPU_STAGES=false`):
  параллель конфликтует по NVML, а RTF и так втрое под бюджетом
- Лимит размера файла — `MAX_FILE_SIZE_MB` в `.env` (по умолчанию 4 ГБ, под видео)
- DER не заявляем как метрику: эталона с таймкодами нет

## Открытые задачи

- **Роли (батч):** в `role_detection.detect_teacher` преподаватель — взвешенная
  сумма (маркеры + граф + доля речи), роль скачет на пограничных файлах. МЛщик
  делает граф основным. Стриминг уже считает граф-первично (`streaming.py`).
- **Формулы:** `mathnorm.py` готов, но не встроен в `text_postprocessing`.
- **WER:** на записи с субтитрами, оба текста через нормализацию.
