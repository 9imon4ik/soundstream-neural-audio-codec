# SoundStream Neural Audio Codec

Нейроаудиокодек для сжатия и ресинтеза речи на основе [SoundStream](https://arxiv.org/abs/2107.03312).

<p align="center">
  <a href="https://arxiv.org/abs/2107.03312">Paper</a> ·
  <a href="https://huggingface.co/9imon4ik/soundstream-neural-audio-codec">Weights</a> ·
  <a href="notebooks/demo.ipynb">Demo</a> ·
  <a href="notebooks/analysis.ipynb">Analysis</a>
</p>

<p align="center">
  <a href="#о-проекте">О проекте</a> ·
  <a href="#архитектура">Архитектура</a> ·
  <a href="#обучение">Обучение</a> ·
  <a href="#установка">Установка</a> ·
  <a href="#использование">Использование</a> ·
  <a href="#инференс-и-демо">Инференс и демо</a> ·
  <a href="#структура-репозитория">Структура</a>
</p>

---

## О проекте

Пайплайн генератора:

```
audio (B, 1, T)
  → Encoder          (SEANet-подобный, strides [2, 4, 5, 5])
  → RVQ              (8 квантайзеров × codebook 1024, EMA; без готовых RVQ-библиотек)
  → Decoder          (симметричный upsampling)
  → reconstructed_audio (B, 1, T)
```

Обучение — с **adversarial** компонентой: два дискриминатора (waveform + STFT), реконструкция в mel-домене, commitment loss. На инференсе нужен только генератор.

| Параметр | Значение |
| --- | --- |
| Датасет | LibriSpeech `train-clean-100` / `test-clean` |
| Sample rate | 16 kHz, mono |
| Обучение | случайные кропы **0.5 с** (`is_train=True`) |
| Инференс / оценка | **полная** запись (`is_train=False`) |
| Предобученные веса | [9imon4ik/soundstream-neural-audio-codec](https://huggingface.co/9imon4ik/soundstream-neural-audio-codec) |
| Оценочный битрейт | ~**6 kbps** при 16 kHz |

---

## Архитектура

### Генератор

Реализация: `src/model/generator/` (`SoundStreamGenerator`).

| Компонент | Параметры |
| --- | --- |
| Strides энкодера | `[2, 4, 5, 5]` (↓200 по времени) |
| `hidden_channels` | 32 |
| `embedding_dim` | 128 |
| Residual blocks | dilations `[1, 3, 9]` |
| RVQ | 8 слоёв, codebook 1024, EMA decay 0.99 |
| Commitment loss | MSE(encoder, quantize.detach()), вес **1.0** |

### Дискриминаторы

`SoundStreamDiscriminator` (`src/model/discriminator/`):

1. **Multi-scale waveform** — свёртки по сырому сигналу, LeakyReLU slope **0.2**
2. **Multi-resolution STFT** — 2D-свёртки по спектрограммам (окно 1024, hop 256)

### Функции потерь (генератор)

| Loss | Вес | Описание |
| --- | ---: | --- |
| `loss_reconstruction` | 1.0 | multi-scale mel + log-mel |
| `loss_adversarial` | 1.0 | GAN / hinge по логитам D |
| `loss_feature_matching` | 100.0 | L1 по промежуточным фичам D |
| `loss_commitment` | 1.0 | привязка энкодера к квантованию |

В логах также пишется **codebook perplexity** — равномерность использования кодов.

Гиперпараметры модели и обучения: `src/configs/soundstream.yaml`.

---

## Обучение

### Данные

| Split | Путь после `download_librispeech.py` |
| --- | --- |
| Train | `data/LibriSpeech/LibriSpeech/train-clean-100` |
| Test | `data/LibriSpeech/LibriSpeech/test-clean` |

- Короткие utterance (< 0.5 с) на train дополняются padding `replicate`
- Кроп 0.5 с только при `is_train=True`; на test/eval — полный файл

### Цикл и гиперпараметры

На каждом шаге (`src/trainer/trainer.py`):

1. Обновление **дискриминатора** (real vs detached fake)
2. Обновление **генератора** (reconstruction + adversarial + feature matching + commitment)

| Параметр | Значение |
| --- | --- |
| Оптимизатор | Adam, lr `2e-4` (G и D отдельно) |
| Batch size | 12 |
| `epoch_len` | 450 итераций / эпоха |
| `n_epochs` | 100 |
| Шагов всего | ~45k (100 × 450) |
| Чекпоинты | каждые 5 эпох → `saved/<run_name>/` |
| Логирование | каждые 50 шагов + аудио в Comet ML |

Имя run и каталог сохранения: `src/configs/writer/cometml.yaml` (`run_name`, по умолчанию `soundstream-final`).

### Метрики

`evaluate.py` считает **STOI** и **NISQA** на `test-clean` (batch size 1, полные записи) и пишет агрегаты в Comet ML.

Перед запуском:

1. Чекпоинт: `saved/soundstream-final/checkpoint-epoch100.pth` (или свой run)
2. В `src/configs/evaluate.yaml` — `writer.run_id` вашего эксперимента в Comet

---

## Установка

### 1. Зависимости

```bash
pip install -r requirements.txt
pre-commit install   # опционально
```

### 2. Данные (обучение / evaluate)

```bash
python scripts/download_librispeech.py
```

Скачивает `train-clean-100` и `test-clean` с [OpenSLR](https://www.openslr.org/12) в `data/LibriSpeech/`.

### 3. Веса (инференс / evaluate без обучения)

```bash
python scripts/download_checkpoint.py
```

| | |
| --- | --- |
| Hub | [9imon4ik/soundstream-neural-audio-codec](https://huggingface.co/9imon4ik/soundstream-neural-audio-codec) |
| Файл | `checkpoint-epoch100.pth` |
| Путь | `saved/soundstream-final/checkpoint-epoch100.pth` |

### 4. Comet ML (обучение и evaluate)

```bash
export COMET_API_KEY=your_key
```

---

## Использование

### Обучение

```bash
python train.py
```

Переопределение параметров (Hydra CLI):

```bash
python train.py trainer.n_epochs=100 trainer.epoch_len=450 dataloader.batch_size=12
python train.py audio.crop_seconds=0.5 generator_optimizer.lr=2e-4
```

Продолжить с чекпоинта:

```bash
python train.py trainer.resume_from=saved/soundstream-final/checkpoint-epoch50.pth
```

Чекпоинт содержит `generator_state_dict` и `discriminator_state_dict`.

### Оценка качества

```bash
python evaluate.py
```

---

## Инференс и демо

Один и тот же пайплайн: загрузка весов (`evaluate.load_generator`) → `reconstruct_batch` → waveform 16 kHz mono.  
`inference.load_audio` принимает **локальный путь** или **HTTP(S) URL** к аудио.

### CLI — `inference.py`

```bash
python inference.py inferencer.input=audio.wav inferencer.output=out.wav
python inference.py inferencer.input=https://keithito.com/LJ-Speech-Dataset/LJ025-0076.wav
```

Без `inferencer.output` результат: `data/saved/inference/reconstructed.wav`.

| Параметр | По умолчанию |
| --- | --- |
| `inferencer.checkpoint` | `saved/soundstream-final/checkpoint-epoch100.pth` |
| `inferencer.input` | обязателен |
| `inferencer.output` | `data/saved/inference/reconstructed.wav` |

### Ноутбук — [`notebooks/demo.ipynb`](notebooks/demo.ipynb)

Для проверки в **Google Colab**: задать `AUDIO_URL`, выполнить все ячейки — original и reconstructed в плеере.

1. (Colab) раскомментировать clone / `pip install` / `download_checkpoint` в ячейке установки  
2. Загрузка модели (конфиг `inference`, те же функции, что в `evaluate.py`)  
3. `load_audio(AUDIO_URL)` → реконструкция → `Audio`

Пример URL: `https://keithito.com/LJ-Speech-Dataset/LJ025-0076.wav`

Отчёт по анализу (qualitative / quantitative): [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb).

---

## Структура репозитория

```
.
├── train.py                 # обучение
├── inference.py             # ресинтез файла или URL
├── evaluate.py            # STOI / NISQA на test-clean
├── requirements.txt
│
├── scripts/
│   ├── download_librispeech.py
│   └── download_checkpoint.py
│
├── notebooks/
│   ├── demo.ipynb         # URL → ресинтез (Colab)
│   └── analysis.ipynb     # текст отчёта + примеры
│
└── src/
    ├── configs/             # soundstream, inference, evaluate, …
    ├── datasets/
    ├── model/
    │   ├── generator/     # encoder, RVQ, decoder
    │   └── discriminator/
    ├── loss/
    ├── trainer/
    ├── logger/
    └── metrics/
```

---

## Ссылки

| Ресурс | URL |
| --- | --- |
| SoundStream (paper) | https://arxiv.org/abs/2107.03312 |
| LibriSpeech | https://www.openslr.org/12 |
| Веса (Hugging Face) | https://huggingface.co/9imon4ik/soundstream-neural-audio-codec |
| Comet (проект) | https://www.comet.com/ |
