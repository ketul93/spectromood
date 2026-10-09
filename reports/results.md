# Results

clef-flash, `plots+features+guide` setup, 50 clips per language (10 per emotion). Chance = 20%.

| language | dataset | baseline | accuracy | macro-F1 | arousal AUC | angry | fearful | happy | sad | neutral |
|---|---|---|---|---|---|---|---|---|---|---|
| English | RAVDESS | per speaker | **48%** | 0.39 | 0.93 | 90% | 0% | 50% | 0% | 100% |
| Russian | RESD | corpus | **40%** | 0.31 | 0.83 | 50% | 0% | 100% | 0% | 50% |
| Spanish | MESD | corpus | **26%** | 0.16 | 0.66 | 0% | 0% | 50% | 0% | 80% |

## Ablation (accuracy)

| language | model | spectrogram | plots | features | plots+features+guide |
|---|---|---|---|---|---|
| English | clef-flash | 20% | 20% | 40% | 48% |
| Russian | clef-flash | 20% | 20% | 38% | 40% |
| Spanish | clef-flash | 20% | 20% | 20% | 26% |
