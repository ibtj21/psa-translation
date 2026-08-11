# PSA Translation — English/Kiswahili → Ekegusii

Two fine-tuned models for low-resource machine translation into Ekegusii, plus a
Streamlit demo app.

## Repo structure

```
├── mt5_training.ipynb        # trains mT5-small, saves checkpoints/mt5_combined_guz/final
├── nllb_training.ipynb       # trains NLLB-200-distilled-600M, saves checkpoints/nllb_combined_guz/final
├── Final_merged_psas.csv     # curated PSA dataset (~21k rows)
├── app.py                    # Streamlit deployment demo
├── requirements.txt          # deployment app dependencies
└── checkpoints/               # created by the training notebooks (not checked in if large)
    ├── mt5_combined_guz/final/
    └── nllb_combined_guz/final/
```

## Running the training notebooks

Each notebook is self-contained: install deps, load `Final_merged_psas.csv`, train,
evaluate, and save a ready-to-use model under `checkpoints/<model>/final/`. No Colab,
Kaggle, or Weights & Biases account needed — all logs/metrics/hyperparameters are
written to local files under `logs/`.

1. Place `Final_merged_psas.csv` alongside the notebook (or edit `DATA_PATH`).
2. Run top to bottom. Training time is roughly 2-3 hours for mT5-small and longer for
   NLLB-200-distilled-600M on a single mid-range GPU (T4-class); an A100 will be
   noticeably faster, and the NLLB notebook's conservative batch-size/gradient-checkpointing
   settings can likely be relaxed on a larger GPU — see the note in that notebook's intro cell.
3. Outputs after a full run:
   - `checkpoints/<model>/final/` — the fine-tuned model, ready to load with
     `AutoModelForSeq2SeqLM.from_pretrained(...)`
   - `logs/<model>_training_log.csv` — per-epoch loss/BLEU/chrF
   - `logs/<model>_results.json` / `logs/<model>_results_table.csv` — zero-shot vs
     few-shot comparison
   - `logs/<model>_hyperparameters.csv` — hyperparameters + training time, for the
     write-up

## Running the deployment app

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app expects `checkpoints/mt5_combined_guz/final` and `checkpoints/nllb_combined_guz/final`
to exist (i.e. run both training notebooks first, or copy those two folders in from
wherever training happened). If deploying somewhere the checkpoints aren't already
present locally (e.g. Streamlit Community Cloud), the simplest option is to push both
model folders to the Hugging Face Hub and change `MT5_MODEL_DIR` / `NLLB_MODEL_DIR` in
`app.py` to the corresponding repo ids (e.g. `"your-username/mt5-en-guz"`) — the
`from_pretrained(...)` calls work identically either way.

## Notes for whoever runs this

- Both notebooks default to a single GPU (`CUDA_VISIBLE_DEVICES=0`) — safe on
  single-GPU machines, and avoids a known multi-GPU memory overhead issue on some
  multi-GPU environments.
- mT5 is trained in fp32 (`fp16=False`) — this is intentional, not an oversight. mT5
  is numerically unstable in fp16 on GPUs without bf16 support and produces NaN
  losses if forced into fp16.
- NLLB uses the Adafactor optimizer with gradient checkpointing and gradient
  accumulation, tuned to fit a 15GB GPU. These are conservative and can be relaxed
  (larger batch size, Adam) on a GPU with more memory (e.g. 80GB A100) for faster
  training.
- Ekegusii has no native language code in NLLB-200; a placeholder target-language tag
  (`swh_Latn`) is used to steer generation, since NLLB requires a valid tag at
  generation time.
