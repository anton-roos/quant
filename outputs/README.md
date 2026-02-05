# Outputs Directory

Generated outputs from the model pipeline.

## Structure

```
outputs/
├── forecasts/        # Forecast CSV files (one per ticker)
├── videos/           # Generated backtest visualization videos  
└── models/           # Trained model weights and checkpoints
```

## Usage

- **forecasts/** - Portfolio forecast CSVs are generated here by `src/inference/run_forecast_.py`
- **videos/** - Backtest performance animations are saved here by `src/inference/run_backtest_.py`
- **models/** - Trained model files should be checkpointed here during training

## Tips

- Use `.gitignore` to exclude these files from version control if they're large
- Clean this directory periodically to save disk space
- Archive old forecasts/videos if needed for historical analysis
