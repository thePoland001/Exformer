# Exformer

This repository contains the PyTorch implementation for the paper "Extreme Adaptive Transformer for Time Series
forecasting".

## Introduction
Time series forecasting remains challenging when the underlying data contain rare but critical extreme events. This issue is particularly important in hydrologic forecasting, where streamflow distributions are often highly skewed and extreme peaks can have substantial impacts on flood monitoring, water resource management, and early warning systems. Although Transformer-based forecasting models have achieved strong performance by modeling long-range temporal dependencies, they typically treat all time points uniformly and may therefore underrepresent rare extreme patterns. In this paper, we propose the Extreme-Adaptive Transformer (Exformer), a forecasting framework designed to explicitly model temporal dependencies involving both normal and extreme events. Exformer introduces an extreme-adaptive attention mechanism composed of three sparse components: Local, Stride, and Extreme. The Local and Stride components capture short-term and periodic temporal dependencies, respectively, while the Extreme component selectively models event-aware dependencies between normal and extreme streamflow patterns. Experiments on four real-world hydrologic streamflow datasets show that Exformer achieves superior 3-day forecasting performance compared with state-of-the-art baselines. Our findings demonstrate that explicitly incorporating extreme-aware attention improves the forecasting capacity of Transformer models on imbalanced time series with rare but consequential events.

## Train and Test
1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the benchmark datasets from [here](https://clp.engr.scu.edu/static/datasets/seed_datasets.zip) and upzip the files. There should now be 4 stream sensor (file names end with _S_fixed.csv) and 4 rain sensor (file names end with _R_fixed.csv) datasets. Place the raw sensor files inside the directory:
   `
   ./data/datasets/watershed/raw/
   `

3. Dataset preprocessing:
   To generate the processed dataset used by the training code, run the preprocessing script from the repository root:

   ```bash
   # for Ross dataset
   python processed_datasets/data_processing.py --arg_file processed_datasets/configs/Ross_withRain.txt
   ```

   The processed outputs will be written to a folder like:

   `
   ./data/datasets/watershed/Ross_withRain/in1440_out288/
   `


3. Reproduce the experiments by running the example scripts in the scripts directory:
   ```bash
   bash ./scripts/Ross.sh
   bash ./scripts/Saratoga.sh
   bash ./scripts/SFC.sh
   bash ./scripts/Upperpen.sh
   ```

## Acknowledgements
We sincerely appreciate the foundational code and inspiration from the following repositories:
- https://github.com/wanghq21/MICN
- https://github.com/zhouhaoyi/Informer2020
- https://github.com/Thinklab-SJTU/Crossformer
- https://github.com/thuml/Time-Series-Library
- https://github.com/cure-lab/SCINet
- https://github.com/davidanastasiu/dan
