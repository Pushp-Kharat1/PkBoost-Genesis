# PKBoost: An Experimental Gradient Boosting Framework with Hybrid Split Criterion

## Overview

PKBoost is an experimental gradient boosting framework for binary classification, built from scratch to explore the inner workings of models like XGBoost and LightGBM. It features a custom C++ backend that integrates Shannon entropy-based mutual information (MI) with traditional Newton gain for split finding. This hybrid approach aims to balance predictive accuracy and information-theoretic feature selection. The framework uses `pybind11` for Python integration and OpenMP for multi-threading.

This project is in **early development**, developed as a personal learning exercise, and is not yet optimized for production use. The repository contains a single Jupyter notebook (`PkBoost.ipynb`) with the full implementation, including the C++ backend, Python classes, and a benchmark against LightGBM.

## Motivation

I created PKBoost out of curiosity about how gradient boosting models like LightGBM and XGBoost function under the hood. Specifically, I wondered why these models rely solely on Newton gain for split decisions and not other metrics. To experiment, I combined Newton gain with mutual information (MI) derived from Shannon entropy, creating a hybrid split criterion: `combined_gain = newton_gain + adaptive_weight * mutual_info`. This allows for splits that not only minimize loss but also maximize information gain about the target variable.

While this is a simple "mash-up" of formulas, it provides an educational tool for understanding and tweaking boosting mechanics. The benchmark shows potential accuracy benefits, but at the cost of speed in its current form.

## Features

- **Hybrid Split Criterion**: Uniquely combines Newton gain (from XGBoost-style second-order approximation) with Shannon entropy-based mutual information. The MI term is weighted adaptively (`mi_weight * exp(-0.1 * depth)`) to favor it more at shallower depths, potentially leading to more informative early splits. This differs from standard libraries, which typically use gain or similar metrics without explicit information theory integration.
- **Custom C++ Backend**: Handles core computations (sigmoid, gradient, hessian, split finding) with OpenMP for parallelization. While optimized for multi-threading, it's currently slower than highly-tuned libraries like LightGBM due to simpler implementations.
- **Smart Histogram Builder**: Dynamically bins features (up to 32 bins) for efficient handling of continuous data.
- **Early Stopping and Flexibility**: Supports validation-based early stopping and hyperparameters like `n_estimators`, `learning_rate`, `max_depth`, etc.
- **Why Choose PKBoost?**: If you're interested in educational experimentation, customizing split criteria, or exploring hybrid approaches that might yield slight accuracy gains in specific datasets (as seen in the benchmark), PKBoost could be useful. However, for production speed and reliability, established libraries like LightGBM are recommended. PKBoost's unique value lies in its transparency and modifiability for learning purposes.

## Project Status

This is an early-stage hobby project and not fully optimized. Known issues include:

- Slower training times (e.g., 80x slower than LightGBM in benchmarks) due to unoptimized histogram building and split finding.
- Potential compilation errors in non-Linux environments or without proper OpenMP support.
- Limited testing, error handling, and support for large datasets or multi-class problems.

Feedback and contributions are welcome to address these!

## Installation

PKBoost requires:

- Python 3.8+
- Libraries: `numpy`, `pybind11`, `lightgbm`, `scikit-learn`
- Compiler: `g++` supporting C++17 and OpenMP (via `#pragma omp`)
- Optional: Kaggle API for the benchmark dataset

Install Python dependencies:

```bash
pip install numpy pybind11 lightgbm scikit-learn
```

For OpenMP on Ubuntu:

```bash
sudo apt-get install g++ libomp-dev
```

**Note**: The C++ backend uses `#pragma omp` for parallelism, which may not compile or work on all systems (e.g., some macOS versions or Windows without specific setups). Test compilation in your environment. If issues arise, consider disabling OpenMP flags in the compile command.

## Usage

The notebook (`PkBoost.ipynb`) is self-contained:

1. Installs dependencies.
2. Defines and compiles the C++ backend.
3. Implements Python classes: `SmartHistogramBuilder`, `CppShannonLoss`, `SimpleTreeShannon`, `PKBoostShannon`.
4. Runs a benchmark on the Credit Card Fraud dataset.

To use in your code:

1. Compile the C++ module (see notebook Cell 3).
2. Train the model:

   ```python
   from PkBoost import PKBoostShannon  # Assuming classes are saved separately
   model = PKBoostShannon(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42, mi_weight=0.5)
   model.fit(X_train, y_train, eval_set=(X_val, y_val))
   predictions = model.predict_proba(X_test)
   ```

## Benchmark Results

PKBoost is compared to LightGBM and Logistic Regression on the Credit Card Fraud dataset (imbalanced binary classification). Sample results:

- **Logistic Regression**: ROC-AUC = 0.967041, Time = N/A
- **PKBoost**: ROC-AUC = 0.977481, Time = 48.72s
- **LightGBM**: ROC-AUC = 0.967728, Time = 0.60s

**Interpretation**: PKBoost achieves a slightly higher ROC-AUC (about 1% better), which may indicate benefits from the hybrid split criterion in this dataset. However, this difference might not be statistically significant without further testing (e.g., cross-validation). The substantial speed gap (PKBoost is \~80x slower) highlights areas for optimization, such as faster histogram construction or GPU acceleration. Performance claims like "fast calculations" are relative to pure Python implementations but not to optimized C++ libraries like LightGBM.

## Contributing

Contributions are encouraged! Focus areas:

- Speed optimizations (e.g., better parallelism or algorithmic improvements).
- Cross-platform compilation fixes.
- Expanded features (e.g., multi-class support).
- Rigorous statistical testing of accuracy gains.

To contribute:

1. Fork the repo.
2. Create a branch (`git checkout -b feature/optimization`).
3. Commit changes.
4. Push and open a PR with details/tests.

## Contact

Suggestions or questions? Email me at: kharatpushp16@outlook.com

## License

MIT License. See LICENSE.

## Acknowledgments

- Inspired by XGBoost and LightGBM.
- Uses `pybind11` and OpenMP.
- Tested on Kaggle's Credit Card Fraud dataset.

## Note

I am an 18-year-old hobbyist and have developed this project on my own. As it is in the early stages, there may be some obvious issues or inefficiencies. I greatly appreciate any feedback or suggestions to help improve PKBoost!
