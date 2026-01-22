# Football (soccer) Betting AI Model

This project explores **machine learning-driven football betting strategies**, combining **supervised deep learning**, **reinforcement learning**, and **risk-aware decision-making**.

The system learns predictive representations of football matches from historical data and uses those predictions to simulate and optimize betting strategies under bankroll and risk constraints.

> **Scope note**: This is a research and engineering project focused on modelling, data systems, and training pipelines, not a production betting product.

## Core Components

### FBRef web scraper (``fbref_scraper/``)

A professional-grade web scraping pipeline for collecting football match data from FBRef.

- Fetches HTML using [``rnet``](https://github.com/0x676e67/rnet) with browser impersonation
- Parses in-game match statistics across multiple domestic leagues and seasons
- Persists structured data into a **SQLite** database via ``sqlite3``
- Designed with respect for [``robots.txt``](https://fbref.com/robots.txt)
> **Status**: Temporarily disabled due to Cloudflare browser-verification enforcement on FBRef. Existing datasets (~76k samples) were collected prior to this change for use in the rest of the project.

### Football data analysis (``analysis/``)

Exploratory **data analysis** and statistical insight generation based on the scraped SQLite database.

- visualisations using ``Matplotlib`` (e.g. tree diagrams, histograms, scatter plots)
- Group-based aggregations and comparisons
- Analysis of win / draw / loss likelihood when dominating a given statistic
- Tree and hierarchy-style diagrams illustrating **odds ratio** for:
  
  - Win / Draw / Loss
  - Conditional on statistically significant features (**p-value** thresholds)

This directory is intended for **interpretability**, **hypothesis generation**, and **sanity checking**.

### Tensor preprocessing pipeline (``tensor_pipeline/``)

Transforms the relational SQLite dataset into **model-ready tensors**.

- ``Tokeniser`` and ``Normaliser`` objects that fit on the train data
- Custom **sharded on-disk tensor store** for scalable datasets
- Persistent indexing for:
  
  - Train / validation / test splits
  - Arbitrary groupings (default is league)
- Support for variable-length historical context windows via **sequence slicing at dataset load time**
- ``MatchDataset`` that inherits from ``torch.Dataset``

#### Custom Data Augmentation / Transforms
Implemented specifically for temporal sports data:
- ``TemporalDropout`` - drops contiguous time steps
- ``RandomTokenUNK`` - mutates categorical tokens to ``<unk>``
- ``MissingValueAugment`` - injects missingness into continuous features
- ``ContinuousFeatureDropout`` - feature-wise dropout for numeric inputs

### Match outcome prediction models (``match_predictor/``)
Modular **deep-learning match predictor** built with ``torch``, including a full **training loop and evaluation logic**

Model architecture consists of:

- **Attention and convolution-based** context encoder
- **Attention and feed-forward-based** feature extractor
- Final **MLP** prediction head
- Optional residual connections throughout (excluding convolution reshaping blocks)

Input support includes:
- Home and away team context tensors
- Temporal masks
- Missing-value indicator tensors

> **Planned extension**: Trial tracking and experiment logging.
This may later be factored into a shared module if reused by reinforcement learning components.

## Planned / Future Components

### Reinforcement Value Learning (planned)
- Learning value functions from historical odds data
- Using transfer learning on pretrained match predictor (fine tuning)

### Betting Strategy Optimization (planned)
- Strategy layer that interprets model predictions
- Stake sizing and selection logic
- Risk-adjusted evaluation (drawdowns, variance, expected value)
- Statistical validation of decision quality

This layer acts as a **model interpreter and decision optimizer**, rather than a black-box policy.

## Project Status

This section serves as a **living development log**.

### Current Focus
- Designing and implementing trial tracking for model training
  - Hyperparameters
  - Dataset configuration
  - Model checkpoints
  - Evaluation metrics

### Recently Completed
- Sharded tensor storage system (``SampleStore``)
- Flexible dataset slicing and grouping
- Custom temporal and feature-level augmentations
- Refactoring ``fbref_scraper`` to fetch HTML using ``rnet``
- Baseline predictive model training on ~76k samples

### Known Limitations
- Scraper blocked by Cloudflare verification
- No browser automation fallback implemented
- Reinforcement learning components not yet integrated

## Disclaimer

This project is for **research and educational purposes only**.
It is not intended for real-world gambling or commercial betting use.

## Author

Created by [**WillEdgington**](https://github.com/WillEdgington)

📧 **willedge037@gmail.com**

🔗 [**LinkedIn**](https://www.linkedin.com/in/williamedgington/)