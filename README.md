# Covered Call Strategy Simulator

Monte Carlo simulation comparing two covered call strategies over a 12-month horizon:

- **Fixed Strike** — Always sell the call at the original stock price ($100), regardless of where the stock moves.
- **Rolling ATM** — Sell a fresh at-the-money call at the current stock price each month.

Uses 1,000,000 simulated stock price paths (geometric Brownian motion) with vectorized NumPy operations. Options are priced via Black-Scholes with a built-in volatility risk premium (25% implied vs 20% realized).

## Installation

Requires Python 3.14+. Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Usage

### Single-scenario simulation

```bash
python simulation.py [drift] [output_pdf_name]
```

- `drift` — annualized stock drift (default: `0.05`)
- `output_pdf_name` — output filename (default: `covered_call_report.pdf`)

Examples:

```bash
python simulation.py                              # 5% drift, default output
python simulation.py 0.0 report_flat.pdf          # 0% drift
python simulation.py -0.05 report_bearish.pdf     # -5% drift
```

### Multi-scenario comparison

```bash
python comparison_report.py
```

Runs all three drift scenarios (+5%, 0%, -5%) and produces a single `comparison_report.pdf` with plain-English explanations, grouped bar charts, and a strategy recommendation.

## Key Parameters

| Parameter | Value |
|-----------|-------|
| Initial stock price | $100 |
| Realized volatility | 20% annualized |
| Implied volatility | 25% annualized |
| Risk-free rate | 5% |
| Option tenor | 21 trading days (1 month) |
| Simulations | 1,000,000 |
| Delta exit threshold | Position delta < 0.1 |

## Output

Both scripts generate PDF reports containing:

- Summary statistics (mean, median, Sharpe ratio, win rate, percentiles)
- Return distribution histograms and CDF plots
- Head-to-head and conditional analysis
- Strategy recommendations with explanations

## License

[MIT](LICENSE)
