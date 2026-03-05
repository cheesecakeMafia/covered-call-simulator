"""
Monte Carlo Simulation: Covered Call Strategies
================================================
Compares two covered call strategies over 100,000 simulations:
  Strategy 1 (Fixed Strike): Always sell call at the ORIGINAL strike (S0)
  Strategy 2 (Rolling ATM):  Sell call ATM at whatever the stock price is at roll time

Uses vectorized NumPy operations for performance.
Generates a PDF report with charts and statistical analysis.
"""

import sys
import os
import time
from io import BytesIO

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# ─── Parameters ──────────────────────────────────────────────────
S0 = 100.0
MU = 0.05
SIGMA_REALIZED = 0.20
SIGMA_IMPLIED = 0.25
R = 0.05
TRADING_DAYS_PER_MONTH = 21
NUM_MONTHS = 12
TOTAL_DAYS = TRADING_DAYS_PER_MONTH * NUM_MONTHS
NUM_SIMS = 1_000_000
DELTA_EXIT_THRESHOLD = 0.1
DT = 1.0 / 252

np.random.seed(42)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─── Black-Scholes (vectorized) ──────────────────────────────────
def bs_d1_vec(S, K, T, r, sigma):
    """Vectorized d1."""
    safe_T = np.maximum(T, 1e-10)
    return (np.log(S / K) + (r + 0.5 * sigma**2) * safe_T) / (sigma * np.sqrt(safe_T))


def bs_call_price_vec(S, K, T, r, sigma):
    """Vectorized BS call price."""
    # For expired options
    expired = T <= 0
    d1 = bs_d1_vec(S, K, np.maximum(T, 1e-10), r, sigma)
    d2 = d1 - sigma * np.sqrt(np.maximum(T, 1e-10))
    price = S * norm.cdf(d1) - K * np.exp(-r * np.maximum(T, 0)) * norm.cdf(d2)
    # Override expired with intrinsic
    price = np.where(expired, np.maximum(S - K, 0.0), price)
    return price


def bs_call_delta_vec(S, K, T, r, sigma):
    """Vectorized BS call delta."""
    expired = T <= 0
    d1 = bs_d1_vec(S, K, np.maximum(T, 1e-10), r, sigma)
    delta = norm.cdf(d1)
    delta = np.where(expired, np.where(S > K, 1.0, 0.0), delta)
    return delta


def bs_call_price_scalar(S, K, T, r, sigma):
    """Scalar BS call price."""
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# ─── Vectorized Simulation ───────────────────────────────────────
def simulate_paths(num_sims, total_days):
    """Generate stock price paths using GBM."""
    daily_log_ret = np.random.normal(
        (MU - 0.5 * SIGMA_REALIZED**2) * DT,
        SIGMA_REALIZED * np.sqrt(DT),
        size=(num_sims, total_days)
    )
    log_prices = np.hstack([np.zeros((num_sims, 1)), np.cumsum(daily_log_ret, axis=1)])
    return S0 * np.exp(log_prices)


def run_strategy_vectorized(prices, fixed_strike: bool):
    """
    Vectorized covered call strategy simulation.

    fixed_strike=True  → Strategy 1: always sell at K=S0
    fixed_strike=False → Strategy 2: sell ATM at current stock price each month

    Returns: (pnl_array, exit_day_array)
    """
    num_sims = prices.shape[0]

    pnl = np.zeros(num_sims)
    active = np.ones(num_sims, dtype=bool)
    stock_entry = np.full(num_sims, S0)
    exit_day = np.full(num_sims, TOTAL_DAYS)  # Default: hold to end

    initial_K = S0
    call_strike = np.full(num_sims, initial_K)

    initial_premium = bs_call_price_scalar(S0, initial_K, TRADING_DAYS_PER_MONTH / 252.0, R, SIGMA_IMPLIED)
    call_premium_received = np.full(num_sims, initial_premium)

    for day in range(1, TOTAL_DAYS + 1):
        if not np.any(active):
            break

        S = prices[active, day]
        K = call_strike[active]
        premiums = call_premium_received[active]
        entries = stock_entry[active]

        day_in_month = (day - 1) % TRADING_DAYS_PER_MONTH + 1
        T_left = (TRADING_DAYS_PER_MONTH - day_in_month) / 252.0

        # Check exit condition
        c_delta = bs_call_delta_vec(S, K, T_left, R, SIGMA_IMPLIED)
        pos_delta = 1.0 - c_delta
        should_exit = pos_delta < DELTA_EXIT_THRESHOLD

        if np.any(should_exit):
            # Get indices of active sims that should exit
            active_indices = np.where(active)[0]
            exit_indices = active_indices[should_exit]

            S_exit = S[should_exit]
            K_exit = K[should_exit]
            prem_exit = premiums[should_exit]
            entry_exit = entries[should_exit]

            call_val = bs_call_price_vec(S_exit, K_exit, T_left, R, SIGMA_IMPLIED)
            stock_pnl = S_exit - entry_exit
            option_pnl = prem_exit - call_val

            pnl[exit_indices] += stock_pnl + option_pnl
            active[exit_indices] = False
            exit_day[exit_indices] = day

        # Monthly roll
        if day_in_month == TRADING_DAYS_PER_MONTH and day < TOTAL_DAYS:
            if not np.any(active):
                break

            active_idx = np.where(active)[0]
            S_roll = prices[active_idx, day]
            K_roll = call_strike[active_idx]
            prem_roll = call_premium_received[active_idx]
            entry_roll = stock_entry[active_idx]

            # Settle expiring call
            intrinsic = np.maximum(S_roll - K_roll, 0.0)
            option_pnl = prem_roll - intrinsic
            stock_pnl = S_roll - entry_roll
            pnl[active_idx] += stock_pnl + option_pnl

            # Reset entry
            stock_entry[active_idx] = S_roll

            # Write new call
            if fixed_strike:
                new_K = np.full(len(active_idx), initial_K)
            else:
                new_K = S_roll.copy()

            call_strike[active_idx] = new_K
            T_new = TRADING_DAYS_PER_MONTH / 252.0
            new_prem = bs_call_price_vec(S_roll, new_K, T_new, R, SIGMA_IMPLIED)
            call_premium_received[active_idx] = new_prem

    # Final settlement
    still_active = np.where(active)[0]
    if len(still_active) > 0:
        S_final = prices[still_active, TOTAL_DAYS]
        K_final = call_strike[still_active]
        prem_final = call_premium_received[still_active]
        entry_final = stock_entry[still_active]

        intrinsic = np.maximum(S_final - K_final, 0.0)
        option_pnl = prem_final - intrinsic
        stock_pnl = S_final - entry_final
        pnl[still_active] += stock_pnl + option_pnl

    return pnl, exit_day


# ─── Generate Charts (in-memory, no files saved) ────────────────
def _fig_to_bytesio(fig):
    """Save a matplotlib figure to a BytesIO buffer."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf


def make_histogram(ret_fixed, ret_rolling):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bins = np.linspace(
        min(np.percentile(ret_fixed, 1), np.percentile(ret_rolling, 1)),
        max(np.percentile(ret_fixed, 99), np.percentile(ret_rolling, 99)), 80)
    axes[0].hist(ret_fixed, bins=bins, alpha=0.7, color='steelblue', edgecolor='white', linewidth=0.5)
    axes[0].axvline(np.mean(ret_fixed), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(ret_fixed):.2f}%')
    axes[0].axvline(np.median(ret_fixed), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(ret_fixed):.2f}%')
    axes[0].set_title('Strategy 1: Fixed Strike', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Return (%)'); axes[0].set_ylabel('Frequency'); axes[0].legend(fontsize=9)
    axes[1].hist(ret_rolling, bins=bins, alpha=0.7, color='darkorange', edgecolor='white', linewidth=0.5)
    axes[1].axvline(np.mean(ret_rolling), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(ret_rolling):.2f}%')
    axes[1].axvline(np.median(ret_rolling), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(ret_rolling):.2f}%')
    axes[1].set_title('Strategy 2: Rolling ATM', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Return (%)'); axes[1].set_ylabel('Frequency'); axes[1].legend(fontsize=9)
    plt.tight_layout()
    return _fig_to_bytesio(fig)


def make_comparison_overlay(ret_fixed, ret_rolling):
    fig, ax = plt.subplots(figsize=(10, 5))
    lo = min(np.percentile(ret_fixed, 1), np.percentile(ret_rolling, 1))
    hi = max(np.percentile(ret_fixed, 99), np.percentile(ret_rolling, 99))
    bins = np.linspace(lo, hi, 80)
    ax.hist(ret_fixed, bins=bins, alpha=0.5, color='steelblue', label='Fixed Strike', edgecolor='white', linewidth=0.3)
    ax.hist(ret_rolling, bins=bins, alpha=0.5, color='darkorange', label='Rolling ATM', edgecolor='white', linewidth=0.3)
    ax.axvline(np.mean(ret_fixed), color='steelblue', linestyle='--', linewidth=2)
    ax.axvline(np.mean(ret_rolling), color='darkorange', linestyle='--', linewidth=2)
    ax.set_title('Return Distribution Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Return (%)'); ax.set_ylabel('Frequency'); ax.legend(fontsize=11)
    plt.tight_layout()
    return _fig_to_bytesio(fig)


def make_cdf_plot(ret_fixed, ret_rolling):
    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_f = np.sort(ret_fixed); sorted_r = np.sort(ret_rolling)
    cdf = np.arange(1, len(sorted_f) + 1) / len(sorted_f)
    ax.plot(sorted_f, cdf, color='steelblue', linewidth=2, label='Fixed Strike')
    ax.plot(sorted_r, cdf, color='darkorange', linewidth=2, label='Rolling ATM')
    ax.set_title('Cumulative Distribution of Returns', fontsize=14, fontweight='bold')
    ax.set_xlabel('Return (%)'); ax.set_ylabel('Cumulative Probability')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _fig_to_bytesio(fig)


def make_diff_histogram(ret_fixed, ret_rolling):
    diff = ret_fixed - ret_rolling
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(diff, bins=80, alpha=0.7, color='seagreen', edgecolor='white', linewidth=0.5)
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(np.mean(diff), color='red', linestyle='--', linewidth=2, label=f'Mean diff: {np.mean(diff):+.2f}%')
    ax.set_title('Per-Simulation Difference (Fixed − Rolling)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Return Difference (%)'); ax.set_ylabel('Frequency'); ax.legend(fontsize=11)
    plt.tight_layout()
    return _fig_to_bytesio(fig)


# ─── PDF Report ──────────────────────────────────────────────────
def generate_pdf_report(ret_fixed, ret_rolling, exit_fixed, exit_rolling, final_prices, pdf_filename="covered_call_report.pdf"):
    """Generate a comprehensive PDF report."""
    pdf_path = os.path.join(OUTPUT_DIR, pdf_filename)
    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                            topMargin=0.6*inch, bottomMargin=0.6*inch,
                            leftMargin=0.75*inch, rightMargin=0.75*inch)

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle('CustomTitle', parent=styles['Title'],
                                  fontSize=22, spaceAfter=6, textColor=colors.HexColor('#1a1a2e'))
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'],
                                     fontSize=12, spaceAfter=12, textColor=colors.grey,
                                     alignment=TA_CENTER)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading1'],
                                    fontSize=16, spaceBefore=18, spaceAfter=10,
                                    textColor=colors.HexColor('#16213e'))
    heading2_style = ParagraphStyle('CustomHeading2', parent=styles['Heading2'],
                                     fontSize=13, spaceBefore=12, spaceAfter=6,
                                     textColor=colors.HexColor('#0f3460'))
    body_style = ParagraphStyle('CustomBody', parent=styles['Normal'],
                                 fontSize=10, leading=14, alignment=TA_JUSTIFY,
                                 spaceAfter=8)
    bold_style = ParagraphStyle('Bold', parent=body_style, fontName='Helvetica-Bold')

    elements = []

    # ── Title Page ──
    elements.append(Spacer(1, 1.5*inch))
    elements.append(Paragraph("Covered Call Strategies", title_style))
    elements.append(Paragraph("Monte Carlo Simulation: Fixed Strike vs. Rolling ATM", subtitle_style))
    elements.append(Spacer(1, 0.3*inch))
    elements.append(HRFlowable(width="80%", thickness=2, color=colors.HexColor('#e94560')))
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph(f"{NUM_SIMS:,} Simulations &bull; 12-Month Horizon &bull; Daily Granularity", subtitle_style))
    elements.append(Spacer(1, 0.5*inch))

    # Parameters table
    param_data = [
        ['Parameter', 'Value'],
        ['Initial Stock Price', f'${S0:.0f}'],
        ['Stock Drift (μ)', f'{MU*100:.1f}% annualized'],
        ['Realized Volatility', f'{SIGMA_REALIZED*100:.1f}% annualized'],
        ['Implied Volatility (BS pricing)', f'{SIGMA_IMPLIED*100:.1f}% annualized'],
        ['Vol Risk Premium (IV − RV)', f'{(SIGMA_IMPLIED - SIGMA_REALIZED)*100:.1f}% (in vol terms)'],
        ['Risk-Free Rate', f'{R*100:.1f}%'],
        ['Option Tenor', f'{TRADING_DAYS_PER_MONTH} trading days (1 month)'],
        ['Simulation Period', f'{NUM_MONTHS} months ({TOTAL_DAYS} trading days)'],
        ['Number of Simulations', f'{NUM_SIMS:,}'],
        ['Delta Exit Threshold', f'Position Δ < {DELTA_EXIT_THRESHOLD}'],
    ]
    param_table = Table(param_data, colWidths=[2.8*inch, 3.5*inch])
    param_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16213e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0f0f0'), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(param_table)

    elements.append(PageBreak())

    # ── Strategy Descriptions ──
    elements.append(Paragraph("Strategy Descriptions", heading_style))

    elements.append(Paragraph("<b>Common Setup:</b> Buy 1 share of stock at $100, sell 1 ATM call option "
                              "(strike = $100, 1-month expiry). Each day, check if the position delta "
                              "falls below 0.1 (meaning the call is deep in-the-money with delta > 0.9). "
                              "If so, close the entire position. At each monthly expiry, settle the call "
                              "and write a new one.", body_style))

    elements.append(Paragraph("<b>Strategy 1 — Fixed Strike:</b> Always sell the call at the original "
                              "strike price of $100, regardless of where the stock has moved. If the stock "
                              "drops to $70, you're selling a $100-strike call (far OTM). If it rallies to "
                              "$130, you're selling a $100-strike call (deep ITM).", body_style))

    elements.append(Paragraph("<b>Strategy 2 — Rolling ATM:</b> At each monthly roll, sell a call struck "
                              "at the current stock price. If the stock is at $70, sell a $70 call. If at "
                              "$130, sell a $130 call. This keeps the call always at-the-money.", body_style))

    elements.append(Paragraph("<b>Exit Condition:</b> Position delta &lt; 0.1 means the short call's delta "
                              "exceeds 0.9 — the stock has rallied far above the strike. The covered call "
                              "has effectively become a capped position with almost no further upside. "
                              "Exiting here avoids carrying a position with extreme negative gamma.", body_style))

    elements.append(Spacer(1, 0.2*inch))

    # ── Summary Statistics ──
    elements.append(Paragraph("Summary Statistics", heading_style))

    def compute_stats(rets, label):
        return {
            'label': label,
            'mean': np.mean(rets),
            'median': np.median(rets),
            'std': np.std(rets),
            'sharpe': np.mean(rets) / np.std(rets) if np.std(rets) > 0 else 0,
            'min': np.min(rets),
            'max': np.max(rets),
            'p5': np.percentile(rets, 5),
            'p25': np.percentile(rets, 25),
            'p75': np.percentile(rets, 75),
            'p95': np.percentile(rets, 95),
            'skew': float(pd.Series(rets).skew()),
            'kurtosis': float(pd.Series(rets).kurtosis()),
            'win_rate': np.mean(rets > 0) * 100,
        }

    sf = compute_stats(ret_fixed, "Fixed Strike")
    sr = compute_stats(ret_rolling, "Rolling ATM")

    stats_data = [
        ['Metric', 'Fixed Strike', 'Rolling ATM', 'Difference'],
        ['Mean Return', f'{sf["mean"]:.2f}%', f'{sr["mean"]:.2f}%', f'{sf["mean"]-sr["mean"]:+.2f}%'],
        ['Median Return', f'{sf["median"]:.2f}%', f'{sr["median"]:.2f}%', f'{sf["median"]-sr["median"]:+.2f}%'],
        ['Std Deviation', f'{sf["std"]:.2f}%', f'{sr["std"]:.2f}%', f'{sf["std"]-sr["std"]:+.2f}%'],
        ['Sharpe Ratio', f'{sf["sharpe"]:.3f}', f'{sr["sharpe"]:.3f}', f'{sf["sharpe"]-sr["sharpe"]:+.3f}'],
        ['Win Rate', f'{sf["win_rate"]:.1f}%', f'{sr["win_rate"]:.1f}%', f'{sf["win_rate"]-sr["win_rate"]:+.1f}%'],
        ['5th Percentile', f'{sf["p5"]:.2f}%', f'{sr["p5"]:.2f}%', f'{sf["p5"]-sr["p5"]:+.2f}%'],
        ['25th Percentile', f'{sf["p25"]:.2f}%', f'{sr["p25"]:.2f}%', f'{sf["p25"]-sr["p25"]:+.2f}%'],
        ['75th Percentile', f'{sf["p75"]:.2f}%', f'{sr["p75"]:.2f}%', f'{sf["p75"]-sr["p75"]:+.2f}%'],
        ['95th Percentile', f'{sf["p95"]:.2f}%', f'{sr["p95"]:.2f}%', f'{sf["p95"]-sr["p95"]:+.2f}%'],
        ['Skewness', f'{sf["skew"]:.3f}', f'{sr["skew"]:.3f}', ''],
        ['Excess Kurtosis', f'{sf["kurtosis"]:.3f}', f'{sr["kurtosis"]:.3f}', ''],
        ['Worst Case', f'{sf["min"]:.2f}%', f'{sr["min"]:.2f}%', ''],
        ['Best Case', f'{sf["max"]:.2f}%', f'{sr["max"]:.2f}%', ''],
    ]

    stats_table = Table(stats_data, colWidths=[1.6*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16213e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0f0f0'), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(stats_table)

    elements.append(Spacer(1, 0.2*inch))

    # Head-to-head
    elements.append(Paragraph("Head-to-Head Comparison", heading2_style))
    diff = ret_fixed - ret_rolling
    fixed_wins = np.mean(diff > 0) * 100
    rolling_wins = np.mean(diff < 0) * 100

    h2h_data = [
        ['Metric', 'Value'],
        ['Fixed Strike wins', f'{fixed_wins:.1f}% of simulations'],
        ['Rolling ATM wins', f'{rolling_wins:.1f}% of simulations'],
        ['Mean PnL difference (Fixed − Rolling)', f'{np.mean(diff):+.2f}%'],
        ['Median PnL difference', f'{np.median(diff):+.2f}%'],
    ]
    h2h_table = Table(h2h_data, colWidths=[3.0*inch, 3.0*inch])
    h2h_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f3460')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0f0f0'), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(h2h_table)

    # Conditional analysis
    elements.append(Spacer(1, 0.15*inch))
    elements.append(Paragraph("Conditional Analysis", heading2_style))

    up_mask = final_prices > S0
    down_mask = final_prices <= S0

    cond_data = [
        ['Condition', 'Fixed Strike', 'Rolling ATM'],
        [f'Stock ends UP (n={np.sum(up_mask):,})',
         f'{np.mean(ret_fixed[up_mask]):+.2f}%', f'{np.mean(ret_rolling[up_mask]):+.2f}%'],
        [f'Stock ends DOWN (n={np.sum(down_mask):,})',
         f'{np.mean(ret_fixed[down_mask]):+.2f}%', f'{np.mean(ret_rolling[down_mask]):+.2f}%'],
    ]

    # Further breakdown
    big_down = final_prices < S0 * 0.8
    small_down = (final_prices >= S0 * 0.8) & (final_prices <= S0)
    small_up = (final_prices > S0) & (final_prices <= S0 * 1.2)
    big_up = final_prices > S0 * 1.2

    for label, mask in [('Stock down >20%', big_down), ('Stock down 0-20%', small_down),
                        ('Stock up 0-20%', small_up), ('Stock up >20%', big_up)]:
        if np.sum(mask) > 0:
            cond_data.append([f'{label} (n={np.sum(mask):,})',
                            f'{np.mean(ret_fixed[mask]):+.2f}%',
                            f'{np.mean(ret_rolling[mask]):+.2f}%'])

    cond_table = Table(cond_data, colWidths=[2.6*inch, 1.8*inch, 1.8*inch])
    cond_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f3460')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0f0f0'), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(cond_table)

    # Exit analysis
    elements.append(Spacer(1, 0.15*inch))
    elements.append(Paragraph("Early Exit Analysis", heading2_style))

    fixed_early = np.sum(exit_fixed < TOTAL_DAYS)
    rolling_early = np.sum(exit_rolling < TOTAL_DAYS)

    exit_data = [
        ['Metric', 'Fixed Strike', 'Rolling ATM'],
        ['Early exits (delta < 0.1)', f'{fixed_early:,} ({fixed_early/NUM_SIMS*100:.1f}%)',
         f'{rolling_early:,} ({rolling_early/NUM_SIMS*100:.1f}%)'],
    ]
    if fixed_early > 0:
        exit_data.append(['Avg exit day (if early)',
                         f'Day {np.mean(exit_fixed[exit_fixed < TOTAL_DAYS]):.0f}',
                         f'Day {np.mean(exit_rolling[exit_rolling < TOTAL_DAYS]):.0f}' if rolling_early > 0 else 'N/A'])

    exit_table = Table(exit_data, colWidths=[2.2*inch, 2.0*inch, 2.0*inch])
    exit_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f3460')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0f0f0'), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(exit_table)

    elements.append(PageBreak())

    # ── Charts (in-memory) ──
    elements.append(Paragraph("Return Distributions", heading_style))

    hist_buf = make_histogram(ret_fixed, ret_rolling)
    overlay_buf = make_comparison_overlay(ret_fixed, ret_rolling)
    cdf_buf = make_cdf_plot(ret_fixed, ret_rolling)
    diff_buf = make_diff_histogram(ret_fixed, ret_rolling)

    elements.append(Image(hist_buf, width=6.5*inch, height=2.7*inch))
    elements.append(Spacer(1, 0.15*inch))
    elements.append(Image(overlay_buf, width=6.0*inch, height=3.0*inch))

    elements.append(PageBreak())

    elements.append(Paragraph("Cumulative Distribution & Difference Analysis", heading_style))
    elements.append(Image(cdf_buf, width=6.0*inch, height=3.0*inch))
    elements.append(Spacer(1, 0.15*inch))
    elements.append(Image(diff_buf, width=6.0*inch, height=3.0*inch))

    elements.append(PageBreak())

    # ── Conclusion ──
    elements.append(Paragraph("Conclusion", heading_style))

    if np.mean(ret_fixed) > np.mean(ret_rolling):
        winner = "Fixed Strike"
        loser = "Rolling ATM"
        margin = np.mean(ret_fixed) - np.mean(ret_rolling)
    else:
        winner = "Rolling ATM"
        loser = "Fixed Strike"
        margin = np.mean(ret_rolling) - np.mean(ret_fixed)

    # Determine Sharpe winner
    sharpe_fixed = sf['sharpe']
    sharpe_rolling = sr['sharpe']
    sharpe_winner = "Fixed Strike" if sharpe_fixed > sharpe_rolling else "Rolling ATM"

    elements.append(Paragraph(
        f"Based on {NUM_SIMS:,} Monte Carlo simulations over a 12-month horizon:", body_style))

    elements.append(Paragraph(
        f"<b>By Mean Return:</b> <b>{winner}</b> outperforms {loser} by "
        f"{margin:.2f}% on average. The {winner} strategy wins in "
        f"{max(fixed_wins, rolling_wins):.1f}% of individual simulations.",
        body_style))

    elements.append(Paragraph(
        f"<b>By Risk-Adjusted Return (Sharpe):</b> <b>{sharpe_winner}</b> has the higher Sharpe ratio "
        f"(Fixed: {sharpe_fixed:.3f}, Rolling: {sharpe_rolling:.3f}).",
        body_style))

    # Key insight about WHY one wins
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("Key Mechanisms", heading2_style))

    elements.append(Paragraph(
        "<b>Fixed Strike strategy</b> keeps selling the same $100-strike call. When the stock drops, "
        "this call becomes far OTM, collecting very little premium but maintaining full downside stock "
        "exposure. When the stock rises, the call is deep ITM and the position is heavily capped. "
        "The strategy's payoff becomes increasingly asymmetric over time — it participates fully in "
        "drawdowns but has capped upside once the stock moves above the original strike.",
        body_style))

    elements.append(Paragraph(
        "<b>Rolling ATM strategy</b> always writes a fresh ATM call, which maximizes the volatility "
        "risk premium captured each month (ATM options have the highest vega and time value). "
        "After a drawdown, it sells a lower-strike call, collecting meaningful premium even on a "
        "beaten-down stock. After a rally, it resets the cap higher, allowing participation in "
        "further upside. This 'resetting' mechanism means the strategy continuously harvests the "
        "vol risk premium regardless of where the stock is trading.",
        body_style))

    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph(
        f"<b>Overall Recommendation:</b> The <b>{winner}</b> strategy is the superior approach based on "
        f"these simulation results, with better {('mean returns' if winner == 'Rolling ATM' or np.mean(ret_fixed) > np.mean(ret_rolling) else 'risk-adjusted returns')} "
        f"across {NUM_SIMS:,} simulated paths.",
        body_style))

    elements.append(Spacer(1, 0.3*inch))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph(
        "<i>Note: This analysis assumes constant implied volatility, lognormal stock returns, "
        "and no transaction costs or slippage. Real-world results may differ due to volatility "
        "smile dynamics, discrete hedging, bid-ask spreads, and regime changes.</i>",
        ParagraphStyle('Disclaimer', parent=body_style, fontSize=8, textColor=colors.grey)))

    # Build PDF
    doc.build(elements)
    return pdf_path


# ─── Main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # CLI: python simulation.py [drift] [output_pdf_name]
    # e.g.: python simulation.py 0.05 covered_call_report_drift5.pdf
    if len(sys.argv) >= 2:
        MU = float(sys.argv[1])
    pdf_filename = sys.argv[2] if len(sys.argv) >= 3 else "covered_call_report.pdf"

    t0 = time.time()

    print(f"Drift (μ) = {MU*100:.1f}%  |  Output: {pdf_filename}")
    print(f"Generating {NUM_SIMS:,} stock price paths...")
    prices = simulate_paths(NUM_SIMS, TOTAL_DAYS)
    t1 = time.time()
    print(f"  Done in {t1-t0:.1f}s")

    print("Running Strategy 1: Fixed Strike...")
    pnl_fixed, exit_fixed = run_strategy_vectorized(prices, fixed_strike=True)
    t2 = time.time()
    print(f"  Done in {t2-t1:.1f}s")

    print("Running Strategy 2: Rolling ATM...")
    pnl_rolling, exit_rolling = run_strategy_vectorized(prices, fixed_strike=False)
    t3 = time.time()
    print(f"  Done in {t3-t2:.1f}s")

    ret_fixed = pnl_fixed / S0 * 100
    ret_rolling = pnl_rolling / S0 * 100
    final_prices = prices[:, -1]

    # Console output
    print(f"\n{'='*60}")
    print(f"  RESULTS ({NUM_SIMS:,} simulations, drift={MU*100:.1f}%)")
    print(f"{'='*60}")
    print(f"  Fixed Strike  — Mean: {np.mean(ret_fixed):>+.2f}%  Median: {np.median(ret_fixed):>+.2f}%  Std: {np.std(ret_fixed):.2f}%  Sharpe: {np.mean(ret_fixed)/np.std(ret_fixed):.3f}")
    print(f"  Rolling ATM   — Mean: {np.mean(ret_rolling):>+.2f}%  Median: {np.median(ret_rolling):>+.2f}%  Std: {np.std(ret_rolling):.2f}%  Sharpe: {np.mean(ret_rolling)/np.std(ret_rolling):.3f}")

    diff = ret_fixed - ret_rolling
    print(f"\n  Fixed wins: {np.mean(diff>0)*100:.1f}%   Rolling wins: {np.mean(diff<0)*100:.1f}%")
    print(f"  Mean diff (Fixed-Rolling): {np.mean(diff):+.2f}%")

    winner = "FIXED STRIKE" if np.mean(ret_fixed) > np.mean(ret_rolling) else "ROLLING ATM"
    print(f"\n  >>> WINNER by mean return: {winner}")

    print(f"\nGenerating PDF report...")
    pdf_path = generate_pdf_report(ret_fixed, ret_rolling, exit_fixed, exit_rolling, final_prices, pdf_filename)
    t4 = time.time()
    print(f"  Report saved to: {pdf_path}")
    print(f"\nTotal time: {t4-t0:.1f}s")
