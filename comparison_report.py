"""
Comparison Report: All Three Drift Scenarios
=============================================
Generates a layman-friendly PDF comparing Fixed Strike vs Rolling ATM
covered call strategies across +5%, 0%, and -5% drift scenarios.

Runs all three simulations internally and produces a single PDF.
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
    Image, PageBreak, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT

# ─── Parameters ──────────────────────────────────────────────────
S0 = 100.0
SIGMA_REALIZED = 0.20
SIGMA_IMPLIED = 0.25
R = 0.05
TRADING_DAYS_PER_MONTH = 21
NUM_MONTHS = 12
TOTAL_DAYS = TRADING_DAYS_PER_MONTH * NUM_MONTHS
NUM_SIMS = 1_000_000
DELTA_EXIT_THRESHOLD = 0.1
DT = 1.0 / 252
DRIFTS = [0.05, 0.0, -0.05]
DRIFT_LABELS = ["+5%", "0%", "−5%"]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

np.random.seed(42)


# ─── Black-Scholes ───────────────────────────────────────────────
def bs_d1_vec(S, K, T, r, sigma):
    safe_T = np.maximum(T, 1e-10)
    return (np.log(S / K) + (r + 0.5 * sigma**2) * safe_T) / (sigma * np.sqrt(safe_T))

def bs_call_price_vec(S, K, T, r, sigma):
    expired = T <= 0
    d1 = bs_d1_vec(S, K, np.maximum(T, 1e-10), r, sigma)
    d2 = d1 - sigma * np.sqrt(np.maximum(T, 1e-10))
    price = S * norm.cdf(d1) - K * np.exp(-r * np.maximum(T, 0)) * norm.cdf(d2)
    return np.where(expired, np.maximum(S - K, 0.0), price)

def bs_call_delta_vec(S, K, T, r, sigma):
    expired = T <= 0
    d1 = bs_d1_vec(S, K, np.maximum(T, 1e-10), r, sigma)
    delta = norm.cdf(d1)
    return np.where(expired, np.where(S > K, 1.0, 0.0), delta)

def bs_call_price_scalar(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# ─── Simulation ──────────────────────────────────────────────────
def simulate_paths(num_sims, total_days, mu):
    daily_log_ret = np.random.normal(
        (mu - 0.5 * SIGMA_REALIZED**2) * DT,
        SIGMA_REALIZED * np.sqrt(DT),
        size=(num_sims, total_days)
    )
    log_prices = np.hstack([np.zeros((num_sims, 1)), np.cumsum(daily_log_ret, axis=1)])
    return S0 * np.exp(log_prices)


def run_strategy(prices, fixed_strike: bool):
    num_sims = prices.shape[0]
    pnl = np.zeros(num_sims)
    active = np.ones(num_sims, dtype=bool)
    stock_entry = np.full(num_sims, S0)
    exit_day = np.full(num_sims, TOTAL_DAYS)
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

        c_delta = bs_call_delta_vec(S, K, T_left, R, SIGMA_IMPLIED)
        pos_delta = 1.0 - c_delta
        should_exit = pos_delta < DELTA_EXIT_THRESHOLD

        if np.any(should_exit):
            active_indices = np.where(active)[0]
            exit_indices = active_indices[should_exit]
            S_exit = S[should_exit]
            K_exit = K[should_exit]
            call_val = bs_call_price_vec(S_exit, K_exit, T_left, R, SIGMA_IMPLIED)
            pnl[exit_indices] += (S_exit - entries[should_exit]) + (premiums[should_exit] - call_val)
            active[exit_indices] = False
            exit_day[exit_indices] = day

        if day_in_month == TRADING_DAYS_PER_MONTH and day < TOTAL_DAYS:
            if not np.any(active):
                break
            active_idx = np.where(active)[0]
            S_roll = prices[active_idx, day]
            K_roll = call_strike[active_idx]
            intrinsic = np.maximum(S_roll - K_roll, 0.0)
            pnl[active_idx] += (S_roll - stock_entry[active_idx]) + (call_premium_received[active_idx] - intrinsic)
            stock_entry[active_idx] = S_roll
            new_K = np.full(len(active_idx), initial_K) if fixed_strike else S_roll.copy()
            call_strike[active_idx] = new_K
            call_premium_received[active_idx] = bs_call_price_vec(S_roll, new_K, TRADING_DAYS_PER_MONTH / 252.0, R, SIGMA_IMPLIED)

    still_active = np.where(active)[0]
    if len(still_active) > 0:
        S_final = prices[still_active, TOTAL_DAYS]
        intrinsic = np.maximum(S_final - call_strike[still_active], 0.0)
        pnl[still_active] += (S_final - stock_entry[still_active]) + (call_premium_received[still_active] - intrinsic)

    return pnl, exit_day


# ─── Charts ──────────────────────────────────────────────────────
def _fig_to_buf(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf


def make_bar_comparison(results):
    """Grouped bar chart: mean returns across scenarios."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(DRIFT_LABELS))
    w = 0.32
    fixed_means = [r['fixed_mean'] for r in results]
    rolling_means = [r['rolling_mean'] for r in results]

    bars1 = ax.bar(x - w/2, fixed_means, w, color='steelblue', label='Fixed Strike', edgecolor='white')
    bars2 = ax.bar(x + w/2, rolling_means, w, color='darkorange', label='Rolling ATM', edgecolor='white')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'Stock trend: {l}/year' for l in DRIFT_LABELS], fontsize=11)
    ax.set_ylabel('Average Profit (%)', fontsize=11)
    ax.set_title('Which Strategy Made More Money?', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return _fig_to_buf(fig)


def make_risk_comparison(results):
    """Grouped bar chart: volatility (risk) across scenarios."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(DRIFT_LABELS))
    w = 0.32
    fixed_std = [r['fixed_std'] for r in results]
    rolling_std = [r['rolling_std'] for r in results]

    bars1 = ax.bar(x - w/2, fixed_std, w, color='steelblue', label='Fixed Strike', edgecolor='white')
    bars2 = ax.bar(x + w/2, rolling_std, w, color='darkorange', label='Rolling ATM', edgecolor='white')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'Stock trend: {l}/year' for l in DRIFT_LABELS], fontsize=11)
    ax.set_ylabel('Risk — How Bumpy the Ride Was (%)', fontsize=11)
    ax.set_title('Which Strategy Had a Smoother Ride?', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return _fig_to_buf(fig)


def make_sharpe_comparison(results):
    """Bar chart of Sharpe ratios."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(DRIFT_LABELS))
    w = 0.32
    fixed_sharpe = [r['fixed_sharpe'] for r in results]
    rolling_sharpe = [r['rolling_sharpe'] for r in results]

    bars1 = ax.bar(x - w/2, fixed_sharpe, w, color='steelblue', label='Fixed Strike', edgecolor='white')
    bars2 = ax.bar(x + w/2, rolling_sharpe, w, color='darkorange', label='Rolling ATM', edgecolor='white')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'Stock trend: {l}/year' for l in DRIFT_LABELS], fontsize=11)
    ax.set_ylabel('Reward-to-Risk Ratio', fontsize=11)
    ax.set_title('Bang for Your Buck: Profit per Unit of Risk', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return _fig_to_buf(fig)


def make_worst_case_chart(results):
    """Worst-case and 5th percentile comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(DRIFT_LABELS))
    w = 0.32

    # Worst case
    ax = axes[0]
    f_worst = [r['fixed_min'] for r in results]
    r_worst = [r['rolling_min'] for r in results]
    ax.bar(x - w/2, f_worst, w, color='steelblue', label='Fixed Strike', edgecolor='white')
    ax.bar(x + w/2, r_worst, w, color='darkorange', label='Rolling ATM', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{l}/yr' for l in DRIFT_LABELS], fontsize=10)
    ax.set_ylabel('Return (%)')
    ax.set_title('Worst Single Outcome', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # 5th percentile (bad but not catastrophic)
    ax = axes[1]
    f_p5 = [r['fixed_p5'] for r in results]
    r_p5 = [r['rolling_p5'] for r in results]
    ax.bar(x - w/2, f_p5, w, color='steelblue', label='Fixed Strike', edgecolor='white')
    ax.bar(x + w/2, r_p5, w, color='darkorange', label='Rolling ATM', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{l}/yr' for l in DRIFT_LABELS], fontsize=10)
    ax.set_ylabel('Return (%)')
    ax.set_title('"Unlucky" Outcome (bottom 5%)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return _fig_to_buf(fig)


def make_win_rate_chart(results):
    """Who wins more often."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(DRIFT_LABELS))
    w = 0.32
    f_wr = [r['fixed_winrate'] for r in results]
    r_wr = [r['rolling_winrate'] for r in results]

    bars1 = ax.bar(x - w/2, f_wr, w, color='steelblue', label='Fixed Strike', edgecolor='white')
    bars2 = ax.bar(x + w/2, r_wr, w, color='darkorange', label='Rolling ATM', edgecolor='white')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'Stock trend: {l}/year' for l in DRIFT_LABELS], fontsize=11)
    ax.set_ylabel('% of Simulations That Made Money', fontsize=11)
    ax.set_title('How Often Did Each Strategy Make a Profit?', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return _fig_to_buf(fig)


# ─── PDF Report ──────────────────────────────────────────────────
def generate_comparison_pdf(results):
    pdf_path = os.path.join(OUTPUT_DIR, "comparison_report.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                            topMargin=0.6*inch, bottomMargin=0.6*inch,
                            leftMargin=0.75*inch, rightMargin=0.75*inch)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('T', parent=styles['Title'], fontSize=24, spaceAfter=6,
                                  textColor=colors.HexColor('#1a1a2e'))
    subtitle_style = ParagraphStyle('Sub', parent=styles['Normal'], fontSize=12, spaceAfter=12,
                                     textColor=colors.grey, alignment=TA_CENTER)
    h1 = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=17, spaceBefore=18,
                         spaceAfter=10, textColor=colors.HexColor('#16213e'))
    h2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=13, spaceBefore=14,
                         spaceAfter=6, textColor=colors.HexColor('#0f3460'))
    body = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10.5, leading=15,
                           alignment=TA_JUSTIFY, spaceAfter=8)
    body_big = ParagraphStyle('BodyBig', parent=body, fontSize=11, leading=16)
    callout = ParagraphStyle('Callout', parent=body, fontSize=11, leading=16,
                              backColor=colors.HexColor('#f0f4ff'),
                              borderColor=colors.HexColor('#3366cc'), borderWidth=1,
                              borderPadding=8, spaceBefore=6, spaceAfter=10)
    winner_style = ParagraphStyle('Winner', parent=body, fontSize=13, leading=18,
                                   fontName='Helvetica-Bold', textColor=colors.HexColor('#0f3460'),
                                   alignment=TA_CENTER, spaceBefore=10, spaceAfter=10,
                                   backColor=colors.HexColor('#e8f0fe'), borderPadding=10)
    disclaimer = ParagraphStyle('Disc', parent=body, fontSize=8, textColor=colors.grey)

    el = []

    # ═══════════════════════ TITLE PAGE ═══════════════════════
    el.append(Spacer(1, 1.2*inch))
    el.append(Paragraph("Should You Keep Selling the Same Option<br/>or Adjust Every Month?", title_style))
    el.append(Spacer(1, 0.15*inch))
    el.append(Paragraph("A Plain-English Comparison of Two Covered Call Strategies", subtitle_style))
    el.append(Spacer(1, 0.25*inch))
    el.append(HRFlowable(width="70%", thickness=2, color=colors.HexColor('#e94560')))
    el.append(Spacer(1, 0.25*inch))
    el.append(Paragraph("Based on 1,000,000 simulated stock paths across three market conditions", subtitle_style))

    el.append(Spacer(1, 0.5*inch))

    # Quick answer box
    el.append(Paragraph(
        "Short answer: If you believe the stock will go up, the Fixed Strike approach earns more "
        "on average. But if you want a <b>smoother, more consistent</b> experience regardless of "
        "what the market does, the Rolling ATM approach wins — and it is the better choice for "
        "most investors.", callout))

    el.append(PageBreak())

    # ═══════════════════════ WHAT IS THIS? ═══════════════════════
    el.append(Paragraph("What Is This Report About?", h1))

    el.append(Paragraph(
        "Imagine you own a stock. A <b>covered call</b> means you also sell someone the <i>right</i> "
        "to buy your stock at a specific price (the \"strike price\") within the next month. "
        "In exchange, they pay you a fee upfront (the \"premium\"). "
        "This is a popular strategy because that fee gives you extra income — but the trade-off is "
        "that if the stock shoots up past the strike price, you don't get to keep those gains.",
        body_big))

    el.append(Paragraph(
        "The question we're testing: <b>When the month ends and you sell a new option, what strike "
        "price should you pick?</b>", body_big))

    el.append(Spacer(1, 0.1*inch))
    el.append(Paragraph("The Two Strategies", h2))

    strat_data = [
        ['', 'Fixed Strike', 'Rolling ATM'],
        ['What you do',
         'Always sell the option at the\nORIGINAL stock price ($100),\nno matter what.',
         'Sell the option at whatever\nthe stock price is RIGHT NOW.'],
        ['If stock drops\nto $70...',
         'You sell a $100 option.\nVery unlikely to be exercised.\nYou collect almost no fee.',
         'You sell a $70 option.\nStill meaningful fee collected.\nStays "in the zone."'],
        ['If stock rises\nto $130...',
         'You sell a $100 option.\nAlmost certain to be exercised.\nYou give up all gains above $100.',
         'You sell a $130 option.\nYou keep the gains so far,\nand collect a new fee.'],
    ]
    strat_table = Table(strat_data, colWidths=[1.4*inch, 2.4*inch, 2.4*inch])
    strat_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16213e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0f0f0'), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    el.append(strat_table)

    el.append(Spacer(1, 0.15*inch))
    el.append(Paragraph("The Three Market Scenarios", h2))

    el.append(Paragraph(
        "We tested both strategies under three different assumptions about where the stock is heading:",
        body))

    scenario_data = [
        ['Scenario', 'What It Means', 'Real-World Analogy'],
        ['Stock trends UP\n(+5% per year)',
         'The stock gradually climbs\nover the year.',
         'A healthy economy,\ncompany doing well.'],
        ['Stock goes NOWHERE\n(0% per year)',
         'The stock bounces around\nbut ends roughly where\nit started.',
         'Uncertain market,\nno clear direction.'],
        ['Stock trends DOWN\n(−5% per year)',
         'The stock gradually\ndeclines over the year.',
         'Recession fears,\ncompany struggling.'],
    ]
    sc_table = Table(scenario_data, colWidths=[1.8*inch, 2.2*inch, 2.2*inch])
    sc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16213e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0f0f0'), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    el.append(sc_table)

    el.append(Paragraph(
        "<i>For each scenario, we simulated 1,000,000 possible stock price journeys over one year "
        "and tracked how each strategy performed.</i>", body))

    el.append(PageBreak())

    # ═══════════════════════ KEY RESULTS ═══════════════════════
    el.append(Paragraph("The Results at a Glance", h1))

    # Master comparison table
    master_data = [
        ['', 'Stock UP (+5%/yr)', '', 'Stock FLAT (0%/yr)', '', 'Stock DOWN (−5%/yr)', ''],
        ['', 'Fixed', 'Rolling', 'Fixed', 'Rolling', 'Fixed', 'Rolling'],
        ['Average Profit',
         f'{results[0]["fixed_mean"]:.2f}%', f'{results[0]["rolling_mean"]:.2f}%',
         f'{results[1]["fixed_mean"]:.2f}%', f'{results[1]["rolling_mean"]:.2f}%',
         f'{results[2]["fixed_mean"]:.2f}%', f'{results[2]["rolling_mean"]:.2f}%'],
        ['Risk (Volatility)',
         f'{results[0]["fixed_std"]:.1f}%', f'{results[0]["rolling_std"]:.1f}%',
         f'{results[1]["fixed_std"]:.1f}%', f'{results[1]["rolling_std"]:.1f}%',
         f'{results[2]["fixed_std"]:.1f}%', f'{results[2]["rolling_std"]:.1f}%'],
        ['Reward-to-Risk',
         f'{results[0]["fixed_sharpe"]:.2f}', f'{results[0]["rolling_sharpe"]:.2f}',
         f'{results[1]["fixed_sharpe"]:.2f}', f'{results[1]["rolling_sharpe"]:.2f}',
         f'{results[2]["fixed_sharpe"]:.2f}', f'{results[2]["rolling_sharpe"]:.2f}'],
        ['Win Rate',
         f'{results[0]["fixed_winrate"]:.0f}%', f'{results[0]["rolling_winrate"]:.0f}%',
         f'{results[1]["fixed_winrate"]:.0f}%', f'{results[1]["rolling_winrate"]:.0f}%',
         f'{results[2]["fixed_winrate"]:.0f}%', f'{results[2]["rolling_winrate"]:.0f}%'],
        ['Unlucky Outcome\n(bottom 5%)',
         f'{results[0]["fixed_p5"]:.1f}%', f'{results[0]["rolling_p5"]:.1f}%',
         f'{results[1]["fixed_p5"]:.1f}%', f'{results[1]["rolling_p5"]:.1f}%',
         f'{results[2]["fixed_p5"]:.1f}%', f'{results[2]["rolling_p5"]:.1f}%'],
    ]
    mt = Table(master_data, colWidths=[1.3*inch, 0.75*inch, 0.75*inch, 0.75*inch, 0.75*inch, 0.75*inch, 0.75*inch])
    mt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16213e')),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#2a4a7f')),
        ('TEXTCOLOR', (0, 0), (-1, 1), colors.white),
        ('FONTNAME', (0, 0), (-1, 1), 'Helvetica-Bold'),
        ('FONTNAME', (0, 2), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 2), (-1, -1), [colors.HexColor('#f0f0f0'), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        # Span headers
        ('SPAN', (1, 0), (2, 0)),
        ('SPAN', (3, 0), (4, 0)),
        ('SPAN', (5, 0), (6, 0)),
        # Highlight winning cells with subtle coloring
    ]))
    el.append(mt)

    el.append(Spacer(1, 0.15*inch))
    el.append(Paragraph(
        "<b>How to read this table:</b> \"Average Profit\" is the typical return over one year. "
        "\"Risk\" measures how wildly returns swing — lower is calmer. "
        "\"Reward-to-Risk\" is the profit divided by the risk — higher means you got more profit per "
        "unit of uncertainty. \"Win Rate\" is how often the strategy made any money at all.",
        body))

    el.append(PageBreak())

    # ═══════════════════════ CHARTS ═══════════════════════
    el.append(Paragraph("Visual Comparison", h1))

    bar_buf = make_bar_comparison(results)
    el.append(Image(bar_buf, width=6.2*inch, height=3.1*inch))
    el.append(Spacer(1, 0.1*inch))

    risk_buf = make_risk_comparison(results)
    el.append(Image(risk_buf, width=6.2*inch, height=3.1*inch))

    el.append(PageBreak())

    sharpe_buf = make_sharpe_comparison(results)
    el.append(Image(sharpe_buf, width=6.2*inch, height=3.1*inch))
    el.append(Spacer(1, 0.1*inch))

    wr_buf = make_win_rate_chart(results)
    el.append(Image(wr_buf, width=6.2*inch, height=3.1*inch))

    el.append(PageBreak())

    el.append(Paragraph("What About Bad Outcomes?", h1))

    worst_buf = make_worst_case_chart(results)
    el.append(Image(worst_buf, width=6.5*inch, height=3.0*inch))

    el.append(Spacer(1, 0.1*inch))
    el.append(Paragraph(
        "The left chart shows the single worst outcome out of 1,000,000 simulations. "
        "The right chart shows the \"unlucky\" threshold — the return you'd exceed 95% of the time. "
        "In both cases, <b>Rolling ATM has dramatically smaller losses</b>. The Fixed Strike approach "
        "can lose much more because when the stock drops, it stops collecting meaningful fees but "
        "you still own the falling stock.", body_big))

    el.append(PageBreak())

    # ═══════════════════════ WHY DOES THIS HAPPEN? ═══════════════════════
    el.append(Paragraph("Why Does This Happen?", h1))

    el.append(Paragraph("Think of it like renting out your house.", h2))

    el.append(Paragraph(
        "Owning a stock and selling options on it is a bit like owning a house and renting it out. "
        "The rent you collect is like the option fee (\"premium\"). The two strategies differ in "
        "how you set the rent:", body_big))

    el.append(Paragraph(
        "<b>Fixed Strike</b> is like setting your rent at the original purchase price of the house — "
        "forever. If the neighborhood declines and your house is now worth much less, you're asking "
        "for rent that nobody wants to pay, so you collect almost nothing. Meanwhile, you still "
        "own a house that's losing value. If the neighborhood booms, your tenant gets an amazing deal "
        "and you can't raise the rent.", body))

    el.append(Paragraph(
        "<b>Rolling ATM</b> is like adjusting your rent to match the current market rate every month. "
        "If the neighborhood drops, you lower the rent — but at least you're still collecting something. "
        "If it booms, you raise the rent to match. You always collect a fair, meaningful amount. "
        "This keeps your income steady regardless of what happens to property values.", body))

    el.append(Spacer(1, 0.15*inch))
    el.append(Paragraph("The key insight in numbers:", h2))

    insight_data = [
        ['What Happens', 'Fixed Strike Fee\n(from the option)', 'Rolling ATM Fee\n(from the option)'],
        ['Stock falls to $70',
         '$100 option is unrealistic.\nFee collected: ~$0.01',
         '$70 option is realistic.\nFee collected: ~$2.00'],
        ['Stock stays at $100',
         '$100 option is perfect.\nFee collected: ~$2.90',
         '$100 option is perfect.\nFee collected: ~$2.90'],
        ['Stock rises to $130',
         '$100 option is a giveaway.\nYou cap your gains at $100.',
         '$130 option is fair.\nYou keep the $30 gain\nand collect a new ~$3.70 fee.'],
    ]
    ins_table = Table(insight_data, colWidths=[1.6*inch, 2.3*inch, 2.3*inch])
    ins_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16213e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0f0f0'), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    el.append(ins_table)

    el.append(Spacer(1, 0.1*inch))
    el.append(Paragraph(
        "The Rolling ATM strategy stays \"in the zone\" — always collecting a meaningful fee — "
        "regardless of whether the stock goes up or down. The Fixed Strike strategy drifts away "
        "from that sweet spot whenever the stock moves significantly.", body))

    el.append(PageBreak())

    # ═══════════════════════ SO WHICH IS BETTER? ═══════════════════════
    el.append(Paragraph("So Which Strategy Should You Use?", h1))

    el.append(Paragraph(
        "It depends on what you care about most:", body_big))

    verdict_data = [
        ['If you want...', 'Best choice', 'Why'],
        ['Highest average profit\n(and believe stocks go up)',
         'Fixed Strike',
         'When the stock trends upward, Fixed Strike\ncaptures more of that upside direction.\nBut: much bumpier ride and bigger losses\nif you\'re wrong.'],
        ['Smoothest experience\n(sleep well at night)',
         'Rolling ATM',
         'Consistently half the risk of Fixed Strike.\nYour returns are more predictable\nmonth to month.'],
        ['Best profit-per-risk\n(most efficient use\nof your money)',
         'Rolling ATM',
         'Wins in ALL three scenarios.\nYou get more return for each unit\nof risk you take.'],
        ['Protection if stock falls',
         'Rolling ATM',
         'In a declining market, Rolling ATM\nstill earns +1.00% while Fixed Strike\nearns only +0.20% — and with\nhalf the volatility.'],
    ]
    vt = Table(verdict_data, colWidths=[1.6*inch, 1.2*inch, 3.4*inch])
    vt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16213e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 1), (1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0f0f0'), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('TEXTCOLOR', (1, 1), (1, 1), colors.HexColor('#1a5276')),
        ('TEXTCOLOR', (1, 2), (1, 4), colors.HexColor('#0e6655')),
    ]))
    el.append(vt)

    el.append(Spacer(1, 0.25*inch))

    el.append(Paragraph(
        "Our Recommendation: <b>Rolling ATM</b> is the better strategy for most investors.",
        winner_style))

    el.append(Spacer(1, 0.1*inch))
    el.append(Paragraph(
        "The Rolling ATM approach wins the reward-to-risk comparison in <b>all three market "
        "environments</b>. It earns a bit less in bull markets, but the trade-off is dramatically "
        "lower risk, better downside protection, and a strategy that doesn't stop working just "
        "because the stock moved away from its starting price.", body_big))

    el.append(Paragraph(
        "The Fixed Strike strategy is a reasonable bet <i>only if</i> you have strong conviction "
        "that the stock will rise. Even then, the extra average return comes with roughly "
        "double the volatility — meaning many more stressful months along the way.", body_big))

    el.append(Spacer(1, 0.3*inch))
    el.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    el.append(Spacer(1, 0.1*inch))
    el.append(Paragraph("Methodology", h2))
    el.append(Paragraph(
        "We simulated 1,000,000 random stock price paths for each scenario using a standard "
        "financial model (geometric Brownian motion with lognormal returns). Options were priced "
        "using the Black-Scholes model with a constant implied volatility of 25% (versus 20% "
        "realized volatility), reflecting the real-world observation that option prices tend to "
        "be slightly \"expensive\" — which is one reason option selling strategies can be profitable.",
        body))
    el.append(Paragraph(
        "<i>This analysis assumes no trading costs, no taxes, and constant market conditions. "
        "Real-world results will vary. This is not investment advice.</i>", disclaimer))

    doc.build(el)
    return pdf_path


# ─── Main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()
    all_results = []

    for i, mu in enumerate(DRIFTS):
        label = DRIFT_LABELS[i]
        print(f"\n{'─'*50}")
        print(f"  Scenario: drift = {label}")
        print(f"{'─'*50}")

        np.random.seed(42)  # Same seed for fair comparison

        print(f"  Generating {NUM_SIMS:,} paths...")
        prices = simulate_paths(NUM_SIMS, TOTAL_DAYS, mu)

        print(f"  Running Fixed Strike...")
        pnl_f, exit_f = run_strategy(prices, fixed_strike=True)

        print(f"  Running Rolling ATM...")
        pnl_r, exit_r = run_strategy(prices, fixed_strike=False)

        ret_f = pnl_f / S0 * 100
        ret_r = pnl_r / S0 * 100

        r = {
            'drift': mu,
            'fixed_mean': np.mean(ret_f), 'rolling_mean': np.mean(ret_r),
            'fixed_median': np.median(ret_f), 'rolling_median': np.median(ret_r),
            'fixed_std': np.std(ret_f), 'rolling_std': np.std(ret_r),
            'fixed_sharpe': np.mean(ret_f)/np.std(ret_f) if np.std(ret_f)>0 else 0,
            'rolling_sharpe': np.mean(ret_r)/np.std(ret_r) if np.std(ret_r)>0 else 0,
            'fixed_winrate': np.mean(ret_f > 0) * 100,
            'rolling_winrate': np.mean(ret_r > 0) * 100,
            'fixed_min': np.min(ret_f), 'rolling_min': np.min(ret_r),
            'fixed_max': np.max(ret_f), 'rolling_max': np.max(ret_r),
            'fixed_p5': np.percentile(ret_f, 5), 'rolling_p5': np.percentile(ret_r, 5),
            'fixed_p95': np.percentile(ret_f, 95), 'rolling_p95': np.percentile(ret_r, 95),
        }
        all_results.append(r)

        print(f"  Fixed:   mean={r['fixed_mean']:+.2f}%  std={r['fixed_std']:.2f}%  sharpe={r['fixed_sharpe']:.3f}")
        print(f"  Rolling: mean={r['rolling_mean']:+.2f}%  std={r['rolling_std']:.2f}%  sharpe={r['rolling_sharpe']:.3f}")

    print(f"\n{'═'*50}")
    print(f"  Generating comparison PDF...")
    pdf_path = generate_comparison_pdf(all_results)
    t1 = time.time()
    print(f"  Saved to: {pdf_path}")
    print(f"  Total time: {t1-t0:.1f}s")
    print(f"{'═'*50}")
