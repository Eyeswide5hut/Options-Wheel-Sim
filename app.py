import os
import tempfile
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
from matplotlib import pyplot as plt
import io
import base64
from flask import Flask, render_template, request

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-dev-key')

def calculate_commission_costs(amount, commission_per_contract=5):
    """Calculate commission costs based on position size"""
    num_contracts = max(1, int(amount / 100))
    return num_contracts * commission_per_contract

def calculate_slippage(amount, avg_slippage_percent=0.05):
    """Calculate typical slippage costs"""
    return amount * (avg_slippage_percent / 100)

def adjust_win_probability(base_probability, vix_value):
    """Adjust win probability based on market volatility"""
    vix_adjustment = (30 - vix_value) / 100
    return min(0.75, max(0.45, base_probability + vix_adjustment))

def simulate_vix():
    """Simulate VIX value between 10 and 50"""
    return max(10, min(50, random.gauss(20, 10)))

def calculate_position_size(account_value, volatility):
    """Calculate safe position size based on account value and market conditions"""
    base_risk = 0.02
    vol_adjustment = 30 / volatility
    return account_value * base_risk * vol_adjustment

def calculate_yearly_tax(salary, yearly_trading_results, tax_bands):
    """
    Calculate tax based on salary and trading profits using progressive tax bands.
    """
    net_trading_result = sum(yearly_trading_results)
    taxable_income = salary + max(0, net_trading_result)

    total_tax = 0
    remaining_income = taxable_income

    for lower_bound, upper_bound, rate in sorted(tax_bands):
        if remaining_income > lower_bound:
            taxable_amount = min(remaining_income, upper_bound) - lower_bound
            total_tax += taxable_amount * rate
            remaining_income -= taxable_amount
            
            if remaining_income <= 0:
                break

    return total_tax

def simulate_investment(initial_investment, interest_rate, loss_rate, monthly_contribution, 
                       total_periods, compounding_frequency, utilized_capital, 
                       win_probability, num_simulations, salary, tax_bands):
    """
    Enhanced investment simulation with realistic market factors
    """
    results = []
    yearly_balances_list = []
    post_tax_balances_list = []
    total_tax_paid_list = []

    for _ in range(num_simulations):
        amount = initial_investment
        yearly_trading_results = []
        yearly_balances = []
        post_tax_balances = []
        total_tax_paid = 0
        
        for period in range(total_periods):
            # Add monthly contribution
            if (compounding_frequency == 'monthly' and period % 1 == 0) or \
               (compounding_frequency == 'weekly' and period % 4 == 0):
                amount += monthly_contribution
            
            # Simulate market conditions
            vix = simulate_vix()
            adjusted_win_prob = adjust_win_probability(win_probability, vix)
            
            # Calculate position size based on volatility
            max_position = calculate_position_size(amount, vix)
            capital_to_use = min(amount * utilized_capital, max_position)
            
            # Calculate trading costs
            commission = calculate_commission_costs(capital_to_use)
            slippage = calculate_slippage(capital_to_use)
            
            # Record starting amount
            old_amount = amount
            
            # Simulate trade result
            if random.random() < adjusted_win_prob:
                # Win scenario
                profit = capital_to_use * interest_rate
                amount += profit - commission - slippage
            else:
                # Loss scenario
                loss = capital_to_use * loss_rate
                amount -= loss + commission - slippage
            
            # Ensure amount doesn't go negative
            amount = max(0, amount)
            
            # Record trading result
            trading_result = amount - old_amount
            yearly_trading_results.append(trading_result)
            
            # Annual tax calculations
            if period > 0 and period % (12 if compounding_frequency == 'monthly' else 52) == 0:
                yearly_tax = calculate_yearly_tax(salary, yearly_trading_results, tax_bands)
                yearly_balances.append(amount)
                amount -= yearly_tax
                total_tax_paid += yearly_tax
                post_tax_balances.append(amount)
                yearly_trading_results = []
        
        results.append(amount)
        yearly_balances_list.append(yearly_balances)
        post_tax_balances_list.append(post_tax_balances)
        total_tax_paid_list.append(total_tax_paid)
    
    avg_yearly_balances = np.mean(yearly_balances_list, axis=0)
    avg_post_tax_balances = np.mean(post_tax_balances_list, axis=0)
    pre_tax_final = avg_yearly_balances[-1] if len(avg_yearly_balances) > 0 else initial_investment
    post_tax_final = avg_post_tax_balances[-1] if len(avg_post_tax_balances) > 0 else initial_investment
    avg_total_tax_paid = np.mean(total_tax_paid_list)
    
    return results, avg_yearly_balances, avg_post_tax_balances, pre_tax_final, post_tax_final, avg_total_tax_paid

def calculate_s_and_p_500_growth(initial_investment, monthly_contribution, years, s_and_p_annual_return=0.10):
    """Calculate S&P 500 growth for comparison"""
    months = years * 12
    monthly_growth_rate = (1 + s_and_p_annual_return) ** (1 / 12) - 1
    s_and_p_balance = initial_investment

    for _ in range(months):
        s_and_p_balance *= (1 + monthly_growth_rate)
        s_and_p_balance += monthly_contribution

    return s_and_p_balance

def generate_plot(results, total_invested):
    """Generate distribution plot of simulation results"""
    plt.figure()
    plt.hist(results, bins=20, edgecolor='black')
    plt.axvline(np.mean(results), color='red', linestyle='dashed', linewidth=1, label=f'Average: £{np.mean(results):.2f}')
    plt.axvline(np.median(results), color='green', linestyle='dashed', linewidth=1, label=f'Median: £{np.median(results):.2f}')
    plt.axvline(total_invested, color='blue', linestyle='dashed', linewidth=1, label=f'Total Invested: £{total_invested:.2f}')
    plt.xlabel('Final Amount (Post-Tax)')
    plt.ylabel('Frequency')
    plt.title('Investment Simulation Results')
    plt.legend()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_growth_chart(yearly_balances, post_tax_balances, s_and_p_balances):
    """Generate portfolio growth chart"""
    plt.figure(figsize=(10, 5))
    plt.plot(yearly_balances, label="Pre-Tax Balance", linestyle='dashed', color='blue')
    plt.plot(post_tax_balances, label="Post-Tax Balance", linestyle='solid', color='red')
    plt.plot(s_and_p_balances, label="S&P 500 Growth", linestyle='dotted', color='green')
    plt.xlabel("Years")
    plt.ylabel("Balance (£)")
    plt.title("Portfolio Growth Over Time")
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_return_distribution(results, years):
    """Generate distribution of annual returns"""
    annualized_returns = [(result / years) for result in results]
    plt.figure(figsize=(10, 5))
    plt.hist(annualized_returns, bins=20, edgecolor='black', alpha=0.7, color='green')
    plt.xlabel("Annualized Return (£)")
    plt.ylabel("Frequency")
    plt.title("Annual Return Distribution Across Simulations")

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_tax_comparison_chart(pre_tax_final, post_tax_final, total_tax_paid):
    """Generate tax impact visualization"""
    plt.figure(figsize=(8, 5))
    labels = ['Pre-Tax', 'Post-Tax', 'Total Tax Paid']
    values = [pre_tax_final, post_tax_final, total_tax_paid]
    colors = ['blue', 'red', 'gray']

    plt.bar(labels, values, color=colors)
    plt.xlabel("Category")
    plt.ylabel("Balance (£)")
    plt.title("Tax Impact on Final Balance")

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_cdf_chart(results):
    """Generate cumulative distribution function chart"""
    sorted_results = np.sort(results)
    cdf = np.arange(len(sorted_results)) / float(len(sorted_results))

    plt.figure(figsize=(10, 5))
    plt.plot(sorted_results, cdf, marker='.', linestyle='none', color='purple')
    plt.xlabel("Final Balance (£)")
    plt.ylabel("Cumulative Probability")
    plt.title("CDF of Final Portfolio Value")

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            initial_investment = float(request.form['initial_investment'])
            interest_rate = float(request.form['interest_rate']) / 100
            loss_rate = float(request.form['loss_rate']) / 100
            monthly_contribution = float(request.form['monthly_contribution'])
            years = int(request.form['years'])
            win_probability = float(request.form['win_probability']) / 100
            num_simulations = int(request.form['num_simulations'])
            salary = float(request.form['salary'])
            compounding_frequency = request.form['compounding_frequency']
            utilized_capital = float(request.form['utilized_capital']) / 100
            
            tax_bracket_lower = list(map(float, request.form.getlist('tax_bracket_lower[]')))
            tax_bracket_upper = list(map(float, request.form.getlist('tax_bracket_upper[]')))
            tax_rate = list(map(float, request.form.getlist('tax_rate[]')))
            tax_bands = list(zip(tax_bracket_lower, tax_bracket_upper, [r / 100 for r in tax_rate]))
            
            # Validate tax bands
            for i in range(1, len(tax_bands)):
                if tax_bands[i][0] <= tax_bands[i-1][1]:
                    return "Error: Tax bands must be in ascending order."
            
            total_periods = years * (52 if compounding_frequency == 'weekly' else 12)
            total_invested = initial_investment + (monthly_contribution * (years * 12))
            
            # Run simulation
            results, yearly_balances, post_tax_balances, pre_tax_final, post_tax_final, total_tax_paid = simulate_investment(
                initial_investment, interest_rate, loss_rate, monthly_contribution, total_periods, 
                compounding_frequency, utilized_capital, win_probability, num_simulations, salary, tax_bands
            )
            
            # Calculate S&P 500 comparison
            s_and_p_balances = []
            for year in range(1, years + 1):
                s_and_p_balances.append(calculate_s_and_p_500_growth(initial_investment, monthly_contribution, year))

            # Generate charts
            growth_chart_data = generate_growth_chart(yearly_balances, post_tax_balances, s_and_p_balances)
            s_and_p_final = calculate_s_and_p_500_growth(initial_investment, monthly_contribution, years)
            plot_data = generate_plot(results, total_invested)
            return_distribution_data = generate_return_distribution(results, years)
            tax_comparison_data = generate_tax_comparison_chart(pre_tax_final, post_tax_final, total_tax_paid)
            cdf_chart_data = generate_cdf_chart(results)
            
            return render_template('index.html',
                               high_end=f"£{max(results):.2f}",
                               low_end=f"£{min(results):.2f}",
                               average=f"£{np.mean(results):.2f}",
                               median=f"£{np.median(results):.2f}",
                               total_invested=f"£{total_invested:.2f}",
                               s_and_p_final=f"£{s_and_p_final:.2f}",
                               plot_data=plot_data,
                               growth_chart_data=growth_chart_data,
                               return_distribution_data=return_distribution_data,
                               tax_comparison_data=tax_comparison_data,
                               cdf_chart_data=cdf_chart_data,
                               pre_tax_final=f"£{pre_tax_final:.2f}",
                               post_tax_final=f"£{post_tax_final:.2f}",
                               total_tax_paid=f"£{total_tax_paid:.2f}")
        except ValueError as e:
            return render_template('index.html', error=f"Error: {e}")
    
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
