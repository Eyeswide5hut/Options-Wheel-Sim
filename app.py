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

# Function to calculate income tax based on salary and trading profits using tax bands
def calculate_yearly_tax(salary, yearly_trading_results, tax_bands):
    """
    Calculate tax based on salary and trading profits using progressive tax bands.
    
    Parameters:
    - salary: Fixed annual salary
    - yearly_trading_results: List of trading profits/losses over the year
    - tax_bands: List of tuples (lower_bound, upper_bound, tax_rate)
    
    Returns:
    - total_tax: Total tax owed for the year
    """
    net_trading_result = sum(yearly_trading_results)
    taxable_income = salary + max(0, net_trading_result)  # Only add positive trading profit

    total_tax = 0
    remaining_income = taxable_income

    for lower_bound, upper_bound, rate in sorted(tax_bands):
        if remaining_income > lower_bound:
            taxable_amount = min(remaining_income, upper_bound) - lower_bound
            total_tax += taxable_amount * rate
            remaining_income -= taxable_amount  # Reduce only this bracket's taxable portion
            
            if remaining_income <= 0:
                break  # Stop once all income is taxed

    return total_tax

def simulate_investment(initial_investment, interest_rate, loss_rate, monthly_contribution, 
                       total_periods, compounding_frequency, utilized_capital, 
                       win_probability, num_simulations, salary, tax_bands):
    """
    Simulate investment returns with proper tax calculations.
    """
    results = []
    yearly_balances_list = []
    post_tax_balances_list = []
    total_tax_paid_list = []

    for _ in range(num_simulations):
        amount = initial_investment
        yearly_trading_results = []  # Store all trading results for the year
        yearly_balances = []
        post_tax_balances = []
        total_tax_paid = 0
        
        for period in range(total_periods):
            # Add monthly contribution
            if (compounding_frequency == 'monthly' and period % 1 == 0) or \
               (compounding_frequency == 'weekly' and period % 4 == 0):
                amount += monthly_contribution
            
            # Calculate trading result for this period
            capital_to_use = amount * utilized_capital
            old_amount = amount
            
            if random.random() < win_probability:
                amount += capital_to_use * interest_rate
            else:
                amount -= capital_to_use * loss_rate
            
            # Record the trading result (profit or loss)
            trading_result = amount - old_amount
            yearly_trading_results.append(trading_result)
            
            # At the end of each year, calculate and apply tax
            if period > 0 and period % (12 if compounding_frequency == 'monthly' else 52) == 0:
                # Calculate tax based on salary and net trading results
                yearly_tax = calculate_yearly_tax(salary, yearly_trading_results, tax_bands)
                
                # Store pre-tax balance
                yearly_balances.append(amount)
                
                # Apply tax
                amount -= yearly_tax
                total_tax_paid += yearly_tax
                
                # Store post-tax balance
                post_tax_balances.append(amount)
                
                # Reset yearly trading results for next year
                yearly_trading_results = []
        
        results.append(amount)
        yearly_balances_list.append(yearly_balances)
        post_tax_balances_list.append(post_tax_balances)
        total_tax_paid_list.append(total_tax_paid)
    
    # Calculate averages for return values
    avg_yearly_balances = np.mean(yearly_balances_list, axis=0)
    avg_post_tax_balances = np.mean(post_tax_balances_list, axis=0)
    
    pre_tax_final = avg_yearly_balances[-1] if len(avg_yearly_balances) > 0 else initial_investment
    post_tax_final = avg_post_tax_balances[-1] if len(avg_post_tax_balances) > 0 else initial_investment
    avg_total_tax_paid = np.mean(total_tax_paid_list)
    
    return results, avg_yearly_balances, avg_post_tax_balances, pre_tax_final, post_tax_final, avg_total_tax_paid

def calculate_s_and_p_500_growth(initial_investment, monthly_contribution, years, s_and_p_annual_return=0.10):
    months = years * 12
    monthly_growth_rate = (1 + s_and_p_annual_return) ** (1 / 12) - 1  
    s_and_p_balance = initial_investment

    for _ in range(months):
        s_and_p_balance *= (1 + monthly_growth_rate)
        s_and_p_balance += monthly_contribution

    return s_and_p_balance

def generate_plot(results, total_invested):
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
            
            for i in range(1, len(tax_bands)):
                if tax_bands[i][0] <= tax_bands[i-1][1]:
                    return "Error: Tax bands must be in ascending order."
            
            total_periods = years * (52 if compounding_frequency == 'weekly' else 12)
            total_invested = initial_investment + (monthly_contribution * (years * 12))
            
            results, yearly_balances, post_tax_balances, pre_tax_final, post_tax_final, total_tax_paid = simulate_investment(
                initial_investment, interest_rate, loss_rate, monthly_contribution, total_periods, compounding_frequency, utilized_capital, win_probability, num_simulations, salary, tax_bands
            )
            
            s_and_p_balances = []
            for year in range(1, years + 1):
                s_and_p_balances.append(calculate_s_and_p_500_growth(initial_investment, monthly_contribution, year))

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
