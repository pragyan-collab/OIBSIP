"""
UNEMPLOYMENT ANALYSIS WITH COVID-19 IMPACT
Author: Pragyan Kumar Beheruk
Internship Task 2 - OIBSIP

This project analyzes unemployment data in India, focusing on:
- Unemployment trends over time
- COVID-19 pandemic impact (March-April 2020)
- Regional and area-wise (Rural/Urban) comparison
- Labour participation rates
- Employment statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for professional visualizations
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)

# ==========================================
# 1. LOAD AND EXPLORE DATA
# ==========================================

print("=" * 70)
print("UNEMPLOYMENT ANALYSIS WITH COVID-19 IMPACT")
print("=" * 70)
print("\n[Step 1] Loading and Exploring Datasets...")

# Load unemployment data
df = pd.read_csv('Unemployment in India.csv')

# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'].str.strip(), format='%d-%m-%Y')

# Sort by date for time series analysis
df = df.sort_values('Date').reset_index(drop=True)

print(f"\n✓ Dataset loaded successfully!")
print(f"  Shape: {df.shape}")
print(f"  Rows: {len(df)}")
print(f"  Columns: {list(df.columns)}")

print(f"\n[Dataset Overview]")
print(df.head(10))

print(f"\n[Dataset Info]")
print(f"  Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"  Total Months: {len(df['Date'].unique())}")
print(f"  States/Regions: {df['Region'].nunique()}")
print(f"  Areas: {df['Area'].unique()}")

print(f"\n[Data Statistics]")
print(df[['Estimated Unemployment Rate (%)', 'Estimated Employed', 'Estimated Labour Participation Rate (%)']].describe())

print(f"\n[Missing Values Check]")
print(df.isnull().sum())

# ==========================================
# 2. DATA PREPROCESSING
# ==========================================

print("\n" + "=" * 70)
print("[Step 2] Data Preprocessing...")

# Convert numeric columns
df['Estimated Unemployment Rate (%)'] = pd.to_numeric(df['Estimated Unemployment Rate (%)'], errors='coerce')
df['Estimated Employed'] = pd.to_numeric(df['Estimated Employed'], errors='coerce')
df['Estimated Labour Participation Rate (%)'] = pd.to_numeric(df['Estimated Labour Participation Rate (%)'], errors='coerce')

# Remove rows with missing unemployment rate
df_clean = df.dropna(subset=['Estimated Unemployment Rate (%)'])

print(f"✓ Data cleaned and preprocessed")
print(f"  Rows after cleaning: {len(df_clean)}")
print(f"  Removed rows: {len(df) - len(df_clean)}")

# ==========================================
# 3. COVID-19 IMPACT ANALYSIS
# ==========================================

print("\n" + "=" * 70)
print("[Step 3] COVID-19 Impact Analysis...")

# Define COVID-19 periods
pre_covid = df_clean[df_clean['Date'] < '2020-03-01']
covid_onset = df_clean[(df_clean['Date'] >= '2020-03-01') & (df_clean['Date'] <= '2020-06-30')]
post_covid = df_clean[df_clean['Date'] > '2020-06-30']

print(f"\nCOVID-19 Period Analysis:")
print(f"\n  Pre-COVID (Before March 2020):")
print(f"    Average Unemployment Rate: {pre_covid['Estimated Unemployment Rate (%)'].mean():.2f}%")
print(f"    Max Unemployment Rate: {pre_covid['Estimated Unemployment Rate (%)'].max():.2f}%")
print(f"    Min Unemployment Rate: {pre_covid['Estimated Unemployment Rate (%)'].min():.2f}%")

print(f"\n  COVID-19 Onset (March-June 2020):")
print(f"    Average Unemployment Rate: {covid_onset['Estimated Unemployment Rate (%)'].mean():.2f}%")
print(f"    Max Unemployment Rate: {covid_onset['Estimated Unemployment Rate (%)'].max():.2f}%")
print(f"    Min Unemployment Rate: {covid_onset['Estimated Unemployment Rate (%)'].min():.2f}%")
print(f"    Peak Increase: {covid_onset['Estimated Unemployment Rate (%)'].max() - pre_covid['Estimated Unemployment Rate (%)'].mean():.2f}%")

print(f"\n  Post-COVID Recovery (After June 2020):")
print(f"    Average Unemployment Rate: {post_covid['Estimated Unemployment Rate (%)'].mean():.2f}%")
print(f"    Max Unemployment Rate: {post_covid['Estimated Unemployment Rate (%)'].max():.2f}%")
print(f"    Min Unemployment Rate: {post_covid['Estimated Unemployment Rate (%)'].min():.2f}%")

# Find month with highest unemployment
max_unemployment_idx = df_clean['Estimated Unemployment Rate (%)'].idxmax()
max_unemployment_row = df_clean.loc[max_unemployment_idx]
print(f"\n  ⚠️  Peak Unemployment:")
print(f"    Date: {max_unemployment_row['Date'].date()}")
print(f"    Rate: {max_unemployment_row['Estimated Unemployment Rate (%)']:.2f}%")
print(f"    Region: {max_unemployment_row['Region']}")

# ==========================================
# 4. REGIONAL ANALYSIS
# ==========================================

print("\n" + "=" * 70)
print("[Step 4] Regional Analysis...")

# Average unemployment by region
regional_stats = df_clean.groupby('Region').agg({
    'Estimated Unemployment Rate (%)': ['mean', 'max', 'min'],
    'Estimated Employed': 'mean',
    'Estimated Labour Participation Rate (%)': 'mean'
}).round(2)

print(f"\n[Regional Unemployment Statistics]")
print(regional_stats)

# COVID-19 impact by region
regional_covid = df_clean.groupby('Region').apply(
    lambda x: x[x['Date'] >= '2020-03-01']['Estimated Unemployment Rate (%)'].mean() - 
              x[x['Date'] < '2020-03-01']['Estimated Unemployment Rate (%)'].mean()
).sort_values(ascending=False)

print(f"\n[COVID-19 Impact by Region (Unemployment Increase)]")
for region, impact in regional_covid.items():
    print(f"  {region}: +{impact:.2f}%")

# ==========================================
# 5. RURAL vs URBAN COMPARISON
# ==========================================

print("\n" + "=" * 70)
print("[Step 5] Rural vs Urban Analysis...")

rural_data = df_clean[df_clean['Area'] == 'Rural']
urban_data = df_clean[df_clean['Area'] == 'Urban']

print(f"\n[Rural Areas]")
print(f"  Average Unemployment Rate: {rural_data['Estimated Unemployment Rate (%)'].mean():.2f}%")
print(f"  Max Unemployment Rate: {rural_data['Estimated Unemployment Rate (%)'].max():.2f}%")
print(f"  Average Labour Participation: {rural_data['Estimated Labour Participation Rate (%)'].mean():.2f}%")

print(f"\n[Urban Areas]")
print(f"  Average Unemployment Rate: {urban_data['Estimated Unemployment Rate (%)'].mean():.2f}%")
print(f"  Max Unemployment Rate: {urban_data['Estimated Unemployment Rate (%)'].max():.2f}%")
print(f"  Average Labour Participation: {urban_data['Estimated Labour Participation Rate (%)'].mean():.2f}%")

print(f"\n[Comparison]")
difference = rural_data['Estimated Unemployment Rate (%)'].mean() - urban_data['Estimated Unemployment Rate (%)'].mean()
print(f"  Rural vs Urban Difference: {abs(difference):.2f}%")
if difference > 0:
    print(f"  → Rural areas have higher unemployment")
else:
    print(f"  → Urban areas have higher unemployment")

# ==========================================
# 6. VISUALIZATIONS
# ==========================================

print("\n" + "=" * 70)
print("[Step 6] Creating Visualizations...")

# 1. Unemployment Rate Over Time
fig, ax = plt.subplots(figsize=(14, 7))
for region in df_clean['Region'].unique():
    region_data = df_clean[df_clean['Region'] == region].sort_values('Date')
    ax.plot(region_data['Date'], region_data['Estimated Unemployment Rate (%)'], 
            marker='o', label=region, linewidth=2, markersize=4, alpha=0.7)

# Add COVID-19 shaded region
ax.axvspan(pd.to_datetime('2020-03-01'), pd.to_datetime('2020-06-30'), 
           alpha=0.2, color='red', label='COVID-19 Period')

ax.set_xlabel('Date', fontweight='bold', fontsize=12)
ax.set_ylabel('Unemployment Rate (%)', fontweight='bold', fontsize=12)
ax.set_title('Unemployment Rate Trends Across Indian States (with COVID-19 Impact)', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/01_Unemployment_Trends.png', dpi=300, bbox_inches='tight')
print("✓ Unemployment trends plot saved")
plt.close()

# 2. COVID-19 Impact Comparison
fig, ax = plt.subplots(figsize=(12, 7))
periods = ['Pre-COVID\n(Before Mar 2020)', 'COVID-19 Onset\n(Mar-Jun 2020)', 'Recovery\n(After Jun 2020)']
rates = [
    pre_covid['Estimated Unemployment Rate (%)'].mean(),
    covid_onset['Estimated Unemployment Rate (%)'].mean(),
    post_covid['Estimated Unemployment Rate (%)'].mean()
]
colors = ['#2ecc71', '#e74c3c', '#f39c12']
bars = ax.bar(periods, rates, color=colors, edgecolor='black', linewidth=2, width=0.6)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

ax.set_ylabel('Average Unemployment Rate (%)', fontweight='bold', fontsize=12)
ax.set_title('COVID-19 Impact on Unemployment - India', fontsize=14, fontweight='bold', pad=20)
ax.set_ylim([0, max(rates) * 1.15])
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/02_COVID19_Impact.png', dpi=300, bbox_inches='tight')
print("✓ COVID-19 impact comparison saved")
plt.close()

# 3. Regional Unemployment Heatmap
regional_pivot = df_clean.pivot_table(
    values='Estimated Unemployment Rate (%)',
    index='Region',
    columns=df_clean['Date'].dt.to_period('M'),
    aggfunc='mean'
)

fig, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(regional_pivot, cmap='RdYlGn_r', cbar_kws={'label': 'Unemployment Rate (%)'}, ax=ax)
ax.set_title('Regional Unemployment Rate Heatmap Over Time', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Month', fontweight='bold')
ax.set_ylabel('Region', fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/03_Regional_Heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Regional heatmap saved")
plt.close()

# 4. Rural vs Urban Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Unemployment comparison
area_data = df_clean.groupby('Area')['Estimated Unemployment Rate (%)'].agg(['mean', 'max', 'min'])
x_pos = np.arange(len(area_data))
width = 0.25

ax1.bar(x_pos - width, area_data['min'], width, label='Min', color='#3498db', alpha=0.8)
ax1.bar(x_pos, area_data['mean'], width, label='Mean', color='#2ecc71', alpha=0.8)
ax1.bar(x_pos + width, area_data['max'], width, label='Max', color='#e74c3c', alpha=0.8)

ax1.set_xlabel('Area Type', fontweight='bold')
ax1.set_ylabel('Unemployment Rate (%)', fontweight='bold')
ax1.set_title('Rural vs Urban Unemployment Rates', fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(area_data.index)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Labour participation comparison
labour_data = df_clean.groupby('Area')['Estimated Labour Participation Rate (%)'].mean()
bars = ax2.bar(labour_data.index, labour_data.values, color=['#9b59b6', '#e67e22'], 
               edgecolor='black', linewidth=2, width=0.5)

for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')

ax2.set_ylabel('Labour Participation Rate (%)', fontweight='bold')
ax2.set_title('Rural vs Urban Labour Participation', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/04_Rural_Urban_Comparison.png', dpi=300, bbox_inches='tight')
print("✓ Rural vs Urban comparison saved")
plt.close()

# 5. Top 5 Most Affected States During COVID-19
covid_data = df_clean[df_clean['Date'] >= '2020-03-01']
top_regions = covid_data.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False).head(5)

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(top_regions.index, top_regions.values, color='#e74c3c', edgecolor='black', linewidth=2)

for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2.,
            f'{width:.2f}%', ha='left', va='center', fontweight='bold', fontsize=11)

ax.set_xlabel('Average Unemployment Rate During COVID-19 (%)', fontweight='bold')
ax.set_title('Top 5 Most Affected States During COVID-19 Pandemic', fontsize=13, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/05_Top_Affected_States.png', dpi=300, bbox_inches='tight')
print("✓ Top affected states plot saved")
plt.close()

# 6. Employment Numbers Trend
fig, ax = plt.subplots(figsize=(14, 7))
employment_trend = df_clean.groupby('Date')['Estimated Employed'].sum().sort_index()

ax.plot(employment_trend.index, employment_trend.values, color='#2ecc71', linewidth=3, marker='o', markersize=5)
ax.fill_between(employment_trend.index, employment_trend.values, alpha=0.3, color='#2ecc71')

# Add COVID-19 shaded region
ax.axvspan(pd.to_datetime('2020-03-01'), pd.to_datetime('2020-06-30'), 
           alpha=0.2, color='red', label='COVID-19 Period')

ax.set_xlabel('Date', fontweight='bold', fontsize=12)
ax.set_ylabel('Total Estimated Employed', fontweight='bold', fontsize=12)
ax.set_title('Total Employment Trend Across India', fontsize=14, fontweight='bold', pad=20)
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/06_Employment_Trend.png', dpi=300, bbox_inches='tight')
print("✓ Employment trend plot saved")
plt.close()

# 7. Labour Participation Rate Trend
fig, ax = plt.subplots(figsize=(14, 7))
labour_trend = df_clean.groupby('Date')['Estimated Labour Participation Rate (%)'].mean().sort_index()

ax.plot(labour_trend.index, labour_trend.values, color='#3498db', linewidth=3, marker='s', markersize=5)
ax.fill_between(labour_trend.index, labour_trend.values, alpha=0.3, color='#3498db')

# Add COVID-19 shaded region
ax.axvspan(pd.to_datetime('2020-03-01'), pd.to_datetime('2020-06-30'), 
           alpha=0.2, color='red', label='COVID-19 Period')

ax.set_xlabel('Date', fontweight='bold', fontsize=12)
ax.set_ylabel('Labour Participation Rate (%)', fontweight='bold', fontsize=12)
ax.set_title('Labour Force Participation Rate Trend', fontsize=14, fontweight='bold', pad=20)
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/07_Labour_Participation_Trend.png', dpi=300, bbox_inches='tight')
print("✓ Labour participation trend saved")
plt.close()

# ==========================================
# 7. KEY FINDINGS AND INSIGHTS
# ==========================================

print("\n" + "=" * 70)
print("KEY FINDINGS AND INSIGHTS")
print("=" * 70)

print("\n📊 UNEMPLOYMENT TRENDS:")
print(f"  • Pre-COVID average: {pre_covid['Estimated Unemployment Rate (%)'].mean():.2f}%")
print(f"  • COVID-19 peak: {covid_onset['Estimated Unemployment Rate (%)'].max():.2f}%")
print(f"  • Recovery average: {post_covid['Estimated Unemployment Rate (%)'].mean():.2f}%")

print("\n⚠️  COVID-19 IMPACT:")
covid_impact = covid_onset['Estimated Unemployment Rate (%)'].mean() - pre_covid['Estimated Unemployment Rate (%)'].mean()
print(f"  • Average increase: +{covid_impact:.2f}%")
print(f"  • Peak increase: +{covid_onset['Estimated Unemployment Rate (%)'].max() - pre_covid['Estimated Unemployment Rate (%)'].mean():.2f}%")
print(f"  • Hardest hit region: {regional_covid.index[0]} ({regional_covid.iloc[0]:.2f}% increase)")

print("\n🗺️  REGIONAL INSIGHTS:")
print(f"  • Highest unemployment: {df_clean.groupby('Region')['Estimated Unemployment Rate (%)'].mean().idxmax()}")
print(f"  • Lowest unemployment: {df_clean.groupby('Region')['Estimated Unemployment Rate (%)'].mean().idxmin()}")
print(f"  • Most variable region: {df_clean.groupby('Region')['Estimated Unemployment Rate (%)'].std().idxmax()}")

print("\n🏙️  AREA COMPARISON:")
if rural_data['Estimated Unemployment Rate (%)'].mean() > urban_data['Estimated Unemployment Rate (%)'].mean():
    print(f"  • Rural unemployment: {rural_data['Estimated Unemployment Rate (%)'].mean():.2f}%")
    print(f"  • Urban unemployment: {urban_data['Estimated Unemployment Rate (%)'].mean():.2f}%")
    print(f"  • Rural areas affected more by {abs(difference):.2f}%")
else:
    print(f"  • Rural unemployment: {rural_data['Estimated Unemployment Rate (%)'].mean():.2f}%")
    print(f"  • Urban unemployment: {urban_data['Estimated Unemployment Rate (%)'].mean():.2f}%")
    print(f"  • Urban areas affected more by {abs(difference):.2f}%")

print("\n💼 EMPLOYMENT IMPACT:")
pre_covid_emp = pre_covid['Estimated Employed'].mean()
covid_emp = covid_onset['Estimated Employed'].mean()
emp_loss = ((pre_covid_emp - covid_emp) / pre_covid_emp) * 100
print(f"  • Pre-COVID employment: {pre_covid_emp:,.0f}")
print(f"  • COVID-19 employment: {covid_emp:,.0f}")
print(f"  • Job loss percentage: {emp_loss:.2f}%")

print("\n📈 RECOVERY ANALYSIS:")
recovery_rate = (post_covid['Estimated Employed'].mean() - covid_onset['Estimated Employed'].mean()) / (pre_covid['Estimated Employed'].mean() - covid_onset['Estimated Employed'].mean())
print(f"  • Recovery rate: {recovery_rate*100:.2f}% (compared to peak loss)")
if post_covid['Estimated Unemployment Rate (%)'].mean() < covid_onset['Estimated Unemployment Rate (%)'].mean():
    print(f"  • Status: Recovery in progress")
else:
    print(f"  • Status: Limited recovery observed")

# ==========================================
# 8. SUMMARY
# ==========================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print("\n✅ ALL VISUALIZATIONS SAVED:")
print("   - 01_Unemployment_Trends.png")
print("   - 02_COVID19_Impact.png")
print("   - 03_Regional_Heatmap.png")
print("   - 04_Rural_Urban_Comparison.png")
print("   - 05_Top_Affected_States.png")
print("   - 06_Employment_Trend.png")
print("   - 07_Labour_Participation_Trend.png")

print("\n📊 ANALYSIS SUMMARY:")
print(f"  • Total data points analyzed: {len(df_clean)}")
print(f"  • Time period: {df_clean['Date'].min().date()} to {df_clean['Date'].max().date()}")
print(f"  • States analyzed: {df_clean['Region'].nunique()}")
print(f"  • COVID-19 impact quantified")
print(f"  • Regional variations identified")
print(f"  • Recovery patterns analyzed")

print("\n" + "=" * 70)
print("Author: Pragyan Kumar Beheruk")
print("Repository: OIBSIP")
print("Task: Unemployment Analysis with COVID-19 Impact - Task 2")
print("=" * 70)
