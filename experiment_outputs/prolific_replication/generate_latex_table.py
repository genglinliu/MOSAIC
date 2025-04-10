import pandas as pd
import numpy as np

# Load the CSV data
df = pd.read_csv('experiment_outputs/prolific_replication/results/statistical_results_full.csv')

# Function to format significance symbols for LaTeX
def format_sig(sig):
    if pd.isna(sig):
        return ''
    elif sig == 'n/a':
        return 'n/a'
    elif sig == 'ns':
        return 'ns'
    else:
        # Convert * to \textsuperscript{*} format for LaTeX
        return '\\textsuperscript{' + sig + '}'

# Function to format p-values
def format_pval(p):
    if pd.isna(p):
        return ''
    else:
        return f"{p:.3f}"

# Format t-statistics and p-values with significance markers
df_formatted = df.copy()
for action in ['likes', 'shares', 'comments']:
    # Format t-stat with p-value and significance
    df_formatted[f'{action}_stat'] = df_formatted.apply(
        lambda row: f"{row[f'{action}_t']:.3f} {format_sig(row[f'{action}_sig'])}" 
        if not pd.isna(row[f'{action}_t']) else "—", 
        axis=1
    )
    
    # Format means for humans and agents
    df_formatted[f'{action}_means'] = df_formatted.apply(
        lambda row: f"{row[f'{action}_human_mean']:.2f} vs {row[f'{action}_agent_mean']:.2f}", 
        axis=1
    )

# Create the LaTeX table content
latex_content = """\\documentclass{article}
\\usepackage{booktabs}
\\usepackage{longtable}
\\usepackage{array}
\\usepackage{multirow}
\\usepackage{siunitx}
\\usepackage{pdflscape}

\\begin{document}

\\begin{landscape}
\\begin{longtable}{llr|ll|ll|ll}
\\toprule
\\textbf{Attribute} & \\textbf{Value} & \\textbf{n} & 
\\multicolumn{2}{c|}{\\textbf{Likes}} & 
\\multicolumn{2}{c|}{\\textbf{Shares}} & 
\\multicolumn{2}{c}{\\textbf{Comments}} \\\\
\\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9}
 &  &  & Human vs Agent & t-stat & Human vs Agent & t-stat & Human vs Agent & t-stat \\\\
\\midrule
\\endhead

\\midrule
\\multicolumn{9}{r}{\\textit{Continued on next page}} \\\\
\\endfoot

\\bottomrule
\\caption{Statistical comparison of human vs agent engagement patterns by demographic attributes.}
\\label{tab:engagement-by-demographic}
\\endlastfoot

"""

# Add each row to the table
current_attribute = None
for _, row in df_formatted.iterrows():
    # Check if this is a new attribute group
    if current_attribute != row['attribute']:
        current_attribute = row['attribute']
        # Add a small gap between attribute groups
        if _ > 0:
            latex_content += "\\midrule\n"
    
    # Format the row data
    latex_content += f"{row['attribute']} & {row['value']} & {row['n_humans']} & "
    latex_content += f"{row['likes_means']} & {row['likes_stat']} & "
    latex_content += f"{row['shares_means']} & {row['shares_stat']} & "
    latex_content += f"{row['comments_means']} & {row['comments_stat']} \\\\\n"

# Close the table
latex_content += "\\end{longtable}\n"
latex_content += "\\begin{tablenotes}\n"
latex_content += "\\small\n"
latex_content += "\\item[*] $p < 0.05$, ** $p < 0.01$, *** $p < 0.001$, ns: not significant\n"
latex_content += "\\end{tablenotes}\n"
latex_content += "\\end{landscape}\n"
latex_content += "\\end{document}"

# Save to tex file
output_file = 'experiment_outputs/prolific_replication/results/demographic_engagement_table.tex'
with open(output_file, 'w') as f:
    f.write(latex_content)

print(f"LaTeX table saved to {output_file}") 