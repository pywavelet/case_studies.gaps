import glob
import os
import re

# Get all files in the directory
files = glob.glob('*.png')


# remove "N=5000_" from the filenames, replace " " with "_", remove "=", "(" and ")" from the filenames
for file in files:
    new_name = file.replace("N=5000_", "").replace(" ", "_").replace("=", "").replace("(", "").replace(")", "").replace("_domain", "")
    print(f"Renaming {file} to {new_name}")
    os.rename(file, new_name)


files = glob.glob('*with_gap_data.png')

# Sort the files by the label ("Cornish", "TDI1", "TDI2") and then by the Nf value
def extract_sort_key(filename):
    label_match = re.findall(r'^(Cornish|TDI1|TDI2)', filename)
    nf_match = re.findall(r'Nf(\d+)', filename)
    label = label_match[0] if label_match else ''
    nf_value = int(nf_match[0]) if nf_match else float('inf')
    return (label, nf_value)


files = sorted(files, key=extract_sort_key)

table_rows = []
# Process each file
for file in files:

    # Extract label from the filename
    label = file.split('_with_gap_data.png')[0]

    gap_data = file
    gap_covar = file.replace("data", "covar")
    no_gap_data = f"{label}_data.png"
    no_gap_covar = no_gap_data.replace("data", "covar")

    # Add rows to the markdown table
    table_rows.append(f"| {label} | ![data]({no_gap_data}) | ![covariance]({no_gap_covar}) |")
    table_rows.append(f"| {label} [GAP] | ![data]({gap_data}) | ![covariance]({gap_covar}) |")

# Create the markdown table
table_header = "| Label | Data | Covariance |\n|-------|------|------------|"
table_content = "\n".join(table_rows)
markdown_table = f"{table_header}\n{table_content}"


# Save the markdown table to a file
with open('summary.md', 'w') as f:
    markdown_table = "# Noise Correlation\n" + markdown_table
    f.write(markdown_table)

print("Markdown summary created successfully.")

