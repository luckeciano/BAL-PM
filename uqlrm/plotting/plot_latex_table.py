import pandas as pd

def txt_to_latex(file1, file2, N):
    # Read the text files line by line
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()[:320]
        lines2 = f2.readlines()[:320]

    # Create a new dataframe for the latex table
    latex_df = pd.DataFrame()

    # Add the columns to the new dataframe
    # latex_df['Row Number'] = range(1, min(len(lines1), len(lines2)) + 1)
    latex_df['First File'] = [line[:N] + '...' if len(line) > N else line for line in lines2]


    # Convert the dataframe to a latex table
    latex_table = latex_df.to_latex(index=False)


    # Write the latex table to a file
    with open('latex_table.tex', 'w') as f:
        f.write(latex_table)

# Example usage:
txt_to_latex('/users/lucelo/UQLRM/batch_prompts_BALD_0.txt', '/users/lucelo/UQLRM/batch_prompts_BAE-PM (ours)_0.txt', 80)
