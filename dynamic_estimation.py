import pandas as pd
from datetime import datetime

def dynamic_cost_estimation(df, team_size_col, duration_col, daily_hours_col):
    """
    Calculate dynamic effort for each project in the dataset.
    """
    df['Dynamic Effort'] = df.apply(lambda row: row[duration_col] * (row[team_size_col] * row[daily_hours_col] * 22), axis=1)

    # Save the updated dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"dynamic_cost_estimation_{timestamp}.xlsx"
    df.to_excel(output_file, index=False)
    print(f"Dynamic cost estimation saved as {output_file}")

    return df