import pandas as pd
DATE_FORMAT = "%Y-%m-%d"
DATE_COL = "date"
full_df = pd.read_csv("../data/full_df.csv")
full_df[DATE_COL] = pd.to_datetime(full_df[DATE_COL], format=DATE_FORMAT, errors="raise")
full_df = full_df.sort_values(DATE_COL).reset_index(drop=True)
print(full_df.head(3))


WINDOW_SIZE = 1095
EVAL_START_DATE = pd.to_datetime("2024-06-29")

eval_start_idx = int(full_df.index[full_df["date"] >= EVAL_START_DATE][0])
start_t = max(WINDOW_SIZE, eval_start_idx)

print("Eval start date target:", EVAL_START_DATE)
print("Computed start_t index:", start_t)
print("First forecast date implied by start_t:", full_df.loc[start_t, "date"])