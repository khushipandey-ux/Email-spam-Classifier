import pandas as pd

# Read raw text file
data = pd.read_csv("SMSSpamCollection", sep='\t', header=None, names=['label', 'message'])

# Save as CSV
data.to_csv("spam_dataset.csv", index=False)

print("✅ spam_dataset.csv created!")