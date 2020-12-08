import pandas as pd
import numpy as np

result = pd.read_csv('output.csv', names=['user', 'item', 'score', 'test_item', 'test_score', 'rank'])

print(result[result["rank"] == 1.0])
print(result[result["rank"] == 24.0])
print(result[result['user'] == 8])

sum = 0
count = 0
for row in result.itertuples():
    sum += 1 / row.rank
    count += 1
    #print(1 / row.rank)

print(sum / count)
