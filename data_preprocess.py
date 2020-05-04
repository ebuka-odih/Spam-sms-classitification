
import pandas as pd

labels = []
data = []
fd = open("SMSSpamCollection", "r")
for line in fd:
    if line[0] == 'h':
        pos = 3
    else:
        pos = 4

    labels.append(line[:pos])
    data.append(line[pos+1:])


dataframe = {'labels': labels,
             'text': data}


df = pd.DataFrame(dataframe, columns=['labels', 'text'])
df.to_csv(r'smspamcollection.csv', index=None, header=True)

print(data[1:10])
