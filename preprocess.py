import pandas as pd
from sklearn.model_selection import train_test_split

# Read the csv file, remove quotation marks in Sentence column
df = pd.read_csv('./data/Game_of_Thrones_Script.csv', encoding='utf-8')
df['Sentence'] = df['Sentence'].str.replace('"', '')
print(df.head(10))

# Get only the Sentence column where Name column is 'jon snow', reset the index
# remove quotation marks from the sentences
jon_snow = df[df['Name'] == 'jon snow']['Sentence'].reset_index(drop=True)
print(jon_snow.head(10))

        

# Get only the Sentence column where its next row's Name is 'jon snow', reset the index
others = df[df['Name'].shift(-1) == 'jon snow']['Sentence'].reset_index(drop=True)
print(others.head(10))

# Create a new dataframe by concatenating rows of jon_snow and others
# ignore_index=True to reset index
# make new columns 'input' and 'response'
conversations = pd.concat([others, jon_snow], ignore_index=True, axis=1)
conversations.columns = ['input', 'response']
print(conversations.head(10))

# save conversations to csv
conversations.to_csv('./data/cleaned/conversations.csv', index=False)

# split conversations into train and test by random sampling
# save train and test to csv
train, test = train_test_split(conversations, test_size=0.2, random_state=42)
train.to_csv('./data/cleaned/train.csv', index=False)
test.to_csv('./data/cleaned/test.csv', index=False)

