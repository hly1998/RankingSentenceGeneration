# Merge the generated data, store the results in a list format, and remove duplicate entries.
import pandas as pd

# list the CSV file names to be merged
file_names = ['../data/ranking_sentences_1.csv']

combined_df = pd.DataFrame()

for file_name in file_names:
    df = pd.read_csv(file_name)
    combined_df = pd.concat([combined_df, df], ignore_index=True)

sents = combined_df["sent"].tolist()
rounds = combined_df["round"].tolist()

# set the generation step
rank_n = 32
rank_list = []
list_sent = []
for sent, r in zip(sents, rounds):
    if r == rank_n:
        list_sent.append(sent)
        if list_sent.count(list_sent[-1]) < 3 and list_sent.count(list_sent[-2]) < 3:
            rank_list.append(list_sent)
        list_sent = []
    elif r < rank_n:
        list_sent.append(sent)
    else:
        pass
df = pd.DataFrame({
    'list_sent': rank_list
})
df.to_csv("../data/rankingsentences_n:{}.csv".format(rank_n), index=False)
