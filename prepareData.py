import tweepy  # https://github.com/tweepy/tweepy
import datetime
import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import utils
import re
from sklearn.model_selection import train_test_split
from secret import consumer_secret, consumer_key
import gensim
import matplotlib.pyplot as plt


# Twitter API credentials

def parce_string_to_date(string_date):
    year, month, day = string_date.split('-')
    return datetime.date(int(year), int(month), int(day))


def write_tweets_to_file(tweets):
    print(len(tweets))
    for tweet in tweets:
        with open(f'tweets/{tweet[0]}.txt', 'w', encoding="utf-8") as f:
            f.write(f'{tweet[0]}////{tweet[1]}////{tweet[2]}')
    pass


def download_tweets(
        screen_name
):
    # Twitter only allows access to a users most recent 3240 tweets with this method

    # authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    # auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    # initialize a list to hold all the tweepy Tweets
    alltweets = []

    # make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name=screen_name, count=200)

    # save most recent tweets
    alltweets.extend(new_tweets)

    # save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    # keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        print(f"getting tweets before {oldest}")

        # all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name=screen_name, count=200, max_id=oldest)

        # save most recent tweets
        alltweets.extend(new_tweets)

        # update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        print(f"...{len(alltweets)} tweets downloaded so far")

    # transform the tweepy tweets into a 2D array that will populate the csv
    return [[tweet.id_str, tweet.created_at, tweet.text] for tweet in alltweets]


TSLADataFrame = pd.read_csv('TSLA/TSLA.csv', encoding='cp1250')

datePriceDict = {}

startCandleDate = None
startPrice = None
endCandleDate = None

candlesList = []

x_data = []
y_data = []

word2vecDF = pd.DataFrame()
# download tweets and write to file
# write_tweets_to_file(download_tweets("elonmusk"))

tweetsDataFrame = pd.read_csv('tweets/TweetsElonMusk.csv', encoding='utf-8')
print(tweetsDataFrame.keys())

tweetsDataFrame["created_at"] = pd.to_datetime(tweetsDataFrame["created_at"])

# verify datatype
print(type(tweetsDataFrame.created_at[0]))

allTweets = []

tweetsDataFrame = tweetsDataFrame.sort_values(by=["created_at"], ignore_index=True)

for rowIndex in range(len(tweetsDataFrame['created_at'])):
    allTweets.append([
        tweetsDataFrame['id'][rowIndex],
        parce_string_to_date(tweetsDataFrame['date'][rowIndex]),
        re.sub(r"@[A-z]+|_", "nickname", tweetsDataFrame['tweet'][rowIndex]),
        tweetsDataFrame["likes_count"][rowIndex],
        tweetsDataFrame["retweets_count"][[rowIndex]]
    ])
print(len(allTweets))

all_tweet_files = os.listdir("tweets/")
# for file_path in all_tweet_files:
#     with open(f'tweets/{file_path}', 'r', encoding="utf-8") as f:
#         tweetFile = f.read()
#         tweet_id, tweet_date_str, tweet_text = tweetFile.split('////')
#         tweet_date = parce_string_to_date(tweet_date_str.split(' ')[0])
#         allTweets.append([tweet_id, tweet_date, tweet_text])

# differ_types
# 0 - equal
# 1 - dump
# 2 - pump

tweetIndex = 0

pump_count = 0
dump_count = 0
neutral_count = 0

for rowIndex in range(len(TSLADataFrame['Date'])):
    nextPrice = TSLADataFrame['Adj Close'][rowIndex]
    if startCandleDate is None:
        startCandleDate = parce_string_to_date(TSLADataFrame['Date'][rowIndex])
        startPrice = nextPrice
    else:
        endCandleDate = parce_string_to_date(TSLADataFrame['Date'][rowIndex])

        # define differ type
        different_type = 0
        ratio = startPrice / nextPrice
        if ratio > 1.001:
            different_type = 2
        if ratio < 0.999:
            different_type = 1
        # For future calculate
        #
        # different = nextPrice - startPrice
        # candlesList.append(different)
        # startPrice = nextPrice

        # datePriceDict[df['Date'][rowIndex]] = df['Adj Close'][rowIndex]

        # print(startCandleDate)
        # print(startPrice)
        # print(endCandleDate)
        # print(nextPrice)

        try:
            tweetsLikesCount = 0
            tweetsRetweetsCount = 0
            periodTweets = []
            periodValues = []

            while allTweets[tweetIndex][1] < endCandleDate:
                if allTweets[tweetIndex][1] >= startCandleDate:
                    tweetsLikesCount += allTweets[tweetIndex][3]
                    tweetsRetweetsCount += allTweets[tweetIndex][4]
                    periodTweets.append(allTweets[tweetIndex])
                    periodValues.append(different_type)
                else:
                    print(f'{allTweets[tweetIndex][1]} - tweet date')
                    print(f'{endCandleDate} - end date')
                tweetIndex = tweetIndex + 1

            if len(periodTweets) > 0:
                tweetsLikesCountMinimum = tweetsLikesCount / len(periodTweets) / 2

                for periodTweetIndex in range(len(periodTweets)):
                    if periodTweets[periodTweetIndex][3] > tweetsLikesCountMinimum:
                        word2vecDF = word2vecDF.append({'text': periodTweets[periodTweetIndex][2], 'type': periodValues[periodTweetIndex]}, ignore_index=True)
                        x_data.append(periodTweets[periodTweetIndex][2])
                        y_data.append(periodValues[periodTweetIndex])
                        if periodValues[periodTweetIndex] == 1:
                            dump_count += 1
                        if periodValues[periodTweetIndex] == 2:
                            pump_count += 1
                        if periodValues[periodTweetIndex] == 0:
                            neutral_count += 1

            startCandleDate = endCandleDate
            startPrice = nextPrice
        except IndexError:
            print(f'tweet with {tweetIndex} index not added')

# print(datePriceDict)
print(len(x_data))
print(len(y_data))


def make_tokenizer(
        VOCAB_SIZE,
        txt_train
):
    tokenizer = Tokenizer(num_words=VOCAB_SIZE,
                          filters='!"#$%&()*+,-–—./…:;<=>?@[\\]^_`{|}~«»\t\n\xa0\ufeff',
                          lower=True,
                          split=' ',
                          oov_token='unrecognized-word',
                          char_level=False)  # prohibit tokenize every symbol

    tokenizer.fit_on_texts(txt_train)
    return tokenizer


def make_train_test(
        tokenizer,
        txt_train,
        txt_test=None
):
    # transform word to tokens
    seq_train = tokenizer.texts_to_sequences(txt_train)

    if txt_test:
        # transform word to tokens
        seq_test = tokenizer.texts_to_sequences(txt_test)
    else:
        seq_test = None

    return seq_train, seq_test


def split_sequence(
        sequence,
        win_size,
        hop
):
    return [sequence[i:i + win_size] for i in range(0, len(sequence) - win_size + 1, hop)]


def vectorize_sequence(
        seq_list,
        y_list,
        win_size,
        hop
):
    tweets_count = len(seq_list)
    x, y = [], []

    for tweet_index in range(tweets_count):
        vectors = split_sequence(seq_list[tweet_index], win_size, hop)
        x += vectors
        y += [utils.to_categorical(y_list[tweet_index], 3)] * len(vectors)

    return np.array(x), np.array(y)


def make_train_test(
        tokenizer,
        txt_train,
        txt_test=None,
        txt_val=None
):
    # transform word to tokens
    seq_train = tokenizer.texts_to_sequences(txt_train)

    if txt_test:
        # transform word to tokens
        seq_test = tokenizer.texts_to_sequences(txt_test)
    else:
        seq_test = None
    if txt_val:
        # transform word to tokens
        seq_val = tokenizer.texts_to_sequences(txt_test)
    else:
        seq_val = None

    return seq_train, seq_test, seq_val

diagramX = [1, 2, 3]
tick_label = ['dump', 'pump', 'neutral']
diagramY = [dump_count, pump_count, neutral_count]

plt.bar(diagramX, diagramY, tick_label=tick_label,
        width=0.8, color=['red', 'green'])
plt.title('tweets data!')
plt.show()
VOCAB_SIZE = 20000
WIN_SIZE = 5
WIN_HOP = 1
train_text_size = int(len(x_data) * 0.7)
test_text_size = int(len(x_data) * 0.05)
val_text_size = int(len(x_data) * 0.25)

text_train = x_data[:train_text_size]
classes_train = y_data[:train_text_size]
text_test = x_data[train_text_size:train_text_size + test_text_size]
classes_test = y_data[train_text_size:train_text_size + test_text_size]
text_val = x_data[train_text_size + test_text_size:]
classes_val = y_data[train_text_size + test_text_size:]

tok = make_tokenizer(VOCAB_SIZE, text_train)
seq_train, seq_test, seq_val = make_train_test(tok, text_train, text_test, text_val)

print("Фрагмент обучающего текста:")
print("В виде оригинального текста:              ", text_train[100][:101])
print("Он же в виде последовательности индексов: ", seq_train[100][:20])

x_train, y_train = vectorize_sequence(seq_train, classes_train, WIN_SIZE, WIN_HOP)
x_test, y_test = vectorize_sequence(seq_test, classes_test, WIN_SIZE, WIN_HOP)
x_val, y_val = vectorize_sequence(seq_val, classes_val, WIN_SIZE, WIN_HOP)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

word2vecDF['text_clean'] = word2vecDF['text'].apply(lambda text: gensim.utils.simple_preprocess(text))
word2vecDF['text_clean'] = word2vecDF['text'].apply(lambda text: gensim.utils.simple_preprocess(text))
w2X_train, w2X_test, w2y_train, w2y_test = train_test_split(word2vecDF['text_clean'], word2vecDF['type'], test_size=0.2)
