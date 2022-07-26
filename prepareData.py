import tweepy  # https://github.com/tweepy/tweepy
from secret import consumer_secret, consumer_key
import datetime
import pandas as pd
import os

# Twitter API credentials

def parce_string_to_date (string_date):
    year, month, day = string_date.split('-')
    return datetime.date(int(year), int(month), int(day))


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


df = pd.read_csv('TSLA/TSLA.csv', encoding='cp1250')

datePriceDict = {}

startCandleDate = None
startPrice = None
endCandleDate = None

candlesList = []

x_data = []
y_data = []

# allTweets = download_tweets("elonmusk")
allTweets = []
all_tweet_files = os.listdir("tweets/")
for file_path in all_tweet_files:
    with open(f'tweets/{file_path}', 'r', encoding="utf-8") as f:
        tweetFile = f.read()
        tweet_id, tweet_date_str, tweet_text = tweetFile.split('////')
        tweet_date = parce_string_to_date(tweet_date_str.split(' ')[0])
        allTweets.append([tweet_id, tweet_date, tweet_text])

# differ_types
# 0 - equal
# 1 - pump
# 2 - dump
tweetIndex = 0

for rowIndex in range(len(df['Date'])):
    nextPrice = df['Adj Close'][rowIndex]

    if startCandleDate == None:
        print(df['Date'][rowIndex])
        startCandleDate = parce_string_to_date(df['Date'][rowIndex])
        startPrice = nextPrice
    else:
        endCandleDate = parce_string_to_date(df['Date'][rowIndex])

        # define differ type
        different_type = 0
        ratio = startPrice / nextPrice
        if ratio > 1.001:
            different_type = 1
        if ratio < 0.999:
            different_type = 2

        # For future calculate
        #
        # different = nextPrice - startPrice
        # candlesList.append(different)
        # startPrice = nextPrice

        # datePriceDict[df['Date'][rowIndex]] = df['Adj Close'][rowIndex]

        print(startCandleDate)
        print(startPrice)
        print(endCandleDate)
        print(nextPrice)
        startCandleDate = endCandleDate
        startPrice = nextPrice
        print(ratio)
        print(tweetIndex)
        while allTweets[tweetIndex][1] < endCandleDate:
            x_data.append(allTweets[tweetIndex][2])
            y_data.append(different_type)
            tweetIndex = tweetIndex + 1
        print(tweetIndex)
        print('--------')

# print(datePriceDict)

def write_tweets_to_file(tweets):
    print(len(tweets))
    for tweet in tweets:
        with open(f'tweets/{tweet[0]}.txt', 'w', encoding="utf-8") as f:
            f.write(f'{tweet[0]}////{tweet[1]}////{tweet[2]}')
    pass