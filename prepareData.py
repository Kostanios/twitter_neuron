import tweepy  # https://github.com/tweepy/tweepy
from secret import consumer_secret, consumer_key
import datetime
import pandas as pd


# Twitter API credentials

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

for rowIndex in range(len(df['Date'])):
    y, m, d = df['Date'][rowIndex].split('-')
    datePriceDict[df['Date'][rowIndex]] = df['Adj Close'][rowIndex]

print(datePriceDict)

download_tweets("elonmusk")


def write_tweets_to_file(tweets):
    print(len(tweets))
    for tweet in tweets:
        print(tweet)
        with open(f'tweets/{tweet[0]}.txt', 'w', encoding="utf-8") as f:
            f.write(f'{tweet[0]}////{tweet[1]}////{tweet[2]}')
    pass
