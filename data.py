from pytrends.request import TrendReq
import pandas as pd
import time


keywords = [
    "depression", "anxiety", "panic attacks", "burnout",
    "brain fog", "no motivation to do anything", "crying for no reason",
    "how to know if I'm depressed", "free therapy near me",
    "job stress", "feeling alone in college", "toxic parents",
    "I hate my life quotes", "how to disappear", "songs to cry to",
  "always tired", "can't stop overthinking", "do I need therapy"]

pytrends = TrendReq(hl='en-US', tz=360)

def fetch_trends_in_batches(keywords, timeframe='today 5-y', geo='US'):
    all_data = pd.DataFrame()

    for i in range(0, len(keywords), 5):
        batch = keywords[i:i+5]
        try:
            pytrends.build_payload(batch, cat=0, timeframe=timeframe, geo=geo)
            data = pytrends.interest_over_time()

            if data.empty:
                # Add empty columns if no data is returned
                data = pd.DataFrame(index=all_data.index if not all_data.empty else pd.date_range(end=pd.Timestamp.today(), periods=261, freq='W'), columns=batch)

            else:
                data = data.drop(columns='isPartial', errors='ignore')

            # Merge with full dataset
            if all_data.empty:
                all_data = data
            else:
                all_data = all_data.join(data, how='outer')

        except Exception as e:
            print(f"Error fetching {batch}: {e}")
            # Add NaN columns for failed batch
            missing_df = pd.DataFrame(index=all_data.index if not all_data.empty else pd.date_range(end=pd.Timestamp.today(), periods=261, freq='W'), columns=batch)
            all_data = all_data.join(missing_df, how='outer') if not all_data.empty else missing_df

        time.sleep(15)  # Avoid getting rate-limited

    return all_data


###Fetch the data

search_data = fetch_trends_in_batches(keywords)

###Save to CSV

search_data.to_csv("mental_health_trends.csv")

print("Data saved to 'mental_health_trends.csv'")