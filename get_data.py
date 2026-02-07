import urllib.request
import zipfile
import io
import pandas as pd
import os

print("Contacting Server...")
# This is the official standard dataset (5,574 rows) that matches our code
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

try:
    response = urllib.request.urlopen(url)
    with zipfile.ZipFile(io.BytesIO(response.read())) as z:
        print("Extracting")
        with z.open('SMSSpamCollection') as f:
            # We force the column names to be v1 and v2 so the AI doesn't crash
            df = pd.read_csv(f, sep='\t', header=None, names=['v1', 'v2'])
            df.to_csv('spam.csv', index=False)

    print(f"SUCCESS! 'spam.csv' is ready in: {os.getcwd()}")
    print("You can now proceed to Phase 2.")

except Exception as e:
    print(f"ERROR: {e}")