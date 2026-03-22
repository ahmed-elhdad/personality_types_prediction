import pandas as pd
import re

df = pd.read_csv("../data/personality_types.csv")


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Remove URLs (http, https, www)
    text = re.sub(
        r"\b(com|org|net|edu)\b", " ", text
    )  # Remove common domain extentions
    text = re.sub(
        r"\b(jpg|png|gif|webp|svg|heic)\b", " ", text
    )  # Remove domain extensions and file extensions
    text = re.sub(r"\byoutube\b", "", text)  # Remove platform names
    text = re.sub(r"\d+", "", text)  # Remove digits
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation and special characters
    text = text.replace("_", "")  # Remove underscores (kept by \w)
    text = text.lower()  # Lowercase and strip extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    segments = text.split("|||")  # Split on delimiter
    return segments


# Feature Engineering
df["words_count"] = df["posts"].apply(lambda x: len(str(x).split()))
df["cleaned_posts"] = df["posts"].apply(clean_text)
df["cleaned_posts"] = df["cleaned_posts"].apply(lambda x: " ".join(x))
df["text_length"] = df["cleaned_posts"].apply(len)
df.drop(columns=["posts"], inplace=True)
df.to_csv("../data/cleaned_data.csv", index=False)
print("Successfully cleaned data and saved to '../data/cleaned_data.csv'")
# FINISHED
