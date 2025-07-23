#------------------------------ PREPROCESSING FUNCTION ------------------
import re
import unicodedata
from bs4 import BeautifulSoup
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
from symspellpy import SymSpell, Verbosity
import contractions
import spacy
import pandas as pd
import numpy as np
import emoji

# Initialize SymSpell for spelling correction
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary('frequency_dictionary_en_82_765.txt', term_index=0, count_index=1)

# Load SpaCy's English NLP model
nlp = spacy.load("en_core_web_sm")


# store contractions set as a variable
contractions_words = contractions.contractions_dict
# print(contractions_words)
updated_contraction_words = {
    "I'm": 'I am', "I'm'a": 'I am about to', "I'm'o": 'I am going to', "I've": 'I have', "I'll": 'I will',
    "I'll've": 'I will have', "I'd": 'I would', "I'd've": 'I would have', 'Whatcha': 'What are you', "amn't": 'am not', "ain't": 'are not',
    "aren't": 'are not', "'cause": 'because', "can't": 'cannot', "can't've": 'cannot have', "could've": 'could have', "couldn't": 'could not',
    "couldn't've": 'could not have', "daren't": 'dare not', "daresn't": 'dare not', "dasn't": 'dare not', "didn't": 'did not', 'didn’t': 'did not',
    "don't": 'do not', 'don’t': 'do not', "doesn't": 'does not', "e'er": 'ever', "everyone's": 'everyone is', 'finna': 'fixing to',
    'gimme': 'give me', "gon't": 'go not', 'gonna': 'going to', 'gotta': 'got to',  "hadn't": 'had not', "hadn't've": 'had not have',
    "hasn't": 'has not', "haven't": 'have not', "he've": 'he have', "he's": 'he is', "he'll": 'he will', "he'll've": 'he will have',
    "he'd": 'he would', "he'd've": 'he would have', "here's": 'here is', "how're": 'how are', "how'd": 'how did', "how'd'y": 'how do you',
    "how's": 'how is', "how'll": 'how will', "isn't": 'is not', "it's": 'it is', "'tis": 'it is', "'twas": 'it was', "it'll": 'it will',
    "it'll've": 'it will have', "it'd": 'it would', "it'd've": 'it would have', 'kinda': 'kind of', "let's": 'let us', 'luv': 'love',
    "ma'am": 'madam', "may've": 'may have', "mayn't": 'may not', "might've": 'might have', "mightn't": 'might not', "mightn't've": 'might not have',
    "must've": 'must have', "mustn't": 'must not', "mustn't've": 'must not have', "needn't": 'need not', "needn't've": 'need not have',
    "ne'er": 'never', "o'": 'of', "o'clock": 'of the clock', "ol'": 'old', "oughtn't": 'ought not', "oughtn't've": 'ought not have',
    "o'er": 'over', "shan't": 'shall not', "sha'n't": 'shall not', "shalln't": 'shall not', "shan't've": 'shall not have', "she's": 'she is',
    "she'll": 'she will', "she'd": 'she would', "she'd've": 'she would have', "should've": 'should have', "shouldn't": 'should not',
    "shouldn't've": 'should not have', "so've": 'so have', "so's": 'so is', "somebody's": 'somebody is', "someone's": 'someone is',
    "something's": 'something is', 'sux': 'sucks', "that're": 'that are', "that's": 'that is', "that'll": 'that will', "that'd": 'that would',
    "that'd've": 'that would have', "'em": 'them', "there're": 'there are', "there's": 'there is', "there'll": 'there will', "there'd": 'there would',
    "there'd've": 'there would have', "these're": 'these are', "they're": 'they are', "they've": 'they have', "they'll": 'they will', "they'll've": 'they will have',
    "they'd": 'they would', "they'd've": 'they would have', "this's": 'this is', "this'll": 'this will', "this'd": 'this would', "those're": 'those are',
    "to've": 'to have', 'wanna': 'want to', "wasn't": 'was not', "we're": 'we are', "we've": 'we have', "we'll": 'we will', "we'll've": 'we will have',
    "we'd": 'we would', "we'd've": 'we would have', "weren't": 'were not', "what're": 'what are', "what'd": 'what did', "what've": 'what have',
    "what's": 'what is', "what'll": 'what will', "what'll've": 'what will have', "when've": 'when have', "when's": 'when is',"where're": 'where are',
    "where'd": 'where did', "where've": 'where have', "where's": 'where is', "which's": 'which is', "who're": 'who are', "who've": 'who have',
    "who's": 'who is', "who'll": 'who will', "who'll've": 'who will have', "who'd": 'who would', "who'd've": 'who would have', "why're": 'why are',
    "why'd": 'why did', "why've": 'why have', "why's": 'why is', "will've": 'will have', "won't": 'will not', "won't've": 'will not have',
    "would've": 'would have', "wouldn't": 'would not', "wouldn't've": 'would not have', "y'all": 'you all', "y'all're": 'you all are',
    "y'all've": 'you all have', "y'all'd": 'you all would', "y'all'd've": 'you all would have', "you're": 'you are', "you've": 'you have',
    "you'll've": 'you shall have', "you'll": 'you will', "you'd": 'you would', "you'd've": 'you would have', 'to cause': 'to cause',
    'will cause': 'will cause', 'should cause': 'should cause', 'would cause': 'would cause', 'can cause': 'can cause', 'could cause': 'could cause',
    'must cause': 'must cause', 'might cause': 'might cause', 'shall cause': 'shall cause', 'may cause': 'may cause', 'jan.': 'january',
    'feb.': 'february', 'mar.': 'march', 'apr.': 'april', 'jun.': 'june', 'jul.': 'july', 'aug.': 'august', 'sep.': 'september', 'oct.': 'october',
    'nov.': 'november', 'dec.': 'december', 'I’m': 'I am', "I’m’a": 'I am about to', "I’m’o": 'I am going to', "I’ve": 'I have', 'I’ll': 'I will',
    "I’ll’ve": 'I will have', "I’d": 'I would', "I’d’ve": 'I would have', "amn’t": 'am not', "ain’t": 'are not', "aren’t": 'are not', "’cause": 'because',
    "can’t": 'cannot', "can’t’ve": 'cannot have', "could’ve": 'could have', "couldn’t": 'could not', "couldn’t’ve": 'could not have', "daren’t": 'dare not',
    "daresn’t": 'dare not', "dasn’t": 'dare not', "doesn’t": 'does not', "e’er": 'ever', 'everyone’s': 'everyone is', "gon’t": 'go not',
    "hadn’t": 'had not', "hadn’t’ve": 'had not have', "hasn’t": 'has not', "haven’t": 'have not', "he’ve": 'he have', "he’s": 'he is',
    "he’ll": 'he will', "he’ll’ve": 'he will have', "he’d": 'he would', "he’d’ve": 'he would have', "here’s": 'here is', "how’re": 'how are',
    "how’d": 'how did', "how’d’y": 'how do you', "how’s": 'how is', "how’ll": 'how will', 'isn’t': 'is not', "it’s": 'it is', "’tis": 'it is',
    "’twas": 'it was', "it’ll": 'it will', "it’ll’ve": 'it will have', "it’d": 'it would', "it’d’ve": 'it would have', "let’s": 'let us', "ma’am": 'madam',
    "may’ve": 'may have', "mayn’t": 'may not', "might’ve": 'might have', "mightn’t": 'might not', "mightn’t’ve": 'might not have', "must’ve": 'must have',
    "mustn’t": 'must not', "mustn’t’ve": 'must not have', "needn’t": 'need not', "needn’t’ve": 'need not have', "ne’er": 'never', "o’": 'of',
    "o’clock": 'of the clock', "ol’": 'old', "oughtn’t": 'ought not', "oughtn’t’ve": 'ought not have', "o’er": 'over', "shan’t": 'shall not',
    "sha’n’t": 'shall not', "shalln’t": 'shall not', "shan’t’ve": 'shall not have', "she’s": 'she is', "she’ll": 'she will', "she’d": 'she would',
    "she’d’ve": 'she would have', "should’ve": 'should have', "shouldn’t": 'should not', "shouldn’t’ve": 'should not have', "so’ve": 'so have',
    "so’s": 'so is', "somebody’s": 'somebody is', "someone’s": 'someone is', "something’s": 'something is', "that’re": 'that are', "that’s": 'that is',
    "that’ll": 'that will', "that’d": 'that would', "that’d’ve": 'that would have', "’em": 'them', "there’re": 'there are', "there’s": 'there is',
    "there’ll": 'there will', "there’d": 'there would', "there’d’ve": 'there would have', "these’re": 'these are', "they’re": 'they are', "they’ve": 'they have',
    "they’ll": 'they will', "they’ll’ve": 'they will have', "they’d": 'they would', "they’d’ve": 'they would have', "this’s": 'this is', "this’ll": 'this will',
    "this’d": 'this would', "those’re": 'those are', "to’ve": 'to have', "wasn’t": 'was not', "we’re": 'we are', 'we’ve': 'we have', "we’ll": 'we will',
    "we’ll’ve": 'we will have', "we’d": 'we would', "we’d’ve": 'we would have', "weren’t": 'were not', "what’re": 'what are', "what’d": 'what did',
    "what’ve": 'what have', "what’s": 'what is', "what’ll": 'what will', "what’ll’ve": 'what will have', "when’ve": 'when have', 'when’s': 'when is',
    "where’re": 'where are', "where’d": 'where did', "where’ve": 'where have', "where’s": 'where is', "which’s": 'which is', "who’re": 'who are',
    "who’ve": 'who have', "who’s": 'who is', "who’ll": 'who will', "who’ll’ve": 'who will have', "who’d": 'who would', "who’d’ve": 'who would have',
    "why’re": 'why are', "why’d": 'why did', "why’ve": 'why have', "why’s": 'why is', "will’ve": 'will have', "won’t": 'will not', "won’t’ve": 'will not have',
    "would’ve": 'would have', "wouldn’t": 'would not', "wouldn’t’ve": 'would not have', "y’all": 'you all', "y’all’re": 'you all are',
    "y’all’ve": 'you all have', "y’all’d": 'you all would', "y’all’d’ve": 'you all would have', "you’re": 'you are', "you’ve": 'you have',
    "you’ll’ve": 'you shall have', "you’ll": "you will", "you’d": 'you would', "you’d’ve": 'you would have', "i'm": 'i am', "i've": 'i have', 'u': 'you',
    '2nite': 'tonight', 'wil': 'will', 'bday': 'birthday', 'hai': 'hi', 'bihday': 'birthday', 'ur': 'your'}

# Create the tweet preprocessor
def preprocess_tweet(tweet):
    """Clean and preprocess a single tweet while retaining emojis for sentiment analysis."""

    # Fix mojibake (misencoded emojis)
    try:
        tweet = tweet.encode('latin1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    # Expand contractions
    for key, value in updated_contraction_words.items():
        tweet = re.sub(r'\b{}\b'.format(re.escape(key)), value, tweet)

    # Define regex patterns
    url_pattern = r'http[s]?://\S+|www\S+|\S+\.com|\S+@\S+'  # URLs and email addresses
    retweet_pattern = r'\bRT\b|@\S+'  # Retweets and user mentions
    numeric_pattern = r'\b\d+\b'  # Purely numeric entries

    # Remove URLs and email addresses
    tweet = re.sub(url_pattern, '', tweet)
    # Remove retweets and user mentions
    tweet = re.sub(retweet_pattern, '', tweet)
    # Remove accented characters using Normalize Unicode
    tweet = unicodedata.normalize('NFKD', tweet)
    # Remove HTML tags
    tweet = BeautifulSoup(tweet, "html.parser").get_text()

    # Remove undesired special characters but KEEP emojis
    emoji_pattern = (
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002700-\U000027BF"  # Dingbats
        u"\U000024C2-\U0001F251"  # Enclosed characters
    )
    tweet = re.sub(
        rf"[^\w\s{emoji_pattern}]",
        '',
        tweet
    )

    # Remove purely numeric entries
    tweet = re.sub(numeric_pattern, '', tweet)
    # Convert to lowercase
    tweet = tweet.lower()

    # Correct spelling
    words = tweet.split()
    corrected_words = []
    for word in words:
        # Get suggestions and choose the best one
        if emoji.is_emoji(word) or word.strip() == '':
            corrected_words.append(word)
            continue
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        corrected_word = suggestions[0].term if suggestions else word
        corrected_words.append(corrected_word)

    # Concatenate corrected words
    tweet = ' '.join(corrected_words)

    # Lemmatization using SpaCy
    doc = nlp(tweet)
    lemmatized_words = [
        token.lemma_ if not emoji.is_emoji(token.text) else token.text
        for token in doc
        if not token.is_stop and (token.is_alpha or emoji.is_emoji(token.text))
    ]

    # Remove additional whitespace
    cleaned_tweet = ' '.join(lemmatized_words).strip()

    return cleaned_tweet  # Return the cleaned tweet