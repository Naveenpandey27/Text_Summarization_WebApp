# Import necessary libraries
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

# Set the input text
text = """Several thousand years ago in north-central India, two people sat in a chariot in the midpoint of a great battlefield. 
        One of them, the yogi Arjuna, knew that it would not be long before the conflict would begin. 
        So he asked Krishna, the Master of Yoga, what should be his attitude and perspective in this moment. 
        And above all: What should he do? There was no time to spare in empty words. 
        In a brief discourse, later turned into seven-hundred Sanskrit verses by the sage Vyasa, Krishna outlined to Arjuna the way to live an entire 
        life so as to gain perfect self-knowledge and self-mastery: The Bhagavad Gita.  
        The Bhagavad Gita tells us that we can attain a Knowing beyond even what it tells us. 
        And it shows us the way. 
        In The Bhagavad Gita for Awakening, Abbot George Burke offers a practical commentary for leading a successful spiritual life. 
        With penetrating insight, he illumines the Bhagavad Gita’s 
        practical value for spiritual seekers, and the timelessness of India’s most beloved scripture."""


# Define a function for summarizing the input text
def summarizer(rawdocs):
    
    # Create a list of stop words for English language
    stopwords = list(STOP_WORDS)
    
    # Load the pre-trained English language model from spaCy
    nlp = spacy.load('en_core_web_sm')
    
    # Parse the input text using spaCy
    doc = nlp(rawdocs)
    
    # Create a list of tokens from the parsed input text
    tokens = [token.text for token in doc]

    # Create a dictionary of word frequency count for each word in the parsed input text
    word_frq = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_frq.keys():
                word_frq[word.text] = 1
            else:
                word_frq[word.text] += 1

    # Normalize the word frequency count dictionary to values between 0 and 1
    max_frq = max(word_frq.values())
    for word in word_frq.keys():
        word_frq[word] =  word_frq[word]/max_frq
        
    # Create a list of sentence tokens from the parsed input text
    sent_tokens = [sent for sent in doc.sents]
    
    # Create a dictionary of sentence scores by summing the normalized frequency count of words present in each sentence
    sent_scores = {}
    for sent in sent_tokens:
        for word in sent:
            if word.text in word_frq.keys():
                if sent not in sent_scores.keys():
                    sent_scores[sent] = word_frq[word.text]
                else:
                    sent_scores[sent] += word_frq[word.text]

    # Calculate the length of the summary to be generated as 30% of the total number of sentences
    select_len = int(len(sent_tokens) * 0.3)
    
    # Extract the top n sentences with the highest scores from the sentence scores dictionary
    summary = nlargest(select_len, sent_scores, key = sent_scores.get)
    
    # Convert the selected sentences into a list of words
    final_summary = [word.text for word in summary]
    
    # Join the list of words into a single string representing the summary of the input text
    summary = " ".join(final_summary)
    
    # Return the summary along with the parsed
    return summary, doc, len(rawdocs.split(' ')), len(summary.split(' '))