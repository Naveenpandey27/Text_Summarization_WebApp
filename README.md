# Text_Summarization_WebApp
## I have created this Summarize.AI tool which will summarize the raw text.

# output 1

## write/upload raw text and click on ##submit##
![summarize](https://user-images.githubusercontent.com/66298494/235884068-d89a3e52-ef5b-4f41-b7a2-eb8d02a29d8e.png)

## output 2

## you text will get summarized and display right hand side.
![output2](https://user-images.githubusercontent.com/66298494/235884105-10158ece-5b2a-4d61-94ba-ece7ea93dd26.png)

# This Python code is a simple implementation of a text summarization algorithm that utilizes Natural Language Processing (NLP) tools from the spaCy library. The code takes an input text and returns a summary of the text. The summary is generated by extracting the most important sentences from the input text.

### To use this code, follow these steps:

# Install the necessary libraries
## This code requires the installation of the following libraries:

# spacy
# heapq

### You can install these libraries using pip in your command prompt or terminal:

### pip install spacy
### pip install heapq

2 - Import the necessary libraries

## The first step in the code is to import the necessary libraries. These include:

3 - Set the input text
The input text that you want to summarize should be stored as a string in the 'text' variable.

4 - Define the summarizer function

The summarizer function takes the input text as an argument and returns the summary along with the parsed input text, the total number of words in the input text, and the total number of words in the summary. The summarizer function performs the following steps:

* - Create a list of stop words for the English language
* - Load the pre-trained English language model from spaCy
* - Parse the input text using spaCy
* - Create a list of tokens from the parsed input text
* - Create a dictionary of word frequency count for each word in the parsed input text
* - Normalize the word frequency count dictionary to values between 0 and 1
* - Create a list of sentence tokens from the parsed input text
* Create a dictionary of sentence scores by summing the normalized frequency count of words present in each sentence
* Calculate the length of the summary to be generated as 30% of the total number of sentences
* Extract the top n sentences with the highest scores from the sentence scores dictionary
* Convert the selected sentences into a list of words
* Join the list of words into a single string representing the summary of the input text
* Return the summary along with the parsed input text, the total number of words in the input text, and the total number of words in the summary.

5 - Call the summarizer function
To generate the summary, simply call the summarizer function and pass the input text as an argument. The function will return the summary and other information as described above.

### Note: The accuracy of the summarization algorithm may vary depending on the complexity and length of the input text. This code provides a basic implementation that can be further improved and customized as per the specific requirements.

