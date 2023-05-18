
# Preprocessing Text Data

NLP Text preprocessing is a method to clean the text in order to make it ready to feed to models. Noise in the text comes in varied forms like emojis, punctuations, different cases. All these noises are of no use to machines and hence need to clean it.

These are some text preprocessing steps that we can add or remove as per the dataset we have:

Step-1 : Remove newlines & Tabs

Step-2 : Strip HTML Tags

Step-3 : Remove Links

Step-4 : Remove Whitespaces

Step-5 : Remove Accented Characters

Step-6 : Case Conversion

Step-7 : Reducing repeated characters and punctuations

Step-8 : Expand Contractions

Step-9 : Remove Special Characters

Step-10: Remove Stopwords

Step-11: Correcting Mis-spelled words

Step-12: Lemmatization / Stemming

#

# Tokenizing


While there's of course been an incredible amount of NLP research in the past decade, a key under-explored area is Tokenization, which is simply the process of converting natural text into smaller parts known as "tokens." 

Tokens here is an ambiguous term. A token that can refer to words, subwords, or even characters. It basically refers to the process of breaking down a sentence into its constituent parts. For instance, in the statement: "Coffee is superior than chai" One way to split the statement into smaller chunks is separate out words from the sentence i.e. "Coffee", "is" "superior", "than", "chai" That's the essence of Tokenization.


As we know that NLP is used to build applications such as sentiment analysis, QA systems, languag translation, smart chatbots, voice systems, etc., hence, in order to build them, it becomes vital to understand the pattern in the text. The tokens, mentioned above, are very useful in finding and understanding these patterns. We can consider tokenization as the base step for other recipes such as stemming and lemmatization.

# Stemming:
Stemming is the process of reducing a word to its word stem that affixes to suffixes and prefixes or to the roots of words known as a lemma. Stemming is important in natural language understanding (NLU) and natural language processing (NLP).

 For example, an error can reduce words like laziness to lazi  instead of lazy.

#### Problems in Stemming:
There are two issues we face in stemming; those are ‘Over-Stemming’ & ‘Under-Stemming’.

#### Over-Stemming :
A simple explanation of over-stemming is when you pass words with different meanings but the stemmer is returning the same stem word for all. Have a look at the diagram below then I’ll show you it in practical implementation too.

![](Screenshot/over_stemming.PNG)

We all know these three words Universal,University & Universe are different in meaning but still we are returned with the same stemmed word for all this is called over-stemming where the meaning is different but stemmer is returning us the same root word for words with different meaning.

#### Under-Stemming :

Under-stemming is in a way you can say it's opposite of over-stemming. Here you pass words with the same meaning but stemmer returns you the different root words.
![](Screenshot/under_stemming.PNG)
Above two words are the same, the only difference is singular and plural so we should have the same root word for both but this isn’t happening here it is providing us with two different root words.





