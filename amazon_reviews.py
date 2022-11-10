 # Importing requests to extract content from a url
import requests  
from bs4 import BeautifulSoup as bs
import re

from wordcloud import WordCloud
import matplotlib.pyplot as plt


# creating empty reviews list
Samsung_reviews=[]

for i in range(1,21):
  ip=[]  
  url="https://www.amazon.in/product-reviews/B08XJBPCTR/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&filterByStar=five_star&reviewerType=all_reviews&pageNumber="+str(i)  
  response = requests.get(url)
  soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
  reviews = soup.find_all("span", attrs={"class","a-size-base review-text review-text-content"})# Extracting the content under specific tags  
  for i in range(len(reviews)):
    ip.append(reviews[i].text)  
 
  Samsung_reviews = Samsung_reviews + ip  # adding the reviews of one page to empty list which in future contains all the reviews for the wordcloud 

# writng reviews in a text file 
with open("samsung.txt", "w", encoding='utf8') as output:
    output.write(str(Samsung_reviews))
	

# Joinining all the reviews into single paragraph  to remove the unwanted symbols
ip_rev_string = " ".join(Samsung_reviews)

import nltk
from nltk.corpus import stopwords

# Removing unwanted Symbols
ip_rev_string = re.sub("[^A-Za-z" "]+", " ", ip_rev_string).lower()
ip_rev_string = re.sub("[0-9" "]+"," ", ip_rev_string)


ip_reviews_words = ip_rev_string.split(" ")

ip_reviews_words = ip_reviews_words[1:]


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1, 1))
X = vectorizer.fit_transform(ip_reviews_words)

with open("E:\DESKTOPFILES\suraj\Datasets NLP\stop.txt", "r") as sw:
    stop_words = sw.read()
    
stop_words = stop_words.split("\n")

stop_words.extend(["samsung","mobile","time","android","phone","device","product","day"])

ip_reviews_words = [w for w in ip_reviews_words if not w in stop_words]

#### Joinining all the reviews into single paragraph for the wordcloud
ip_rev_string = " ".join(ip_reviews_words)

# WordCloud can be performed on the string inputs.


wordcloud_ip = WordCloud(background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_rev_string)
plt.imshow(wordcloud_ip)

# positive words # Choose the path for +ve words stored in system
with open("E:\DESKTOPFILES\suraj\Datasets NLP\positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")

# Positive word cloud
# Choosing the only words which are present in positive words for the positive Wordcloud
ip_pos_in_pos = " ".join ([w for w in ip_reviews_words if w in poswords])

wordcloud_pos_in_pos = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_pos_in_pos)
plt.figure(2)
plt.imshow(wordcloud_pos_in_pos)

# negative words Choose path for -ve words stored in system
with open(r"E:\DESKTOPFILES\suraj\Datasets NLP\negative-words.txt", "r") as neg:
  negwords = neg.read().split("\n")

# negative word cloud
# Choosing the only words which are present in negwords For the Negative WordCloud
ip_neg_in_neg = " ".join ([w for w in ip_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_neg_in_neg)
plt.figure(3)
plt.imshow(wordcloud_neg_in_neg)


# wordcloud with bigram for the most frequenctly used words 
nltk.download('punkt')
from wordcloud import WordCloud, STOPWORDS

WNL = nltk.WordNetLemmatizer()

# Lowercase and tokenize
text = ip_rev_string.lower()

# Remove single quote causes problems with the tokenizer with help of replace.
text = text.replace("'", "")

tokens = nltk.word_tokenize(text)
text1 = nltk.Text(tokens)

# Remove extra chars and remove stop words which are unnecessary.
text_content = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text1]

# Create a set of stopwords
stopwords_wc = set(STOPWORDS)
customised_words = ['price', 'great'] 

new_stopwords = stopwords_wc.union(customised_words)

# Remove stop words
text_content = [word for word in text_content if word not in new_stopwords]

# Take only non-empty entries with the help of list comprehension 
text_content = [s for s in text_content if len(s) != 0]

##### Best to get the lemmas of each word to reduce the number of similar words
text_content = [WNL.lemmatize(t) for t in text_content]

nltk_tokens = nltk.word_tokenize(text)  
bigrams_list = list(nltk.bigrams(text_content))
print(bigrams_list)

dictionary2 = [' '.join(tup) for tup in bigrams_list]
print (dictionary2)

# Using count vectoriser to view the frequency of bigrams
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(2, 2))
bag_of_words = vectorizer.fit_transform(dictionary2)
vectorizer.vocabulary_

sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
print(words_freq[:100])

# Generating wordcloud
words_dict = dict(words_freq)
WC_height = 1000
WC_width = 1500
WC_max_words = 100
wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width, stopwords=new_stopwords)

wordCloud.generate_from_frequencies(words_dict)
plt.figure(4)
plt.title('Most frequently occurring bigrams connected by same colour and font size')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()
