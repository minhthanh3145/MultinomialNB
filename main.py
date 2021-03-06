import pandas as pd
import lxml
import requests
from lxml import etree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import warnings

warnings.filterwarnings("ignore")


# This program reads from 2 websites, filter in only the title, description and category.
# A corpus is created, then each description is treated as a word count vector.
# Finally, the classifier is trained using multi binomial Naive Bayesian method


url_list = ["http://feeds.reuters.com/reuters/businessNews",
            "http://feeds.reuters.com/reuters/technologyNews",
            ]
documents =[]

# Function to parse url into xml
for url in url_list:
    response = requests.get(url)
    xml_page = response.text
    parser = lxml.etree.XMLParser(recover=True, encoding='utf-8')
    documents.append(lxml.etree.fromstring(xml_page.encode("utf-8"), parser=parser))

# Function to inspect html tags and contents
def print_tag(node):
    print("<%s %s>%s" % (node.tag, " ".join(["%s=%s" % (k, v) for k, v in node.attrib.iteritems()]), node.text))
    for item in node[:25]:
        print("  <%s %s>%s</%s>" % (
        item.tag, " ".join(["%s=%s" % (k, v) for k, v in item.attrib.iteritems()]), item.text, item.tag))
    print('</%s>' % node.tag)

# Break down of dom tree traversal
temp_node = documents[0]
temp_node = temp_node[0]
temp_node = temp_node.xpath("item")[0]

# Three main variables in interest
title_list = []
description_list = []
category_list = []

# Extract the corresponding information from feeds
for xml_doc in documents:
    articles = xml_doc.xpath("//item")
    for article in articles:
        title_list.append(article[0].text)
        description_list.append(article[1].text)
        category_list.append(article[4].text)

news_data = pd.DataFrame(title_list,columns=['Title'])
news_data["Description"] = description_list
news_data["Category"] = category_list

# Extract description after LOCATIONS (Reuters) - and before the first "<" in <a href>
news_data["Short description"] = [item[item.find(" - ") + 3  : item.find("<")] for item in news_data["Description"]]
corpus = news_data["Short description"]

# Initialize the Vectorizer
vectorizer = CountVectorizer()

# Learn and return term-matrix
X = vectorizer.fit_transform(corpus).toarray()

# Give each category a unique index
categories = news_data["Category"].unique()
category_dict = {value:index for index, value in enumerate(categories)}

# Map 20 rows' categories to unique indices
results = news_data["Category"].map(category_dict)

# Split arrays into random train and test subsets, with default 25% of dataset will be included in test split
x_train, x_test, y_train, y_test = train_test_split(X, results)

# Initialize the classifier and train it with training datasets
classifer = MultinomialNB()
classifer.fit(x_train, y_train)

# Testing for technology
text = ["Tech"]
vec_text = vectorizer.transform(text).toarray()
cat = {k for (k,v) in category_dict.items() if v == classifer.predict(vec_text)[0]}
print(cat)

# Testing for business
text1 = ["Donald Trump"]
vec_text1 = vectorizer.transform(text1).toarray()
cat1 = {k for (k,v) in category_dict.items() if v == classifer.predict(vec_text1)[0]}
print(cat1)
