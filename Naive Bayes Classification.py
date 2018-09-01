from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from nltk.corpus import stopwords as sw

stopwords = set(sw.words('english'))

# News Article related to politics
test_article = ["US sanctions on Iran: Indian macros may face collateral damage - The commentary around US' sanctions on Iran, which have made it quite difficult for India to import cheaper oil, will likely impact key domestic macro-economic indicators, Kotak Institutional Equities said in a report. India imported 4.3 million barrels of crude oil per day in June, of which 680,000 barrels per day came from Iran. The research firm said that even a change of $10 per barrel in the price of crude oil could result in Impact of 50 basis points on CAD/GDP Impact of 30 basis points on inflation Modest impact on GFD through higher subsidies on kerosene and LPG For the equity market, the nature of sanctions on oil exports from Iran will be the biggest short-term variable. Why is there a possibility of crude oil prices rising further? Recent statements by the US about its sanctions on Iran have brought concerns of a rise in crude oil prices back to the table. US' withdrawal from the Iran nuclear deal, its warning to others who conduct business with Iran, and its re-imposition of sanctions on Iran oil exports are seen as some of the reasons. The new US sanctions, effective November 5, 2018, include: Purchase of oil from National Iranian Oil Company and other Iranian oil and gas companies Transactions by foreign financial institutions with the Central Bank of Iran and designated Iranian financial institutions"]

train_data = fetch_20newsgroups(subset='train')
vectorizer = TfidfVectorizer(stop_words=stopwords)
vector = vectorizer.fit_transform(train_data.data)

test_data = fetch_20newsgroups(subset='test')
vector_test = vectorizer.transform(test_data.data)
clf = MultinomialNB(alpha=0.01)
clf.fit(vector, train_data.target)
pred = clf.predict(vector_test)
print("Accuracy wrt given train and test data: ", metrics.f1_score(test_data.target, pred, average='macro'))

test_art = vectorizer.transform(test_article)
print("Predicted class of the new News Article:", clf.predict(test_art)[0], ":", train_data.target_names[clf.predict(test_art)[0]])
