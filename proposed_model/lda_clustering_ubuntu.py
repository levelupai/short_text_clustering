from __future__ import print_function
from time import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pickle
import matplotlib.pyplot as plt

wordslist=[]
wordsplit=[]
all_lines = []
np.random.seed(0)

filename = ('/home/levelup/Ayushi_folder/windows/windows_new_1.csv')
output_file = ('/home/levelup/Ayushi_folder/windows/windows_1_output.csv')
line_labels_in_order = []
with open(filename, "r") as infile:
    with open(output_file, "w+") as outfile:
        infile.readline()
        for raw_line in infile.readlines():
            columns = raw_line.split(",")
            print(columns)
            line_label = int(columns[-1].strip())
            line_labels_in_order.append(line_label)
            line = ''.join(columns[:-1])
            stoplist = set('for a of up into no those many other now which until always called it uses did says say he though like can one all most if or some has been got can will does from than there with got us given what was but havb have  had how when this do you the oh and to in is their its u if are i it not on my an we am you that this at be as who do what me but your dont where so have '.split())
            texts = [word for word in line.lower().split() if word not in stoplist]

            line_without_stopwords = ' '.join(texts).lower()
            outfile.write("\n%s" % line_without_stopwords)
            all_lines.append(line_without_stopwords)

        print(all_lines)


#The maximum distance between the current and predicted word within a sentence.
# alpha (float) – The initial learning rate.
# min_alpha (float) – Learning rate will linearly drop to min_alpha as training progresses.
# min_count (int) – Ignores all words with total frequency lower than this.
# workers (int) – Use these many worker threads to train the model (=faster training with multicore machines).
n_samples = 20000
n_features = 20000
n_components = 2
n_top_words = 20


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


t0 = time()
dataset= all_lines
data_samples = dataset[:n_samples]
print("done in %0.3fs." % (time() - t0))

# Use tf-idf features for NMF.
print("Extracting features for NMF")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features)
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(data_samples)
print(tfidf)
print("done in %0.3fs." % (time() - t0))


print("Extracting features for LDA")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features)
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))
print()

# Fitting the NMF model
print("Fitting the NMF model (Frobenius norm) with tf-idf features, "
      "n_samples=%d and n_features=%d"
      % (n_samples, n_features))
t0 = time()
nmf = NMF(n_components=n_components, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in NMF model (Frobenius norm):")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

# Fitting the NMF model
print("Fitting the NMF model (generalized Kullback-Leibler divergence) with "
      "tf-idf features, n_samples=%d and n_features=%d"
      % (n_samples, n_features))
t0 = time()
nmf = NMF(n_components=n_components, random_state=1,
          beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
          l1_ratio=.5).fit(tfidf)
print("completed  in %0.3fs." % (time() - t0))

print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words= 20)
print("Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d"
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)

# plotting the words according toweights in 2D space

lines = open('word_ids.py', 'r').readline()
word_to_id = eval(lines)

components = None
with open('components_dump.txt', 'br') as c:
    components = pickle.load(c)

windows_sum = sum(components[0]) # windows
not_windows_sum = sum(components[1]) # not windows



colors = []

area = []
for i in range(len(components[0])):
    area.append(max(components[1][i], components[0][i])*10)

    not_windows_proportion = components[1][i] / not_windows_sum
    windows_proportion = components[0][i] / windows_sum
    if not_windows_proportion == windows_proportion:
        colors.append(0)
    elif not_windows_proportion > windows_proportion:
        colors.append(1)
    else:
        colors.append(2)

labels = ['Unknown', 'Non-Windows', 'Windows']
fig, ax = plt.subplots()
color_choices = ['black', 'blue', 'green']
for i in range(1, 3):
    x_subset = []
    y_subset = []
    area_subset = []
    for j in range(len(colors)):
        if colors[j] == i:
            x_subset.append(components[0][j])
            y_subset.append(components[1][j])
            area_subset.append(area[j])
    with open('dumpfile', 'w+') as w:
        w.write("%s %s %s %s %s %s" % (x_subset, y_subset, area_subset, i, 0.5, labels[i]))
    ax.scatter(x_subset, y_subset, s=area_subset, c=color_choices[i], alpha=0.5, label=labels[i])

ax.legend()
ax.grid(False)
plt.show()

