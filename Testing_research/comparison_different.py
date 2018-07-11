
import time
import warnings
from sklearn import cluster, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn import metrics
from sklearn import svm, datasets
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.model_selection import StratifiedKFold


wordslist=[]
wordsplit=[]
all_lines = []
np.random.seed(0)
filename = ('/home/levelup/Ayushi_folder/ubuntuchat/ubuntu_complete_sentences_10k_noleftright.csv')
output_file = ('/home/levelup/Ayushi_folder/ubuntuchat/ubuntu_complete_sentences_10k_output.csv')
cluster_0= open('/home/levelup/Ayushi_folder/ubuntuchat/ubuntu_complete_sentences_10k_output_cluster0.txt', 'w+')
cluster_1= open('/home/levelup/Ayushi_folder/ubuntuchat/ubuntu_complete_sentences_10k_output_cluster1.txt', 'w+')
cluster_2= open('/home/levelup/Ayushi_folder/ubuntuchat/ubuntu_complete_sentences_10k_output_cluster2.txt', 'w+')
cluster_0_ms= open('/home/levelup/Ayushi_folder/ubuntuchat/ubuntu_complete_sentences_10k_output_cluster0_ms.txt', 'w+')
cluster_1_ms= open('/home/levelup/Ayushi_folder/ubuntuchat/ubuntu_complete_sentences_10k_output_cluster1_ms.txt', 'w+')
cluster_2_ms= open('/home/levelup/Ayushi_folder/ubuntuchat/ubuntu_complete_sentences_10k_output_cluster2_ms.txt', 'w+')
cluster_3_ms= open('/home/levelup/Ayushi_folder/ubuntuchat/ubuntu_complete_sentences_10k_output_cluster3_ms.txt', 'w+')

with open(filename, "r") as infile:
    with open(output_file, "w+") as outfile:
        infile.readline() # throw away the first line
        FIRSTLINE = True # record whether this is the first proper line of the file
        for line in infile.readlines():
# removing stop words from the sentences
            stoplist = set('for a of the and to in is their its u if are i it not on my an we am you that this at be as who do what me but your  '.split())
            texts = [word for word in line.lower().split() if word not in stoplist]

            line_without_stopwords = ' '.join(texts).lower()

            if FIRSTLINE:
                outfile.write("%s" % line_without_stopwords)
                FIRSTLINE = False
            else:
                outfile.write("\n%s" % line_without_stopwords)
            all_lines.append(line_without_stopwords)
            print(line_without_stopwords)

sentenceLabeled = []
for sentenceID, sentence in enumerate(all_lines):
    print(sentence)
    sentenceL = TaggedDocument(words=sentence.split(), tags = ['SENT_%s' %sentenceID])
    sentenceLabeled.append(sentenceL)

print(sentenceLabeled)

#The maximum distance between the current and predicted word within a sentence.
# alpha (float) – The initial learning rate.
# min_alpha (float) – Learning rate will linearly drop to min_alpha as training progresses.
# min_count (int) – Ignores all words with total frequency lower than this.
# workers (int) – Use these many worker threads to train the model (=faster training with multicore machines).
model = Doc2Vec(vector_size=2, window=10, min_count=20, workers=11, alpha=0.1,
min_alpha=0.0001)
doc2vec_model_X = model.build_vocab(sentenceLabeled)

print(model.docvecs.vectors_docs)


print("finished doc2vec")

n_samples = len(sentenceLabeled)

doc2vec_data_with_fake_labels = [model.docvecs.vectors_docs, [0,1]*(int(len(model.docvecs.vectors_docs)/2))] # len needs to be even

random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)


# Set up cluster parameters
plt.figure(figsize=(9 * 2 + 3, 12.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1

default_base = {'quantile': .3,
                'eps': .3,
                'damping': .9,
                'preference': -200,
                'n_neighbors': 10,
                'n_clusters': 2}


cluster_2_ms= open('/home/levelup/Ayushi_folder/ubuntuchat/ubuntu_complete_sentences_10k_output_cluster2_ms.txt', 'w+')
datasets = [
    (doc2vec_data_with_fake_labels, {'damping': .77, 'preference': -240, 'quantile': .2, 'n_clusters': 2})]
cluster_2_ms= open('/home/levelup/Ayushi_folder/ubuntuchat/ubuntu_complete_sentences_10k_output_cluster2_ms.txt', 'w+')

for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)
    X, y = dataset

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X, n_neighbors=params['n_neighbors'], include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # Create cluster objects
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    print(ms)

    two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
    print(two_means)
    ward = cluster.AgglomerativeClustering(
        n_clusters=params['n_clusters'], linkage='ward',
        connectivity=connectivity)
    print(ward)
    spectral = cluster.SpectralClustering(
        n_clusters=params['n_clusters'], eigen_solver='arpack',
        affinity="nearest_neighbors")
    print(spectral)
    dbscan = cluster.DBSCAN(eps=params['eps'])
    print(dbscan)
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock",
        n_clusters=params['n_clusters'], connectivity=connectivity)
    print(average_linkage)
    birch = cluster.Birch(n_clusters=params['n_clusters'])
    print(birch)
    gmm = mixture.GaussianMixture(
        n_components=params['n_clusters'], covariance_type='full')
    print(gmm)

    clustering_algorithms = (
        ('MiniBatchKMeans', two_means),
        ('MeanShift', ms),
        ('SpectralClustering', spectral),
        ('Ward', ward),
        ('AgglomerativeClustering', average_linkage),
        ('DBSCAN', dbscan),
        ('Birch', birch),
        ('GaussianMixture', gmm)
    )


    for name, algorithm in clustering_algorithms:
        t0 = time.time()
        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " +
                "connectivity matrix is [0-9]{1,2}" +
                " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding" +
                " may not work as expected.",
                category=UserWarning)
            # actually start processing the data here
            result_of_algorithm = algorithm.fit(X)
            2==3
            wordfreq= []
            total_lines=[]
            if name == 'MiniBatchKMeans':
                for i, line in enumerate(all_lines):
                    if result_of_algorithm.labels_[i] == 0:
                        print(line + ":" + str(result_of_algorithm.labels_[i]))
                        cluster_0.write("%s" % line + ":" + str(result_of_algorithm.labels_[i]) + "\n")
                        cluster_0.flush()
                    elif result_of_algorithm.labels_[i] == 1:
                         print(line + ":" + str(result_of_algorithm.labels_[i]))
                         cluster_1.write("%s" % line + ":" + str(result_of_algorithm.labels_[i]) + "\n")
                         cluster_1.flush()
                    elif result_of_algorithm.labels_[i] == 2:
                         print(line + ":" + str(result_of_algorithm.labels_[i]))
                         cluster_2.write("%s" % line + ":" + str(result_of_algorithm.labels_[i]) + "\n")
                         cluster_2.flush()

            elif name == 'MeanShift':
                for i, line in enumerate(all_lines):
                    if result_of_algorithm.labels_[i] == 0:
                        print(line + ":" + str(result_of_algorithm.labels_[i]))
                        cluster_0_ms.write("%s" % line + ":" + str(result_of_algorithm.labels_[i]) + "\n")
                        cluster_0_ms.flush()
                    elif result_of_algorithm.labels_[i] == 1:
                        print(line + ":" + str(result_of_algorithm.labels_[i]))
                        cluster_1_ms.write("%s" % line + ":" + str(result_of_algorithm.labels_[i]) + "\n")
                        cluster_1_ms.flush()
                    elif result_of_algorithm.labels_[i] == 2:
                        print(line + ":" + str(result_of_algorithm.labels_[i]))
                        cluster_2_ms.write("%s" % line + ":" + str(result_of_algorithm.labels_[i]) + "\n")
                        cluster_2_ms.flush()
                    else:
                        print(line + ":" + str(result_of_algorithm.labels_[i]))
                        cluster_3_ms.write("%s" % line + ":" + str(result_of_algorithm.labels_[i]) + "\n")
                        cluster_3_ms.flush()


        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1
plt.show()

# clustering performance evaluation
def uniform_labelings_scores(score_func, n_samples, n_clusters_range,
                             fixed_n_classes= None, n_runs=5, seed=42):
    random_labels = np.random.RandomState(seed).randint
    scores = np.zeros((len(n_clusters_range), n_runs))

    if fixed_n_classes is not None:
        labels_a = random_labels(low=0, high=fixed_n_classes, size=n_samples)

    for i, k in enumerate(n_clusters_range):
        for j in range(n_runs):
            if fixed_n_classes is None:
                labels_a = random_labels(low=0, high=k, size=n_samples)
            labels_b = random_labels(low=0, high=k, size=n_samples)
            scores[i, j] = score_func(labels_a, labels_b)
    return scores

score_funcs = [
    metrics.adjusted_rand_score,
    metrics.v_measure_score,
    metrics.mutual_info_score,
]

# 2 independent random clusterings with equal cluster number

n_clusters_range = np.linspace(2, n_samples, 10).astype(np.int)

plt.figure(1)

plots = []
names = []
for score_func in score_funcs:
    print("Computing %s for %d values of n_clusters and n_samples=%d"
          % (score_func.__name__, len(n_clusters_range), n_samples))

    t1 = time.time()
    scores = uniform_labelings_scores(score_func, n_samples, n_clusters_range)
    print("done in %0.3fs" % (time.time() - t1))
    plots.append(plt.errorbar(
        n_clusters_range, np.median(scores, axis=1), scores.std(axis=1))[0])
    names.append(score_func.__name__)

plt.title("Clustering measures for 2 random uniform labelings\n"
          "with equal number of clusters")
plt.xlabel('Number of clusters (Number of samples is fixed to %d)' % n_samples)
plt.ylabel('Score value')
plt.legend(plots, names)
plt.ylim(ymin=-0.05, ymax=1.05)


# Random labeling with varying n_clusters against ground class labels
# with fixed number of clusters

n_samples = 1000
n_clusters_range = np.linspace(2, 100, 10).astype(np.int)
n_classes = 10

plt.figure(2)

plots = []
names = []
for score_func in score_funcs:
    print("Computing %s for %d values of n_clusters and n_samples=%d"
          % (score_func.__name__, len(n_clusters_range), n_samples))

    t1 = time.time()
    scores = uniform_labelings_scores( score_func, n_samples, n_clusters_range,
                                      fixed_n_classes=n_classes)
    print("done in %0.3fs" % (time.time() - t1))
    plots.append(plt.errorbar(
        n_clusters_range, scores.mean(axis=1), scores.std(axis=1))[0])
    names.append(score_func.__name__)

plt.title("Clustering measures for random uniform labeling\n"
          "against reference assignment with %d classes" % n_classes)
plt.xlabel('Number of clusters (Number of samples is fixed to %d)' % n_samples)
plt.ylabel('Score value')
plt.ylim(ymin=-0.05, ymax=1.05)
plt.legend(plots, names)
plt.show()


# precision and recall curve
random_state = np.random.RandomState(0)

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=.45,
                                                    random_state=random_state)

# Create a simple classifier

classifier = svm.LinearSVC(random_state=random_state)
classifier.fit(X_train, y_train)
print("training finished")
y_score = classifier.decision_function(X_test)
average_precision = average_precision_score(y_test, y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

precision, recall, _ = precision_recall_curve(y_test, y_score)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
plt.show()


n_samples, n_features = X.shape
# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=6)
classifier = svm.SVC(kernel='linear', probability=True,
                     random_state=random_state)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X, y).predict_proba(X)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y, probas_[:,1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic(ROC) curve')
plt.legend(loc="lower right")
plt.show()
# otherwise the final few lines will be missing from the txt files
cluster_0.flush()
cluster_0.close()

cluster_1.flush()
cluster_1.close()

cluster_2.flush()
cluster_2.close()

