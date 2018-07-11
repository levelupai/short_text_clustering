import pickle
import numpy
import random

lines = open('word_ids.py', 'r').readline()
word_to_id = eval(lines)

components = None
with open('components_dump.txt', 'br') as c:
    components = pickle.load(c)

windows_sum = sum(components[0]) # windows
not_windows_sum = sum(components[1]) # not windows

# recall = probability of detecting 0 (windows) when it really is 0
# false positive rate = probability of detecting 0 (windows) when it really is 1

def get_score(filename, expected, threshold_val):
    correct = 0
    total = 0
    unmatched = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    with open(filename, 'r') as c:
        lines = c.readlines()
        for line in lines:
            score_for_0 = 0.0
            score_for_1 = 0.0
            significant_words = 0
            new_line = line.lower().strip().replace("\"", "")
            words = new_line.split(" ")
            for word in words:
                if word in word_to_id:
                    if components[0][word_to_id[word]] > threshold_val or components[1][word_to_id[word]] > threshold_val:
                        score_for_0 += (components[0][word_to_id[word]] / windows_sum)
                        score_for_1 += (components[1][word_to_id[word]] / not_windows_sum)
                        significant_words += 1
                    #print(word)
            if score_for_0 == score_for_1:
                final_category = "Undecided"
                unmatched += 1
                score_for_0 += random.randint(1,100000)
                score_for_1 += random.randint(1,100000)
                #false_positive += 0.5
                #true_positive += 0.5
                #false_positive += 1
                #false_negative += 1

            elif score_for_0 > score_for_1:
                final_category = "0"
                if expected == 0:
                    correct += 1
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                final_category = "1"
                if expected == 1:
                    correct += 1
                    true_negative += 1
                else:
                    false_negative += 1
            #print("%s, sig words: %s, 0: %s, 1: %s, final category: %s" % (line, significant_words, score_for_0, score_for_1, final_category))
            total += 1
    incorrect = total - correct - unmatched
    # correct is true positive rate
    return [correct, incorrect, total, unmatched, true_positive, true_negative, false_positive, false_negative]


not_windows_correct_rate = []
windows_correct_rate = []
total_size = 0
csvfile = open('roc.csv', 'w+')

for threshold in range(0, 101):
    scaled_threshold = threshold / 10.0
    correct, incorrect, total, unmatched, true_positive, true_negative, false_positive, false_negative = get_score('../windows/out_non_random100.csv', 1, scaled_threshold)
    correct2, incorrect2, total2, unmatched2, true_positive2, true_negative2, false_positive2, false_negative2 = get_score('../windows/out_random100.csv', 0, scaled_threshold)
    #print("windows", correct, incorrect, total, unmatched)
    csvfile.write('%s,%s,%s,%s,%s\n' % (scaled_threshold, total, unmatched, false_positive / total, true_positive2 / (total2)))
    csvfile.flush()

csvfile.close()
