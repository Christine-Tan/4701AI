import sys
import string
import math
from collections import defaultdict

"""
Performance Report
######  k=1 m=1 ######
train.txt   0.9751       
dev.txt     0.9515         
test.txt    0.9676

###### k=0.2 m=2 ######
train.txt   0.9913        
dev.txt     0.9803         
test.txt    0.9820

###### k=0.2 m=2 stop words improved ######
train.txt   0.9926      
dev.txt     0.9838         
test.txt    0.9856
"""


class NbClassifier(object):
    """
    A Naive Bayes classifier object has three parameters, all of which are populated during initialization:
    - a set of all possible attribute types
    - a dictionary of the probabilities P(Y), labels as keys and probabilities as values
    - a dictionary of the probabilities P(F|Y), with (feature, label) pairs as keys and probabilities as values
    """

    def __init__(self, training_filename, stopword_file):
        self.attribute_types = set()
        self.label_prior = {}
        self.word_given_label = {}

        self.collect_attribute_types(training_filename)
        if stopword_file is not None:
            self.remove_stopwords(stopword_file)
        self.train(training_filename)

    """
    A helper function to transform a string into a list of word strings.
    You should not need to modify this unless you want to improve your classifier in the extra credit portion.
    """

    def extract_words(self, text):
        # no_punct_text = "".join([x for x in text.lower() if not x in string.punctuation])
        # return [word for word in no_punct_text.split()]
        punctuations = (".", ",", ":", "(", ")", "[", "]", "//")
        words = []
        digit_label = "DIGIT"
        for x in text.split():
            if x not in punctuations:
                if x.isnumeric():
                    words.append(digit_label)
                else:
                    words.append(x)
        return words

    """
    Given a stopword_file, read in all stop words and remove them from self.attribute_types
    Implement this for extra credit.
    """

    def remove_stopwords(self, stopword_filename):
        stopword_file = open(stopword_filename, "r")
        stopwords = set()
        for line in stopword_file.readlines():
            word = line.strip()
            stopwords.add(word)
        self.attribute_types.difference(stopwords)

    """
    Given a training datafile, add all features that appear at least m times to self.attribute_types
    """

    def collect_attribute_types(self, training_filename, m=1):
        # m = 2
        att = set()
        voc = defaultdict(int)
        train_file = open(training_filename, "r")
        for line in train_file.readlines():
            words = line.split("\t")
            for word in self.extract_words(words[1]):
                voc[word] += 1;

        for word in voc:
            if voc[word] >= m:
                att.add(word)

        self.attribute_types = att

    """
    Given a training datafile, estimate the model probability parameters P(Y) and P(F|Y).
    Estimates should be smoothed using the smoothing parameter k.
    """

    def train(self, training_filename, k=1):
        k = 0.2
        self.label_prior = {}
        self.word_given_label = {}
        # store the labels with counts
        labels = defaultdict(int)
        num_labels = 0;
        # store the (word,label) pairs with counts
        word_labels = defaultdict(int)
        train_file = open(training_filename, "r")
        for line in train_file.readlines():
            words = line.split("\t")
            label = words[0]
            labels[label] += 1
            num_labels += 1
            for word in self.extract_words(words[1]):
                word_labels[(word, label)] += 1

        # calculate the possibilities of parameters with smoothing parameter k
        for label in labels:
            self.label_prior[label] = labels[label] / num_labels

        for word in self.attribute_types:
            for label in labels:
                self.word_given_label[(word, label)] = (word_labels[(word, label)] + k) / (
                        labels[label] + k * len(self.attribute_types))

    """
    Given a piece of text, return a relative belief distribution over all possible labels.
    The return value should be a dictionary with labels as keys and relative beliefs as values.
    The probabilities need not be normalized and may be expressed as log probabilities. 
    """

    def predict(self, text):
        words = self.extract_words(text)
        prediction = {}
        for label in self.label_prior:
            prob = math.log(self.label_prior[label], math.e);
            for word in words:
                if (word, label) in self.word_given_label:
                    prob += math.log(self.word_given_label[(word, label)], math.e)

            prediction[label] = prob
        return prediction

    """
    Given a datafile, classify all lines using predict() and return the accuracy as the fraction classified correctly.
    """

    def evaluate(self, test_filename):
        test_file = open(test_filename, "r")
        right_times = 0
        total_times = 0
        for line in test_file.readlines():
            total_times += 1
            words = line.split("\t")
            gold_label = words[0]

            # find the predicted label
            prediction = self.predict(words[1])
            max_prob = -sys.maxsize

            for label in prediction:
                if prediction[label] > max_prob:
                    max_prob = prediction[label]
                    predict_label = label

            if predict_label == gold_label:
                right_times += 1
        return right_times / total_times


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("\nusage: ./hmm.py [training data file] [test or dev data file] [(optional) stopword file]")
        exit(0)
    elif len(sys.argv) == 3:
        classifier = NbClassifier(sys.argv[1], None)
    else:
        classifier = NbClassifier(sys.argv[1], sys.argv[3])
    print(len(classifier.attribute_types))
    # print(classifier.evaluate(sys.argv[2]))


