import math
import numpy as np
import matplotlib.pyplot as plt

''' READING THE DATA FROM THE FILE'''
file_object = open("SMSSpamCollection", "r")
messages = file_object.read()
message_lines = messages.split("\n")
'''print(message_lines_list)'''

'''CONVERTING INTO LOWERCASE AND REMOVING UNWANTED PUNCTUATIONS'''
lowercase_lines_list = []
punctuation_lines_list = []

for j in message_lines:
    lowercase_lines_list.append(j.lower())
'''print(lowercase_lines_list)'''
lst = []
for k in lowercase_lines_list:
    punctuation_lines = ""
    for l in k:
        if (l >= 'a' and l <= 'z') or (l == '\t') or (l == ' '):
            punctuation_lines = punctuation_lines + l
    lst.append(punctuation_lines)
'''print(len(lst))'''

'''IGNORING LAST MESSAGE AS IT IS SPACE'''
lst_all = []
for i in lst:
    if (len(lst_all) < len(lst)-1):
        lst_all.append(i)
'''print(lst_all)
print(len(lst_all))'''

'''CLASSIFYING THE LABELS AND MESSAGES INTO SPAM AND HAM'''
labels = []
messages = []

for j in lst_all:
    classify = j.split('\t')
    if classify[0] == "spam":
        labels.append(0)
        messages.append(classify[1])
    elif classify[0] == "ham":
        labels.append(1)
        messages.append(classify[1])
'''print("labels")
print(labels)
print("messages")
print(messages)
print(len(labels))
print(len(messages))'''

'''SHUFFLING THE INDEXES OF ALL MESSAGES'''
random_shuffle = np.arange(len(lst_all))
np.random.shuffle(random_shuffle)
len_of_eighty_percent = math.floor(0.8 * len(lst_all))
len_of_twenty_percent = math.floor(0.2 * len(lst_all))

'''SPLITTING DATA INTO TRAINING AND TESTING SETS'''
training_messages = []
training_labels = []
testing_messages = []
testing_labels = []

for i in random_shuffle:
    if len(training_messages) < len_of_eighty_percent:
        training_messages.append(messages[i])
    else:
        testing_messages.append(messages[i])

    if len(training_labels) < len_of_eighty_percent:
        training_labels.append(labels[i])
    else:
        testing_labels.append(labels[i])

'''print(training_messages)
print(len(training_messages))
print(training_labels)
print(len(training_labels))
print(testing_messages)
print(len(testing_messages))
print(testing_labels)
print(len(testing_labels))'''

'''FUNCTION TO TEST TRAINING AND TESTING DATASET'''

classifier_calculations = {}

'''CALCULATING THE COUNT OF WORDS IN SPAM AND HAM MESSAGES'''


def classifier(training_messages, training_labels, testing_messages, testing_labels, alpha):
    spam_counts = {}
    ham_counts = {}
    num_of_spam_words = 0
    num_of_ham_words = 0
    for i in range(len(training_messages)):
        temp = []
        temp = training_messages[i].split()
        for j in range(len(temp)):
            word = temp[j]
            if training_labels[i] == 0:
                num_of_spam_words = num_of_spam_words + 1
                if word in spam_counts:
                    spam_counts[word] = spam_counts[word] + 1
                else:
                    spam_counts.update({word: 1 + alpha})
            else:
                num_of_ham_words = num_of_ham_words + 1
                if word in ham_counts:
                    ham_counts[word] = ham_counts[word] + 1
                else:
                    ham_counts.update({word: 1 + alpha})

    ''' print(spam_counts)
    print(len(spam_counts))
    print(num_of_spam_words)
    print(ham_counts)
    print(len(ham_counts))
    print(num_of_ham_words)'''

    no_of_spam_messages = 747
    no_of_ham_messages = 4827
    correct_prediction_ham = 0
    incorrect_prediction_ham = 0
    correct_prediction_spam = 0
    incorrect_prediction_spam = 0

    for i in range(len(testing_messages)):
        probability_spam_message = no_of_spam_messages / (no_of_spam_messages + no_of_ham_messages)
        probability_ham_message = no_of_ham_messages / (no_of_spam_messages + no_of_ham_messages)
        temp1 = []
        temp1 = testing_messages[i].split()
        for j in temp1:
            if j in spam_counts and j in ham_counts:
                probability_spam_message = probability_spam_message * (spam_counts[j] / (num_of_spam_words + (alpha * 20000)))
                probability_ham_message = probability_ham_message * (ham_counts[j]/(num_of_ham_words + (alpha * 20000)))

        if probability_ham_message > probability_spam_message:
            if testing_labels[i] == 1:
                correct_prediction_ham = correct_prediction_ham + 1
            else:
                incorrect_prediction_ham = incorrect_prediction_ham + 1
        else:
            if testing_labels[i] == 0:
                correct_prediction_spam = correct_prediction_spam + 1
            else:
                incorrect_prediction_spam = incorrect_prediction_spam + 1
    '''print(correct_prediction_ham)
    print(incorrect_prediction_ham)
    print(correct_prediction_spam)
    print(incorrect_prediction_spam)'''

    '''ACCURACY'''
    accuracy = ((correct_prediction_spam + correct_prediction_ham)/(correct_prediction_ham + correct_prediction_spam + incorrect_prediction_spam + incorrect_prediction_ham))*100

    '''CONFUSION MATRIX'''
    confusion_matrix = [[0 for x in range(2)] for y in range(2)]
    confusion_matrix[0][0] = correct_prediction_spam
    confusion_matrix[0][1] = incorrect_prediction_spam
    confusion_matrix[1][0] = incorrect_prediction_ham
    confusion_matrix[1][1] = correct_prediction_ham

    '''PRECISION'''
    precision = correct_prediction_spam / (correct_prediction_spam + incorrect_prediction_spam)

    '''RECALL'''
    recall = correct_prediction_spam / (correct_prediction_spam + incorrect_prediction_ham)

    '''F-SCORE'''
    fscore = 2 * ((precision * recall) / (precision + recall))

    classifier_calculations = {"Alpha": alpha, "Accuracy": accuracy, "Confusion Matrix": confusion_matrix,
                               "Precision": precision, "Recall": recall, "F-Score": fscore}

    return classifier_calculations


'''Result of PART A with alpha = 0.1'''
alpha = 0.1
result_a = {}
a_results = []
result_a = classifier(training_messages, training_labels, testing_messages, testing_labels, alpha)
a_results.append(result_a)
print("PART A with alpha = 0.1")
print(result_a)

'''Results of PART B with various alpha values for training set'''
values_a = [-5, -4, -3, -2, -1, 0]
results_training = []
training_results = {}
for i in values_a:
    alpha = 2 ** i
    training_results = classifier(training_messages, training_labels, training_messages, training_labels, alpha)
    results_training.append(training_results)
print("PART B for training set")
print(results_training)

'''Results of PART B with various alpha values for testing set'''
values_b = [-5, -4, -3, -2, -1, 0]
results_testing = []
testing_results = {}

for j in values_b:
    alpha = 2 ** j
    testing_results = classifier(training_messages, training_labels, testing_messages, testing_labels, alpha)
    results_testing.append(testing_results)
print("PART B for testing set")
print(results_testing)

'''PLOT 1'''
x1 = [-5, -4, -3, -2, -1, 0]
y1 = [training_results["Accuracy"] for training_results in results_training]
plt.plot(x1, y1, label='training', color='green', linestyle='dashed', linewidth=3,marker='o', markerfacecolor='blue', markersize=12)

x2 = [-5, -4, -3, -2, -1, 0]
y2 = [testing_results["Accuracy"] for testing_results in results_testing]
plt.plot(x2, y2, label='testing', color='red', linestyle='dashed', linewidth=3, marker='o', markerfacecolor='yellow', markersize=12)
plt.ylim(80, 99)
plt.xlim(-5, 0)
plt.xlabel("VALUES")
plt.ylabel("ACCURACY")
plt.title("TRAINING AND TEST ACCURACY OVER DIFFERENT VALUES OF ALPHA")
plt.legend()
plt.show()

'''PLOT 2'''
x3 = [-5, -4, -3, -2, -1, 0]
y3 = [training_results["F-Score"] for training_results in results_training]
plt.plot(x3, y3, label='training', color='orange', linestyle='dashed', linewidth=3,
         marker='o', markerfacecolor='skyblue', markersize=12)
x4 = [-5, -4, -3, -2, -1, 0]
y4 = [testing_results["F-Score"] for testing_results in results_testing]
plt.plot(x4, y4, label='testing', color='pink', linestyle='dashed', linewidth=3,
         marker='o', markerfacecolor='blue', markersize=12)
plt.ylim(0.70, 0.97)
plt.xlim(-5, 0)
plt.xlabel("VALUES")
plt.ylabel("F-SCORE")
plt.title("TRAINING AND TEST F-SCORE OVER DIFFERENT VALUES OF ALPHA")
plt.legend()
plt.show()







