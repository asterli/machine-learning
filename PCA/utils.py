import numpy as np
import matplotlib.pyplot as plt
import csv

############################
# Read .csv and .txt files #
############################
def read_csv(path = 'co_occurrence.csv'):
    """
    Reads a .csv file.

    Input:
        - path: Path of the .csv file

    Return:
        - data: A 3,000 x 3,000 numpy array (the covariance matrix)
    """

    data = []

    with open(path, 'r') as file:

        csvreader = csv.reader(file)

        for row in csvreader:
            a = []

            for el in row:
                a.append(float(el))

            a = np.array(a)
            data.append(a)

        data = np.array(data)
        print(np.shape(data))

    return data


def read_txt(path):
    """
    Reads a .txt file.

    Input:
        - path: Path of the .txt file

    Return:
        - words: A list of strings (i^th item is the i^th row from the file)
    """

    with open(path) as f:

        data = f.read()
        words = data.split("\n")

        print(len(words))

    return words


#################################
# Plot Eigenvalues (For Part 1) #
#################################
def plot_evs(D):
    '''
    Input:
        - D: A sequence of eigenvalues

    Plots the eigenvalues against their indices and save
    the figure in 'ev_plot.png'
    '''
    n = np.shape(D)[0]
    x = range(1, n+1)
    fig, ax = plt.subplots()
    ax.plot(x, D)

    ax.set(xlabel = 'Index', ylabel = 'Eigenvalue')
    fig.savefig("ev_plot.png")
    plt.show()


###################################################################################
# Find the Embedding of a Given Word (e.g. 'man', 'woman') (For Parts 2, 4, 5, 6) #
###################################################################################
def find_embedding(word, words, E):
    '''
    Inputs:
        - word: The query word (string)
        - words: The list of all words read from dictionary.txt
        - E: The embedding matrix (3,000 x m), where m = length of embeddings
    Return:
        - emb: The word embedding
    '''
    ##################################################################
    # TODO: Implement the following steps:
    # i) Find the index/position of 'word' in 'words'.
    # ii) The row of 'E' at this index will be the embedding 'emb'.
    ##################################################################
    i = np.where(words == word)[0][0]
    emb = E[i]
    return emb


###########################################################
# Find the Word Most Similar to a Given Word (For Part 2) #
###########################################################
def find_most_sim_word(word, words, E):
    '''
    Inputs:
        - word: The query word (string)
        - words: The list of all words read from dictionary.txt
        - E: The embedding matrix (3,000 x m), where m = length of embeddings
    Return:
        - most_sim_word: The word which is most similar to the input 'word'
    '''
    ###############################################################################################################
    # TODO: Implement the following steps:
    # i) Get the word embedding of 'word' using the 'find_embedding' function.
    # ii) Compute the similarity (dot product) of this embedding with every embedding, i.e. with each row of 'E'.
    # iii) Find the index of the row of 'E' which gives the largest similarity, excluding the row which is the
    # embedding of 'word' from the search (for this, you will need to find the index/position of 'word' in 'words').
    # iv) Find the word corresponding to that index from 'words'.
    ###############################################################################################################
    w = find_embedding(word,words,E)
    similarity = E @ w
    
    i = np.where(words == word)[0][0]
    similarity[i] = -np.inf
    best_i = np.argmax(similarity)
    
    most_sim_word = words[best_i]
    return most_sim_word


############################################################
# Interpret Principal Components/Eigenvectors (For Part 3) #
############################################################
def find_info_ev(v, words, k=20):
    # Find words corresponding to the 'k' largest magnitude elements of an eigenvector,
    # to see what kind of information is captured by 'v'.
    '''
    Inputs:
        - v: An eigenvector (with 3,000 entries)
        - words: The list of all words read from dictionary.txt
        - k: The number of words the function should return
    Return:
        - info: The set of k words
    '''
    #########################################################################################################################
    # TODO: Implement the following steps:
    # i) Obtain a set of indices (postions in 'v') which would sort the entries of 'v' in decreasing order of absolute value.
    # ii) Using the set of first 'k' indices, get the corresponding words from the list 'words'.
    ##########################################################################################################################
    indices = np.argsort(np.abs(v))[::-1][:k]
    
    info = words[indices]
    
    return info


################################################################################
# Explore Semantic/Syntactic Concepts Captured by Word Embeddings (For Part 4) #
################################################################################

def plot_projections(word_seq, proj_seq, filename = 'projections.png'):
    '''
    Inputs:
        - word_seq: A sequence of words (strings), used as labels.
        - proj_seq: A sequence of projection values.
        - filename: The plot will be saved

    Plot the set of values in 'proj_seq' on a line with the corresponding labels from 'word_seq and
    save the figure
    '''
    y = np.zeros(len(proj_seq))
    fig, ax = plt.subplots()
    ax.scatter(proj_seq, y)

    # to avoid overlaps in the plot
    plt.rcParams.update({'font.size': 9})

    # Convert proj_seq to a NumPy array for proper indexing
    proj_seq = np.array(proj_seq)

    idx = np.argsort(proj_seq)
    for i in range(len(idx)):
        idx[i] = int(idx[i])
    proj_seq = proj_seq[idx]
    word_seq = word_seq[idx]
    del_y_prev = 0

    for i, label in enumerate(word_seq):
        # to avoid overlaps in the plot
        if i<len(proj_seq)-1:
            if np.abs(proj_seq[i]-proj_seq[i+1])<0.02:
                del_y0 = -0.005-0.0021*len(label)
                if del_y_prev == 0:
                    del_y = del_y0
                else:
                    del_y = 0
                del_y_prev = del_y0
            elif (del_y_prev!=0 and del_y == 0 and np.abs(proj_seq[i]-proj_seq[i-1])<0.02):
                del_y = -0.005-0.0021*len(label)
                del_y_prev = del_y
            else:
                del_y = 0
                del_y_prev = 0

        ax.text(x = proj_seq[i]-0.01, y = y[i]+0.005+del_y, s = label, rotation = 90)

    ax.set_xlim(-0.5,0.5)

    fig.savefig(filename)
    plt.show()


def get_projections(word_seq, words, E, w):
    '''
    Inputs:
        - word_seq: A sequence of words (strings)
        - words: The list of all words read from dictionary.txt
        - E: The embedding matrix (3,000 x m), where m = length of embeddings
        - w: An embedding vector (with m elements)
    Return:
        - proj_seq: The sequence of projections of the words in 'word_seq'
    '''
    ###############################################################################################################
    # TODO: Implement the following steps:
    # i) For each word in 'word_seq':
    #     - Get its word embedding using the 'find_embedding' function.
    #     - Find the projection of this embedding onto 'w', i.e. the dot product with normalized 'w'.
    # ii) Return the set of projections.
    ###############################################################################################################
    w /= np.linalg.norm(w)
    proj_seq = []
    for word in word_seq:
        emb = find_embedding(word, words,E)
        proj_seq.append(np.dot(emb, w))
    
    
    return proj_seq

def comparative_projections(word_seq, words, E, wd1, wd2):
    '''
    Inputs:
        - word_seq: A sequence of words (strings)
        - words: The list of all words read from dictionary.txt
        - E: The embedding matrix (3,000 x m), where m = length of embeddings
        - wd1, wd2: A pair of words
    Return:
        - proj_seq: The sequence of projections of the words in 'word_seq' on the difference between the embeddings of wd1 and wd2
    '''
    ###############################################################################################################
    # TODO: Implement the following steps:
    # i) Use the 'find_embedding' function to find w1 and w2 (the embeddings for wd1 and wd2, respectively).
    # ii) Get w = w1 - w2 and normalize it.
    # iii) Return the set of projections of the words in word_seq on normalized 'w'.
    ###############################################################################################################
    w1 = find_embedding(wd1,words,E)
    w2 = find_embedding(wd2,words,E)
    
    w = w1-w2
    w /= np.linalg.norm(w)
    
    proj_seq = get_projections(word_seq, words, E, w)
    return proj_seq

#####################################################
# Finding Answers to Analogy Questions (For Part 5) #
#####################################################

def find_most_sim_word_w(w, words, E, ids):
    # Find the word whose embedding is most similar to a given vector in the embedding space.
    # Similar to the 'find_most_sim_word' function.
    '''
    Inputs:
        - w: The query vector (with m entries) [for an analogy question, this is a combination of three
                                word embeddings (say words wd1, wd2, wd3), as mentioned in the problem statement]
        - words: The list of all words read from dictionary.txt
        - E: The embedding matrix (3,000 x m)
        - ids: The indices of words (wd1, wd2, wd3) which need to be excluded from the search while finding the most similar word
    Return:
        - most_sim_word: The word whose embedding which is most similar to the input 'w'
    '''
    ########################################################################################################################################
    # TODO: Implement the following steps:
    # i) Compute the similarity (dot product) of 'w' with every embedding, i.e. with each row of 'E'.
    # ii) Find the index of the row of 'E' which gives the largest similarity, exculding the rows at indices given by 'ids' from the search.
    # iii) Find the word corresponding to that index from 'words'.
    #########################################################################################################################################
    similarity = E @ w
    
    for i in ids:
        similarity[i] = -np.inf
    idx = np.argmax(similarity)
    
    most_sim_word = words[idx]      
    return most_sim_word

def find_analog(wd1, wd2, wd3, words, E):
    '''
    Inputs:
        - wd1, wd2, wd3: Three words (strings) posing an analogy question (e.g. 'man', 'woman', 'king')
        - words: The list of all words read from dictionary.txt
        - E: The embedding matrix (3,000 x m)
    Return:
        - wd4: The word whose embedding is most similar to the input 'w'
    '''
    ########################################################################################################################################
    # TODO: Implement the following steps:
    # i) Find the word embeddings of wd1, wd2 and wd3 using the 'find_embedding' function: w1, w2, w3, respectively.
    # ii) Get vector w = w2 - w1 + w3, and normalize it.
    # iii) Find the indices/positions of wd1, wd2, wd3 from the list 'words'.
    # iv) Find wd4, the closest word to w (excluding the given three words from the search by using the indices from the previous step),
    # using the 'find_most_sim_word_w' function. This will be the answer to the analogy question.
    #########################################################################################################################################
    w1 = find_embedding(wd1, words, E)
    w2 = find_embedding(wd2, words, E)
    w3 = find_embedding(wd3, words, E)
    
    w = w2 - w1 + w3
    w /= np.linalg.norm(w)
    
    ids = [
        np.where(words==wd1)[0][0],
        np.where(words==wd2)[0][0],
        np.where(words==wd3)[0][0]
    ]
    
    wd4 = find_most_sim_word_w(w, words, E, ids)
    return wd4

def check_analogy_task_acc(task_words, words, E):
    # Check the accuracy for the analogy task by finding answers to a set of analogy questions #
    '''
    Inputs:
        - task_words: The list of word sequences (wd1, wd2, wd3, wd4), for an analogy statement (e.g. 'man', 'woman', 'king', queen)
        - words: The list of all words read from dictionary.txt
        - E: The embedding matrix (3,000 x m)
    Returns:
        - acc: The average accuracy for the task
    '''

    acc = 0

    for word_seq in task_words:
        word_seq = word_seq.split()
        wd4_ans = find_analog(word_seq[0], word_seq[1], word_seq[2], words, E)
        if wd4_ans == word_seq[3]:
            acc+=1
    acc = acc/len(task_words)

    return acc

##################################################################
# Exploring Relationship with Synonyms and Antonyms (For Part 6) #
##################################################################

def similarity(wd1, wd2, words, E):
    '''
    Inputs:
        - wd1, wd2: Two words (strings) to be compared for similarity (e.g. 'often', 'frequently' or 'often', 'rare')
        - words: The list of all words read from dictionary.txt
        - E: The embedding matrix (3,000 x m)
    Return:
        - similarity: The cosine similarity between the words wd1 and wd2
    '''
    ########################################################################################################################################
    # TODO: Implement the following steps:
    # i) Find the word embeddings of wd1 and wd2 using the 'find_embedding' function: w1 and w2, respectively.
    # ii) Get the cosine similarity between wd1 and wd2, which is given by the dot product of w1 and w2.
    #########################################################################################################################################
    w1 = find_embedding(wd1, words, E)
    w2 = find_embedding(wd2, words, E)
    similarity = np.dot(w1,w2)
    return similarity

def check_similarity_task_acc(task_words, words, E):
    # Check the accuracy for the similarity task by differentiating the synonym from the antonym for each word #
    '''
    Inputs:
        - task_words: The list of word sequences (wd1, wd2, wd3), where wd2 is a synonym of wd1 
                                                    and wd3 is an antonym of wd1 (e.g. 'often', 'frequently', 'rare')
        - words: The list of all words read from dictionary.txt
        - E: The embedding matrix (3,000 x m)
    Returns:
        - acc: The average accuracy for the task
    '''

    acc = 0

    for word_seq in task_words:
        word_seq = word_seq.split()
        if similarity(word_seq[0],word_seq[1],words,E) > similarity(word_seq[0],word_seq[2],words,E):
            acc+=1
    acc = acc/len(task_words)

    return acc