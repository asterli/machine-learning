import sys
from utils import read_csv, read_txt, plot_evs, find_most_sim_word, find_info_ev, comparative_projections, plot_projections, find_analog, check_analogy_task_acc, check_similarity_task_acc
from pca import pca_approx, compute_embedding
import numpy as np

def main():
    # Get the highest step to run from command line arguments (default to 6)
    if len(sys.argv) > 1:
        try:
            run_until = int(sys.argv[1])
        except ValueError:
            print("Please provide an integer for the step number.")
            sys.exit(1)
    else:
        run_until = 6

    ##########################################################################
    # Part 0: Read the files (Make sure the filenames and paths are correct) #
    ##########################################################################
    # M: The co-occurrence matrix
    # words: The set of all words in 'dictionary.txt'
    # analogy_task_words: The set of analogy queries from 'analogy_task.txt'
    # similarity_task_words: The set of word triplets from 'similarity_task.txt'
    
    if run_until >= 0:
        print("\n" + "="*50)
        print("Running Task 0: Reading Files")
        print("="*50)
        M = read_csv("co_occurrence.csv")
        words = read_txt("dictionary.txt")
        words = np.array(words)
        analogy_task_words = read_txt("analogy_task.txt")
        similarity_task_words = read_txt("similarity_task.txt")

    ##########################################################
    # Part 1: Applying PCA to get the low rank approximation #
    ##########################################################
    
    if run_until >= 1:
        print("\n" + "="*50)
        print("Running Task 1: PCA Approximation")
        print("="*50)
        m = 100 # Number of principal components
        Mc, V, eigenvalues, frac_var = pca_approx(M, m)

        # Plot top m eigenvalues
        plot_evs(eigenvalues)

        # Percentage of variance explained
        print(f"Percentage of variance explained by top {m} eigenvectors = {frac_var*100}%")

    #########################################################
    # Part 2: Finding the word most similar to a given word #
    #########################################################
    
    if run_until >= 2:
        print("\n" + "="*50)
        print("Running Task 2: Finding Most Similar Words")
        print("="*50)
        E = compute_embedding(Mc, V)

        # Test embeddings on random words
        word_list = ["university","learning","california"]
        #word_list = ["howdy", "green", "wonderful"]
        for word in word_list:
            most_sim_word = find_most_sim_word(word, words, E)
            print(f"The most similar word to '{word}' is '{most_sim_word}'")

    ########################################################################################
    # Part 3: Looking at the information captured by the principal components/eigenvectors #
    ########################################################################################
    
    if run_until >= 3:
        print("\n" + "="*50)
        print("Running Task 3: Exploring Eigenvector Information")
        print("="*50)
        # Choose a set of eigenvectors: here first 10
        eigenvector_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # Choose the number of top words for each eigenvector: here 10
        num_words = 10

        for index in eigenvector_indices:
            eigenvector = V[:, index]
            info = find_info_ev(eigenvector, words, k=num_words)
            print(f"Top {num_words} words of PC {index}: {info}")

    #############################################################################
    # Part 4: Exploring Semantic/Syntactic Concepts Captured by Word Embeddings #
    #############################################################################
    
    if run_until >= 4:
        print("\n" + "="*50)
        print("Running Task 4: Semantic/Syntactic Projections")
        print("="*50)
        
        # Part 4.1
        word_seq_1 = np.array(['boys', 'girls', 'father', 'mother', 'king', 'queen', 'he', 'she','john', 'mary', 'wall', 'tree', 'baby'])

        proj_seq_1 = comparative_projections(word_seq_1, words, E, "man", "woman")
        plot_projections(word_seq_1, proj_seq_1, filename = 'projections_1.png')

        # Part 4.2
        word_seq_2 = np.array(['mathematics', 'history', 'author', 'doctor', 'teacher', 'engineer', 'science', 'arts', 'literature', 'john', 'mary'])

        proj_seq_2 = comparative_projections(word_seq_2, words, E, "man", "woman")
        plot_projections(word_seq_2, proj_seq_2, filename = 'projections_2.png')

    ################################################
    # Part 5: Finding answers to analogy questions #
    ################################################
    
    if run_until >= 5:
        print("\n" + "="*50)
        print("Running Task 5: Analogy Task")
        print("="*50)
        #analogy_question = 'man woman king queen'
        analogy_question = 'man woman john mary'
        word_seq = analogy_question.split()
        wd4_ans = find_analog(word_seq[0], word_seq[1], word_seq[2], words, E)
        print(f"The answer to the analogy question '{analogy_question}' is '{wd4_ans}'.")

        print(f"Accuracy on the analogy task: {check_analogy_task_acc(analogy_task_words, words, E) * 100}%")

    ###################################################
    # Part 6: Finding answers to similarity questions #
    ###################################################
    
    if run_until >= 6:
        print("\n" + "="*50)
        print("Running Task 6: Similarity Task")
        print("="*50)
        print(f"Accuracy on the similarity task: {check_similarity_task_acc(similarity_task_words, words, E) * 100}%")
        
if __name__ == "__main__":
    main()