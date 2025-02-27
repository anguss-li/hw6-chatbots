import numpy as np
import argparse
import joblib
import re
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
import nltk
from collections import defaultdict, Counter
from typing import List, Dict, Union, Tuple

import util


class Chatbot:
    """Class that implements the chatbot for HW 6."""

    def __init__(self):
        # The chatbot's default name is `moviebot`.
        self.name = "film-reviewer"

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, self.ratings = util.load_ratings('data/ratings.txt')

        # Load sentiment words
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

        # Train the classifier
        self.train_logreg_sentiment_classifier()

        # TODO: put any other class variables you need here
        self.all_title_idxs = []
        self.user_ratings = {}
        self.recommendations = []

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.
        """

        intro = """ Hello, I'am your personalized movie recommending system, "Sydney".
        I will first ask you about your movie preferences, and then use that information
        to recommend movies. To exit the
        program, write ":quit" and press enter.

        """

        return intro

        """
        Your task is to implement the chatbot as detailed in the HW6
        instructions (README.md).

        To exit: write ":quit" (or press Ctrl-C to force the exit)

        MovieBot is designed to make movie recommendations to the user, asking
        them to enter the names of movies they have liked (or disliked) and prompting
        them to disambiguate between film titles when necessary. It determines
        whether the user liked or disliked a given title by analysing what they
        have written, using that to predict a new movie they might enjoy.
        """

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # Write a short greeting message                                 #
        ########################################################################

        greeting_message = """
        Hi! I'm film-reviewer! I'm going to recommend a movie to you. First I will ask
        you about your taste in movies. Tell me about a movie that you have seen.
        Please write in the format, e.g
        "I liked "[Movie name]", or I disliked "[Movie name].
        """

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "Thanks! It was fun chatting with you!  "

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message

    def debug(self, line):
        """
        Returns debug information as a string for the line string from the REPL

        No need to modify this function.
        """
        return str(line)

    ############################################################################
    # 2. Extracting and transforming                                           #
    ############################################################################

    def process(self, line: str) -> str:
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this script.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        Arguments:
            - line (str): a user-supplied line of text

        Returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
        # line = self.function1(line)

        if len(self.all_title_idxs) > 1:
            # Since all_title_idxs has not been modified yet this iteration of
            # the loop, if it has more than one element then we know we asked
            # for a clarification last iteration, so we treat user_input as such
            self.all_title_idxs = self.disambiguate_candidates(
                line, self.all_title_idxs)
        else:
            # This is a new movie. We extract the titles from the user input
            # and all the indexes associated with said input
            titles = self.extract_titles(line)

            # If user did not use quotes
            if len(titles) == 0:
                return "I am sorry. I did not understand. Please enter \"[Movie name]\" in quotation marks."

            self.all_title_idxs = self.get_all_title_idx(titles)

        # If they enter a movie that does not exist in our database
        if len(self.all_title_idxs) == 0:
            return "I am sorry. The movie you entered does not exist in our repository. Please enter a new movie."

        # If needed, ask the user for clues for disambiguation
        if len(self.all_title_idxs) > 1:
            return "Did you mean: \n" + ", or \n".join(
                self.find_movies_title_by_idx(self.all_title_idxs)) + "?"

        # Predict the sentiment of the user regarding the movie given their input
        user_idx = self.all_title_idxs[0]
        self.user_ratings[user_idx] = self.predict_sentiment_statistical(self.function1(line))

        if len(self.user_ratings) >= 5 and not self.recommendations:
            self.recommendations = self.recommend_movies(self.user_ratings)
            next_recc = 0
            continue_recc = "yes"
            print("Thanks! That's enough for me to make a recommendation. ")
            while continue_recc != "no":
                print("I suggest you watch",
                      self.recommendations[next_recc] + ".")
                print(
                    "Would you like to hear another recommendation? (Or enter 'no' if you're done.)")
                next_recc += 1
                continue_recc = input("> ").lower()
                if next_recc >= len(self.recommendations):
                    return "That's all of the recommendations I have! Enter ':quit' if you're done."
            return "It was nice chatting with you! Enter ':quit' if you're done."

        # prompt the user about their input + ask them about their next choice.
        return self.function2(line) + self.get_title(user_idx)

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
    def get_all_title_idx(self, titles_list):
        """
        Given a list of titles, returns a list of list of index of all matching movies.
        """
        titles = []
        for item in titles_list:
            titles.extend(self.find_movies_idx_by_title(item))
        return titles

    def extract_titles(self, user_input: str) -> list:
        """Extract potential movie titles from the user input.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example 1:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I do not like any movies'))
          print(potential_titles) // prints []

        Example 2:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        Example 3:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'There are "Two" different "Movies" here'))
          print(potential_titles) // prints ["Two", "Movies"]

        Arguments:
            - user_input (str) : a user-supplied line of text

        Returns:
            - (list) movie titles that are potentially in the text

        Hints:
            - What regular expressions would be helpful here?
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################
        # Captures any possible sequence of characters between quotes
        title_regex = r'"(.*?)"'
        return re.findall(title_regex, user_input)
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    def find_movies_title_by_idx(self, idxs: List[int]) -> List[str]:
        """Given a list of indices, return a list of titles of matching movies
        The titles correspond to those in data/movies.txt.

        (We're adding this little helper function to make things cleaner)

        Args:
            idxs (List[int]): the list of movie indices to be matched to titles

        Returns:
            List[str]: the corresponding titles.
        """
        return [self.get_title(i) for i in idxs]

    def get_title(self, idx: int) -> str:
        """Quick helper function to abstract accessing self.titles.

        Args:
            idx (int): the index to be matched to title.

        Returns:
            str: the corresponding title in self.titles.
        """
        return self.titles[idx][0]

    def find_movies_idx_by_title(self, title: str) -> list:
        """ Given a movie title, return a list of indices of matching movies
        The indices correspond to those in data/movies.txt.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list that contains the index of that matching movie.

        Example 1:
          ids = chatbot.find_movies_idx_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        Example 2:
          ids = chatbot.find_movies_idx_by_title('Twelve Monkeys')
          print(ids) // prints [31]

        Arguments:
            - title (str): the movie title

        Returns:
            - a list of indices of matching movies

        Hints:
            - You should use self.titles somewhere in this function.
              It might be helpful to explore self.titles in scratch.ipynb
            - You might find one or more of the following helpful:
              re.search, re.findall, re.match, re.escape, re.compile
            - Our solution only takes about 7 lines. If you're using much more than that try to think
              of a more concise approach
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################
        # This only shows indices of movies with an EXACT match for title!
        # TODO: is this the behavior we want?
        title = re.compile(re.escape(title))
        return [i for i, movie in enumerate(self.titles) if title.match(movie[0])]
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    def disambiguate_candidates(self, clarification: str, candidates: list) -> list:
        """Given a list of candidate movies that the user could be
        talking about (represented as indices), and a string given by the user
        as clarification (e.g. in response to your bot saying "Which movie did
        you mean: Titanic (1953) or Titanic (1997)?"), use the clarification to
        narrow down the list and return a smaller list of candidates (hopefully
        just 1!)


        - If the clarification uniquely identifies one of the movies, this
        should return a 1-element list with the index of that movie.
        - If the clarification does not uniquely identify one of the movies, this
        should return multiple elements in the list which the clarification could
        be referring to.

        Example 1 :
          chatbot.disambiguate_candidates("1997", [1359, 2716]) // should return [1359]

          Used in the middle of this sample dialogue
              moviebot> 'Tell me one movie you liked.'
              user> '"Titanic"''
              moviebot> 'Which movie did you mean:  "Titanic (1997)" or "Titanic (1953)"?'
              user> "1997"
              moviebot> 'Ok. You meant "Titanic (1997)"'

        Example 2 :
          chatbot.disambiguate_candidates("1994", [274, 275, 276]) // should return [274, 276]

          Used in the middle of this sample dialogue
              moviebot> 'Tell me one movie you liked.'
              user> '"Three Colors"''
              moviebot> 'Which movie did you mean:  "Three Colors: Red (Trois couleurs: Rouge) (1994)"
                 or "Three Colors: Blue (Trois couleurs: Bleu) (1993)"
                 or "Three Colors: White (Trzy kolory: Bialy) (1994)"?'
              user> "1994"
              movieboth> 'I'm sorry, I still don't understand.
                            Did you mean "Three Colors: Red (Trois couleurs: Rouge) (1994)" or
                            "Three Colors: White (Trzy kolory: Bialy) (1994)" '

        Arguments:
            - clarification (str): user input intended to disambiguate between the given movies
            - candidates (list) : a list of movie indices

        Returns:
            - a list of indices corresponding to the movies identified by the clarification

        Hints:
            - You should use self.titles somewhere in this function
            - You might find one or more of the following helpful:
              re.search, re.findall, re.match, re.escape, re.compile
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################
        # if clarification string in the title, include the title
        return [i for i in candidates if re.search(clarification, self.get_title(i))]
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    ############################################################################
    # 3. Sentiment                                                             #
    ###########################################################################

    def predict_sentiment_rule_based(self, user_input: str) -> int:
        """Predict the sentiment class given a user_input

        In this function you will use a simple rule-based approach to
        predict sentiment.

        Use the sentiment words from data/sentiment.txt which we have already loaded for you in self.sentiment.
        Then count the number of tokens that are in the positive sentiment category (pos_tok_count)
        and negative sentiment category (neg_tok_count)

        This function should return
        -1 (negative sentiment): if neg_tok_count > pos_tok_count
        0 (neural): if neg_tok_count is equal to pos_tok_count
        +1 (postive sentiment): if neg_tok_count < pos_tok_count

        Example:
          sentiment = chatbot.predict_sentiment_rule_based('I LOVE "The Titanic"'))
          print(sentiment) // prints 1

        Arguments:
            - user_input (str) : a user-supplied line of text
        Returns:
            - (int) a numerical value (-1, 0 or 1) for the sentiment of the text

        Hints:
            - Take a look at self.sentiment (e.g. in scratch.ipynb)
            - Remember we want the count of *tokens* not *types*
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################
        tokens = user_input.split()  # For now, we tokenize by whitespace

        counts = Counter(self.sentiment.get(token.lower()) for token in tokens)

        if counts['neg'] > counts['pos']:
            return -1
        elif counts['neg'] < counts['pos']:
            return 1
        return 0
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    def train_logreg_sentiment_classifier(self):
        """
        Trains a bag-of-words Logistic Regression classifier on the Rotten Tomatoes dataset

        You'll have to transform the class labels (y) such that:
            -1 inputted into sklearn corresponds to "rotten" in the dataset
            +1 inputted into sklearn correspond to "fresh" in the dataset

        To run call on the command line:
            python3 chatbot.py --train_logreg_sentiment

        Hints:
            - Review how we used CountVectorizer from sklearn in this code
                https://github.com/cs375williams/hw3-logistic-regression/blob/main/util.py#L193
            - You'll want to lowercase the texts
            - Review how you used sklearn to train a logistic regression classifier for HW 5.
            - Our solution uses less than about 10 lines of code. Your solution might be a bit too complicated.
            - We achieve greater than accuracy 0.7 on the training dataset.
        """
        # load training data
        texts, y = util.load_rotten_tomatoes_dataset()

        # variable name that will eventually be the sklearn Logistic Regression classifier you train
        self.model = sklearn.linear_model.LogisticRegression()
        # variable name will eventually be the CountVectorizer from sklearn
        self.count_vectorizer = CountVectorizer(lowercase=True,
                                                min_df=20,  # only look at words that occur in at least 20 docs
                                                stop_words='english',  # remove english stop words
                                                max_features=1000,  # only select the top 1000 features
                                                )

        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################
        X_train = self.count_vectorizer.fit_transform(texts).toarray()
        Y_train = np.array(y)
        # Transforming y:
        Y_train[Y_train == 'Fresh'], Y_train[Y_train == 'Rotten'] = 1, -1
        Y_train = Y_train.astype('int')

        assert X_train.shape[0] == Y_train.shape[0]
        self.model.fit(X_train, Y_train)
        return self.model.score(X_train, Y_train)  # TODO: Remove after testing
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    def predict_sentiment_statistical(self, user_input: str) -> int:
        """ Uses a trained bag-of-words Logistic Regression classifier to classifier the sentiment

        In this function you'll also uses sklearn's CountVectorizer that has been
        fit on the training data to get bag-of-words representation.

        Example 1:
            sentiment = chatbot.predict_sentiment_statistical('This is great!')
            print(sentiment) // prints 1

        Example 2:
            sentiment = chatbot.predict_sentiment_statistical('This movie is the worst')
            print(sentiment) // prints -1

        Example 3:
            sentiment = chatbot.predict_sentiment_statistical('blah')
            print(sentiment) // prints 0

        Arguments:
            - user_input (str) : a user-supplied line of text
        Returns: int
            -1 if the trained classifier predicts -1
            1 if the trained classifier predicts 1
            0 if the input has no words in the vocabulary of CountVectorizer (a row of 0's)

        Hints:
            - Be sure to lower-case the user input
            - Don't forget about a case for the 0 class!
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################
        X_test = self.count_vectorizer.transform(
            [user_input.lower()]).toarray()
        # If user_input is all 0's, return 0
        if not np.any(X_test):
            return 0

        y_hat = self.model.predict_proba(X_test)
        if y_hat[0][0] > 0.5:
            return -1
        # Else, p_neg <= 0.5 so we return 1
        return 1
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    ############################################################################
    # 4. Movie Recommendation                                                  #
    ############################################################################

    def recommend_movies(self, user_ratings: dict, num_return: int = 3) -> List[str]:
        """
        This function takes user_ratings and returns a list of strings of the
        recommended movie titles.

        Be sure to call util.recommend() which has implemented collaborative
        filtering for you. Collaborative filtering takes ratings from other users
        and makes a recommendation based on the small number of movies the current user has rated.

        This function must have at least 5 ratings to make a recommendation.

        Arguments:
            - user_ratings (dict):
                - keys are indices of movies
                  (corresponding to rows in both data/movies.txt and data/ratings.txt)
                - values are 1, 0, and -1 corresponding to positive, neutral, and
                  negative sentiment respectively
            - num_return (optional, int): The number of movies to recommend

        Example:
            bot_recommends = chatbot.recommend_movies({100: 1, 202: -1, 303: 1, 404:1, 505: 1})
            print(bot_recommends) // prints ['Trick or Treat (1986)', 'Dunston Checks In (1996)',
            'Problem Child (1990)']

        Hints:
            - You should be using self.ratings somewhere in this function
            - It may be helpful to play around with util.recommend() in scratch.ipynb
            to make sure you know what this function is doing.
        """
        ########################################################################
        #                          START OF YOUR CODE                          #
        ########################################################################
        # user_rating_all_movies is a 1-D numpy array of length i for the i total
        # movies in self.ratings. Each index x corresponds to the movie in the
        # x'th row of self.ratings.
        #
        # The array contains 0s for all movies NOT rated by the user according
        # to user_ratings. Otherwise, a given index in the array will contain
        # the binary rating (1 or -1) given by the user for that movie.
        user_rating_all_movies = np.zeros(shape=self.ratings.shape[0])
        # Populating using the ratings given by user
        for i in user_ratings:
            user_rating_all_movies[i] = user_ratings[i]

        recommendations = util.recommend(user_rating_all_movies=user_rating_all_movies,
                                         ratings_matrix=self.ratings,
                                         num_return=num_return)

        return self.find_movies_title_by_idx(recommendations)
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

    ############################################################################
    # 5. Open-ended                                                            #
    ############################################################################

    def function1(self, sentence: str) -> str:
        """
        Function that performs basic spell checking on an input sentence. 
        Function takes in an input sentence and returns a sentence with corrected spelling.
        """
        # Importing packages
        import re
        import nltk

        nltk.download("words", quiet = True)
        nltk.download("wordnet", quiet = True)
        nltk.download("punkt", quiet = True)
    
        from nltk.corpus import words
        from nltk.corpus import wordnet
        from nltk.metrics.distance import edit_distance


        # Obtaining a dictionary of english words that we treat as the set of correctly spelled words
        wordnet_dict = set()
        for synset in wordnet.all_synsets():
            for lemma in synset.lemmas():
                wordnet_dict.add(lemma.name())
        words_dict = set(words.words())
        dictionary = wordnet_dict | words_dict
    
        # Tokenizing all words in input sentence
        # Capitalization is preserved in case word is in beginning of a sentence
        pattern = r'"([^"]+)"|([\w]+)'
        matches = re.findall(pattern, sentence)
        tokens = [(match[1], match[1].lower()) 
              for match in matches if match[1]]
    
        # Obtaining a list of all mispelled words 
        mispelled = [(token, lowercase_token) 
                 for token, lowercase_token in tokens 
                 if lowercase_token not in dictionary]

        # Obtaining closest word to mispelled word
        suggestion = {}
        for mis, lowercase_mis in mispelled:
            suggestion[mis] = []
            closest_dist = float('inf')
            closest_word = None
            word_len = len(lowercase_mis)

            # For every word in dictionary, we compute the closest word to mispelled word in edit distance
            # A penalty for character frequency difference is added
            for word in dictionary:
                if len(word) - 1 <= word_len <= len(word) + 1:
                    distance = nltk.edit_distance(lowercase_mis, word, transpositions=True)
                    if distance + self.char_freq_distance(lowercase_mis, word) <= closest_dist:
                        if mis[0].isupper():
                            closest_word = word.capitalize()
                        else:
                            closest_word = word
                        suggestion[mis] = closest_word
                        closest_dist = distance

        # Correcting all mispelled words in original sentence              
        for k, v in suggestion.items():
            sentence = sentence.replace(k, v)
        return sentence


    def char_freq_distance(self, word_a, word_b) -> int:
        """
        Function computes an integer character frequency distance between two words.
        It is a helper function for function1().
        """
        from collections import Counter
        word_a_freq = Counter(word_a)
        word_b_freq = Counter(word_b)
    
        distance = 0
        for char in set(word_a_freq.keys()) | set(word_b_freq.keys()):
            distance += abs(word_a_freq[char] - word_b_freq[char])
        
        return distance
        

    def function2(self, line: str) -> str:
        """
        Function reads in a user input as string, and prints output according to user sentiment
        In this function we use the sentiment SentimentIntensityAnalyzer class
        in the nltk.sentiment package.

        If user sentiment is neutral, then the output is a neutral statement
        If user input is positve, the output printed to the user is positive, and so forth
        """
        import random
        import nltk
        nltk.download('vader_lexicon', quiet = True)

        # We decided to use the SentimentIntensityAnalyzer from nltk.sentiment because
        # the predict_sentiment_statistical was trained on less data and was not working well.
        # we import this sentiment intensity classifies thta does a much better job.

        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        # method polarity_scores returns a dict that has scores for neutral, negative, positive
        sentiment = sia.polarity_scores(line)
        # if positive
        if sentiment['pos'] > sentiment['neg']:
            # different ways to convey a positive sentiment
            possible_positive_outputs = ["So you liked ", "So you loved ",
            "So it sounds like you enjoyed ",
            "If I'm understanding you correctly, you had a positive experience with ",
            "It seems that you were a fan of ",
            "It appears that you found a lot to like about ",
            "So it's safe to say that you had a good opinion of ",
            "It seems like you were impressed by "]
            random_idx = random.randint(0,7)
            return (possible_positive_outputs[random_idx])
        # if negative
        elif sentiment['neg'] > sentiment['pos'] and sentiment['neg'] > sentiment['neu']:
            # different ways to convey a positive sentiment
            possible_negative_outputs = ["I am sorry you did not like ",
            "So you did not like ", "So it seems that you weren't a fan of ",
            "If I'm understanding you correctly, you didn't enjoy ",
            "It sounds like you had a negative experience with ",
            "From what you've said, it appears that you didn't care for ",
            "So it's safe to say that you didn't find much to like about "]
            random_idx = random.randint(0,6)
            return (possible_negative_outputs[random_idx])
        # if neutral
        else:
            return ("So you entered ")       
        


    def function3():
        """
        Any additional functions beyond two count towards extra credit
        """
        pass


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')
