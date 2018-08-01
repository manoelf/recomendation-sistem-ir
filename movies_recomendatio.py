from surprise import Dataset
from surprise import KNNWithMeans 
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import cross_validate, train_test_split
import pandas as pd

'''
    As we used movieslen data so we downloaded it and read below
'''
file_path = './data/ml-100k/ml-100k/u.data'
reader = Reader(line_format='user item rating timestamp', sep='\t')

data = pd.read_table('./data/ml-100k/ml-100k/u.data')
data.columns = ['user', 'movie', 'rating', 'time']
movies_data = open('./data/ml-100k/ml-100k/u.item')#, encoding="ISO-8859-1")

def map_movie_id_name(movies_data):
    '''
        Mapping the movies id to movies name
        
        ARGS:
            All movies names and ids
        RETURN:
            Dictionary with key:id and value: movie name
    '''
    dict_movies = dict()
    for line in movies_data:
        item = line.split('|')
        movie_id = item[0]
        movie_name = item[1]
        dict_movies[movie_id] = movie_name
    return dict_movies


dict_movies = map_movie_id_name(movies_data)

def get_all_movies(data):
    '''
        Putting all movies in a set
        
        ARGS:
            All data which is users and movies, also the rating
        RETURN:
            Set: A set with all movies Ids
    '''
    all_movies = set()
    for i in range(data.shape[0]):
        movie = data.loc[i, 'movie']
        all_movies.add(movie)
    return all_movies

def map_user_movies(data):
    '''
        Getting all watched movies by the user
        
        ARGS:
            All data which is users and movies, also the rating
        RETURN:
            Dict: A dictionary with all users and their watched movies (key =  userId and values = List of watched movies)
    '''
    dict_user_movies = dict()    
    for i in range( data.shape[0]):
        user = data.loc[i, 'user']
        movie = data.loc[i, 'movie']
        if (user in dict_user_movies):
            dict_user_movies[user].add(movie)
        else:
            dict_user_movies[user] = set()
            dict_user_movies[user].add(movie)
    return dict_user_movies
    

def map_unwatched(data, all_movies):
    '''
        Mapping the movies that each user didn't watch
        
        ARGS:
            data: All data which is users and movies, also the rating
            all_movies: The set of all movies 
        RETURN:
            Dict: A dictionary with all users and their not watched movies (key =  userId and values = List of not watched movies)
    '''
    dict_unwatched = dict()
    for key in dict_user_movies.keys():
        dict_unwatched[key] = all_movies - dict_user_movies[key]
    return dict_unwatched

def get_unwatched(user, dict_unwatched):
    return dict_unwatched[user]


def predict_to_user(user, unwatcheds, algo):
    '''
        Here is very important. Doing the prediction to the movies by users.
        So basically, get a movie unwatched by a specific user and makes the prediction
        which is what rating that user should.
        
        ARGS:
            user: user to make the prediction to its movies
            unwatcheds: movies did not watch by user
            algo: algorithm used to predict the rating (Here specifically we use KNN, but it could change)
        RETURN:
            List: list with the movies not watched by the user and its predicted rating (tupe: rating, movie_id)
    '''
    movies_rating = []
    for movie in unwatcheds:
        pred_rating = algo.predict(uid=str(user), iid=str(movie)).est
        movies_rating.append((pred_rating, movie))
    return movies_rating


def top5_recomendation(user, algo, dict_unwatched):
    '''
        Getting the movies which had the greater rating predicted by the algoritm
        
        ARGS:
            user: user to make the prediction to its movies
            algo: algorithm used to predict the rating (Here specifically we use KNN, but it could change)
            dict_unwatcheds: dictionary of user and their movies  not watched
        RETURN:
            List: the top 5 most indicate movie to the user
    '''
    unwatcheds = dict_unwatched[user]
    result = predict_to_user(user, unwatcheds, algo)
    result.sort()
    return result[-5:]


def  show_top_3_neighborsshow_top(uid):
    '''
        Getting the most close neighbor of user
        
        ARGS:
            uid: user id get its neighbor
           
        RETURN:
            List: the top 5 most close neighbor of the user
    '''
    inner_uid = algo.trainset.to_inner_uid(str(uid))
    closest_neighbors = algo.get_neighbors(iid=inner_uid, k=3)
    return closest_neighbors

dict_user_movies = map_user_movies(data)
all_movies = get_all_movies(data)
dict_unwatched = map_unwatched(data, all_movies)

#Loading the dataset
dataset = Dataset.load_from_file(file_path, reader=reader)

# Spliting data in train (85%) and test (15%)
trainset, testset = train_test_split(dataset, test_size=.15)

'''
    The algorithm used to make the predictions (Recomendation in generally) was KNN, which had shown 
    the best RMSE, so it suggest a smaller error
'''
algo = KNNWithMeans(k=40, sim_options={'name': 'cosine', 'user_based': True})

# Training the model
algo.fit(trainset)

def get_accuracy():
    '''
        This module provides with tools for computing accuracy metrics on a set of predictions.
                 
        RETURN:
            Floar: the accyracy for the actual module
    '''
    predictions_knn = algo.test(testset)
    knn_rmse = accuracy.rmse(predictions_knn, verbose=False)
    return float("{0:.2f}".format(knn_rmse))

def get_top5_by_array(user):
    '''
        Getting top 5 movies for the user
        
        ARGS:
            user: user for get its top5
           
        RETURN:
            List: the top 5 most indicated movie
    '''
    result = []
    top5 = top5_recomendation(user, algo, dict_unwatched)
    for movie in top5:
        result.append(dict_movies[str(movie[1])])
    return result

def get_already_watched(user):
    '''
        Getting the movies already watched by user
        
        ARGS:
            user: user id get its watched movies
           
        RETURN:
            List: all wathced movies
    '''
    result = []
    for i in dict_user_movies[user]:
        result.append(dict_movies[str(i)])
    return result




