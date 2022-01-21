import pickle


def save_to_pickle(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_from_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

