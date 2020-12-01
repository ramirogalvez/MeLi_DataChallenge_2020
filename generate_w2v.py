import json
import tqdm
import gc
import logging
import gensim


def load_history_data(data_path):

    print("Loading history data")

    with open(data_path) as f:
        file_data = f.read().split("\n")

    out = []
    for l in tqdm.tqdm(file_data):
        l = json.loads(l)
        out.append(l)

    gc.collect()
    return(out)


def gen_user_sentence(user_history):

    user_sentence = []
    for e in user_history:
        if e["event_type"] == "view":
            user_sentence.append(str(e["event_info"]))

    return user_sentence


def create_corpus_items_w2v(train_dataset):
    print("Creating p2v corpus")
    corpus = []
    item_bought = []
    for u in tqdm.tqdm(train_dataset):
        corpus.append(gen_user_sentence(u["user_history"]))

    return corpus


def train_w2v(data_to_train, w2v_path):

    corpus = create_corpus_items_w2v(data_to_train)        

    logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                        datefmt= '%H:%M:%S', level=logging.INFO)

    model = gensim.models.Word2Vec(size=400,
                                   window=7,
                                   ns_exponent=-0.5,
                                   min_count=3,
                                   negative=5,
                                   sample=10**-5,
                                   workers=8,
                                   sg=1)

    model.build_vocab(corpus, progress_per=30000)

    model.train(corpus,
                total_examples=model.corpus_count,
                epochs=100,
                report_delay=1)

    model.save(w2v_path)


def generate_item_embeddings(train_data_path, eval_data_path, w2v_path):

    train_data = load_history_data(train_data_path)
    eval_data = load_history_data(eval_data_path)
    train_data.extend(eval_data)
    train_w2v(train_data, w2v_path)
