import pickle
import json
import re
import tqdm
import gc
from collections import Counter
from math import log
import numpy as np
import gc
import bottleneck
import gensim
from generate_w2v import generate_item_embeddings
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


ITEM_DATA_PATH = "./data/input/item_data.jl"
TRAIN_DATA_PATH = "./data/input/train_dataset.jl"
EVAL_DATA_PATH = "./data/input/test_dataset.jl"
W2V_PATH = "./data/embeddings/meli_w2v.model"
PREDS_PATH = "./predictions/final_preds.csv"

def load_item_data(data_path):

    print("Loading item data")

    with open(data_path) as f:
        file_data = f.read().split("\n")

    item_data = {}
    for l in tqdm.tqdm(file_data):
        l = json.loads(l)
        item_id = l.pop("item_id")
        item_data[item_id] = l
        if item_data[item_id]["price"]:
            item_data[item_id]["price"] = float(item_data[item_id]["price"])

    return(item_data)


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


def get_item_popularity(train_data):

    print("Getting item popularity")

    pop_in_domain = {}
    item_popularity = Counter()
    domain_pop = Counter()

    for i in tqdm.tqdm(train_data):
        domain_id = item_data[i["item_bought"]]["domain_id"]
        item_popularity[i["item_bought"]] += 1
        domain_pop[domain_id] += 1
        if domain_id not in pop_in_domain:
            pop_in_domain[domain_id] = Counter()
        pop_in_domain[domain_id][i["item_bought"]] += 1

    for d in pop_in_domain:
        pop_in_domain[d] = [e[0] for e in sorted(pop_in_domain[d].items(), key = lambda x: -x[1])]

    pop_general = [e[0] for e in (sorted(item_popularity.items(), key=lambda x: -x[1]))]

    default_preds = sorted([e for e in domain_pop.items()], key=lambda x: -x[1])
    default_preds = [pop_in_domain[e[0]][0] for e in default_preds]

    return pop_in_domain, pop_general, default_preds


def predict_baseline(data_to_predict, pop_in_domain):

    print("Getting baseline preds")

    base_looked = []
    base_domain = []

    for u in tqdm.tqdm(data_to_predict):

        h = u["user_history"]
        items = [e for e in h if e["event_type"] == "view"]
        items = [e["event_info"] for e in sorted(items, key=lambda x: x['event_timestamp'], reverse=True)]

        items_front = items.copy()
        items_back = []

        if items:
            domains_count = Counter([item_data[e]["domain_id"] for e in items])
            domains_order = _remove_dup([item_data[e]["domain_id"] for e in items])
            domains_order = [(i, e, domains_count[e]) for (i, e) in enumerate(domains_order)]
            domains_order = sorted(domains_order, key=lambda x: (-x[2], x[0]))

            while domains_order:
                _, top_domain, _ = domains_order.pop(0)

                if top_domain in pop_in_domain:
                    items_back.extend(pop_in_domain[top_domain])

        base_looked.append(items_front)
        base_domain.append(items_back)

    gc.collect()

    return base_looked, base_domain


def _create_corpus_items(users_data):

    corpus = []
    item_bought = []
    for u in users_data:
        h = u["user_history"]
        items = [str(e["event_info"]) for e in h if e["event_type"] == "view"]
        corpus.append(items)
        item_bought.append(u["item_bought"] if "item_bought" in u else np.nan)
    return corpus, np.array(item_bought)


def _create_corpus_search(users_data):

    corpus = []
    item_bought = []
    for u in users_data:
        h = u["user_history"]
        items = [e["event_info"].lower() for e in h if e["event_type"] == "search"]
        if items:
            tokens = [e2 for e1 in items for e2 in e1.split()]
        else:
            tokens = []
        corpus.append(tokens)
        item_bought.append(u["item_bought"] if "item_bought" in u else np.nan)
    return corpus, np.array(item_bought)


def _gen_intervals(n_rows, length = 10000):

    max_val = 0
    intervals = []
    e = 0
    while max_val <= n_rows+1:
        upper = length * e + length
        intervals.append((length * e, upper))
        max_val = upper
        e += 1

    return intervals


def bow_preds(to_train_data, to_pred_data, item_or_search, to_pred):

    print(f"Generating bow preds - {item_or_search}")

    if item_or_search == "item":
        train_corpus, train_bought = _create_corpus_items(to_train_data)
        to_pred_corpus, _ = _create_corpus_items(to_pred_data)
        min_df = 1
        max_ngram = 4
    elif item_or_search == "search":
        train_corpus, train_bought = _create_corpus_search(to_train_data)
        to_pred_corpus, _ = _create_corpus_search(to_pred_data)
        min_df = 5
        max_ngram = 4

    cv = CountVectorizer(tokenizer=lambda x: x,
                         preprocessor=lambda x: x,
                         min_df=min_df,
                         ngram_range=(1, max_ngram))

    tdm_train = cv.fit_transform(train_corpus)
    del train_corpus
    tdm_to_pred = cv.transform(to_pred_corpus)
    del to_pred_corpus, cv

    tfidf = TfidfTransformer(sublinear_tf=True)
    tdm_train = tfidf.fit_transform(tdm_train).T
    tdm_to_pred = tfidf.transform(tdm_to_pred)

    intervals = _gen_intervals(tdm_to_pred.shape[0])

    predictions = []
    for l, u in tqdm.tqdm(intervals):
        cos_sim = np.dot(tdm_to_pred[l:u,:], tdm_train)

        for j in range(cos_sim.shape[0]):
            row_sim = cos_sim[j,:].tocoo()
            data_sim = row_sim.data
            closest = np.argsort(-data_sim)
            closest = row_sim.col[closest]
            pr_tmp = train_bought[closest]
            predictions.append(list(pr_tmp[:to_pred]))

    return predictions


def _remove_dup(items):

    items_no_dup = []
    added = set([])
    for e in items:
        if e not in added:
            items_no_dup.append(e)
            added.add(e)
    return items_no_dup


def complete_preds(predictions, default_preds, item_data):

    completed = []
    for p_orig in predictions:
        i = 0
        p = p_orig.copy()
        while len(p) < 10:
            set_p = set(p)
            if default_preds[i] not in set_p:
                pred_domains = set([item_data[e]["domain_id"]  for e in p])
                if item_data[default_preds[i]]["domain_id"] not in pred_domains:
                    p.append(default_preds[i])
            i += 1
        completed.append(p)

    return completed


def _remove_used(predictions):

    no_used_preds = []
    for p in predictions:
        no_used_preds.append([e for e in p if item_data[e]["condition"] == "new"])
    return no_used_preds


def pred_w2v(data_to_pred, n_preds):

    print("Generating p2v preds")

    model = gensim.models.Word2Vec.load(W2V_PATH)

    emb_to_pred = []
    to_ignore = set([])
    for i, u in enumerate(data_to_pred):
        views = u["user_history"]
        views = [e["event_info"] for e in views if e["event_type"] == "view"]
        views = [str(e) for e in views]
        views = [e for e in views if e in model.wv]
        if views:
            emb_to_pred.append(np.mean([model.wv.__getitem__([e]) for e in views], axis=0).squeeze())
        else:
            emb_to_pred.append(np.array([0.0 for e in range(400)]))
            to_ignore.add(i)

    emb_to_pred = np.vstack(emb_to_pred)

    intervals = _gen_intervals(len(data_to_pred), length = 1000)
    vocabulary = list(model.wv.vocab.keys())
    X = model.wv.__getitem__(vocabulary).T
    voc_rec = np.array([int(e) for e in vocabulary])

    predictions = []
    i = 0
    for l, u in tqdm.tqdm(intervals):

        cos_sim = -np.dot(emb_to_pred[l:u,:], X)
        for j in range(cos_sim.shape[0]):

            if i in to_ignore:
                predictions.append([])
            else:
                closest = bottleneck.argpartition(cos_sim[j,:], n_preds)[:n_preds]
                preds_row = [e for e in voc_rec[closest]]
                row_sim = cos_sim[j, closest]
                preds_row = [preds_row[e] for e in row_sim.argsort()]
                predictions.append(preds_row)
            i += 1

        del cos_sim

    return predictions


def make_submit_file(predictions, filename):

    predictions = [",".join([str(i) for i in e]) + "\n" for e in predictions]
    predictions = "".join(predictions)
    with open(filename, "w") as f:
        f.write(predictions)


def make_preds_vote(base_items, base_domain, bow_item, bow_search, w2v_item):
    
    print("Making differents algorithms vote")    

    all_preds = zip(base_items, base_domain, bow_item, bow_search, w2v_item)

    merged = []

    for bs_look, bs_dom, bow_it, bow_s, w2v_it in tqdm.tqdm(all_preds):

        set_bs_dom = set(bs_dom)
        bs_look = [e for e in bs_look if e in set_bs_dom]
        bow_it = [e for e in bow_it if e in set_bs_dom]
        w2v_it = [e for e in w2v_it if e in set_bs_dom]
        if set_bs_dom: w2v_it = [e for e in w2v_it if e in set_bs_dom]

        freqs = Counter()
        for i, e in enumerate(bs_dom):
            freqs[e] += 0.625
        for i, e in enumerate(bs_look):
            freqs[e] += 2.325 / log(i + 1.55, 1.55)
        for i, e in enumerate(w2v_it):
            freqs[e] += 1.17 / log(i + 2.15125, 2.15125)
        for i, e in enumerate(bow_it):
            freqs[e] += 0.715 / log(i + 2.05, 2.05)
        for i, e in enumerate(bow_s):
            freqs[e] += 0.5975 / log(i + 2, 2)

        bs_look = _remove_dup(bs_look)
        bs_dom = _remove_dup(bs_dom)
        bow_it = _remove_dup(bow_it)
        bow_s = _remove_dup(bow_s)

        all_items = []

        set_all_items = set(all_items)
        all_items.extend([e for e in bs_look if e not in set_all_items])

        set_all_items = set(all_items)
        all_items.extend([e for e in w2v_it if e not in set_all_items])

        set_all_items = set(all_items)
        all_items.extend([e for e in bs_dom if e not in set_all_items])

        set_all_items = set(all_items)
        all_items.extend([e for e in bow_s if e not in set_all_items])

        set_all_items = set(all_items)
        all_items.extend([e for e in bow_it if e not in set_all_items])

        all_items = [(e1, e2, freqs[e2]) for (e1, e2) in enumerate(all_items)]
        all_items = sorted(all_items, key=lambda x: (-x[2], x[0]))
        all_items = [e[1] for e in all_items]
        all_items = [e for e in all_items if item_data[e]["domain_id"]]
        all_items = all_items[:10]

        assert len(set(all_items)) == len(all_items)
        merged.append(all_items)

    return merged


def make_final_preds(train_data_path, eval_data_path,
                     item_data_path, preds_path):

    item_data = load_item_data(item_data_path)
    data_to_pred = load_history_data(eval_data_path)
    gc.collect()

    # Make word2vec predictions
    w2v_item_preds = pred_w2v(data_to_pred, 70)

    # Make baseline predictions
    pop_in_domain, pop_general, default_preds = get_item_popularity(train_data)
    train_data = load_history_data(train_data_path)
    base_looked, base_domain = predict_baseline(data_to_pred, pop_in_domain, pop_general)
    gc.collect()

    # Make tf-idf preds with items
    bow_item_predictions = bow_preds(train_data, data_to_pred, "item", 110)
    bow_item_predictions = _remove_used(bow_item_predictions)
    gc.collect()

    # Make tf-idf preds with searches
    bow_search_predictions = bow_preds(train_data, data_to_pred, "search", 150)
    bow_search_predictions = _remove_used(bow_search_predictions)
    gc.collect()

    # Delete unnecesary data
    del train_data, data_to_pred
    gc.collect()

    # Make differents predictions vote
    final_preds = make_preds_vote(base_looked,
                                  base_domain,
                                  [e[:103] for e in bow_item_predictions],
                                  [e[:144] for e in bow_search_predictions],
                                  [e[:65] for e in w2v_item_preds])

    # Fill non-length 10 predictions
    final_preds = complete_preds(final_preds, default_preds, item_data)

    # Build the predictions csv file
    make_submit_file(final_preds, preds_path)


if __name__ == "__main__":

    generate_item_embeddings(TRAIN_DATA_PATH,
                             EVAL_DATA_PATH,
                             W2V_PATH)

    make_final_preds(TRAIN_DATA_PATH,
                     EVAL_DATA_PATH,
                     ITEM_DATA_PATH,
                     PREDS_PATH)
