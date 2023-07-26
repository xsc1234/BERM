import csv
from transformers import BertTokenizer
import joblib
import tqdm
import math
zimu = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
def compute_idf_wiki(path):
    word_idf_dic = {}
    word_tf = {}
    toknieze = BertTokenizer.from_pretrained(bert_model)
    with open(path, encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter="\t")
        doc_all_count = 0
        lines = [line for line in reader]
        for row in tqdm.tqdm(lines[1:]):
            text = row[0]
            doc_all_count = doc_all_count + 2
            text = text.lower().strip()
            tok_doc = toknieze.encode(text, add_special_tokens=True, max_length=256, truncation=True)
            tok_doc = toknieze.convert_ids_to_tokens(tok_doc)
            passage_set = set(tok_doc)

            for word in passage_set:
                if word in word_idf_dic.keys():
                    word_idf_dic[word] += 1
                else:
                    word_idf_dic[word] = 1

            text = row[1]
            text = text.lower().strip()
            tok_doc = toknieze.encode(text, add_special_tokens=True, max_length=256, truncation=True)
            tok_doc = toknieze.convert_ids_to_tokens(tok_doc)
            passage_set = set(tok_doc)

            for word in passage_set:
                if word in word_idf_dic.keys():
                    word_idf_dic[word] += 1
                else:
                    word_idf_dic[word] = 1

        for word in word_idf_dic.keys():
            word_tf[word] = word_idf_dic[word]
            word_idf_dic[word] = math.log(doc_all_count / (word_idf_dic[word]+1))

    joblib.dump(word_tf, '{}/word_tf_tok_256_bert'.format(marco_path))
    joblib.dump(word_idf_dic,'{}/word_idf_tok_256_bert'.format(marco_path))
    return word_tf,word_idf_dic

def compute_bm25(query,doc_stem):
    k1 = 1.5
    b = 0.75
    bm_25_score = 0
    for q_stem in query:
        if q_stem in doc_stem:
            try:
                bm_25_score += dic_idf[q_stem]*(k1+1) / k1*(1-b+b*len(doc_stem)/100)
            except:
                bm_25_score += 15*(k1+1) / k1*(1-b+b*len(doc_stem)/100)
                print(q_stem)
    return bm_25_score

def load_marco_triples(path):
    toknieze = BertTokenizer.from_pretrained(bert_model)
    with open(path, encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter="\t")
        score = []
        lines = [line for line in reader]
        count = 0
        for row in tqdm.tqdm(lines[1:]):
            count += 1
            query = row[0]
            doc = row[1]
            query = query.lower().strip()
            doc = doc.lower().strip()
            tok_doc = toknieze.encode(doc,add_special_tokens=True, max_length=128,truncation=True)
            tok_doc = toknieze.convert_ids_to_tokens(tok_doc)
            row_score_dist = []
            tok_query = toknieze.encode(query,add_special_tokens=True, max_length=128,truncation=True)
            tok_query = toknieze.convert_ids_to_tokens(tok_query)
            sentence = []
            for tok in tok_doc:
                if not tok == '.':
                    sentence.append(tok)
                elif not sentence == []:
                    bm25 = compute_bm25(tok_query, sentence)
                    row_score_dist.append(bm25)
                    sentence = []
            score.append(row_score_dist)
        joblib.dump(score,'{}/score_tok_128_lower_bert'.format(marco_path)) #list id->score_list


if __name__ == '__main__':
    bert_model = ''
    marco_path = ''
    _,dic_idf = compute_idf_wiki('{}/triples.train.small.tsv'.format(marco_path))
    load_marco_triples(path='{}/triples.train.small.tsv'.format(marco_path))