import torch
import joblib
#score_list = joblib.load('/data/xushicheng/ir_dataset/msmarco/msmarco/score_tok_128_lower')
def pad_ids(input_ids, attention_mask, token_type_ids, max_length, pad_token, mask_padding_with_zero, pad_token_segment_id, pad_on_left=False):
    padding_length = max_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        attention_mask = ([0 if mask_padding_with_zero else 1]
                          * padding_length) + attention_mask
        token_type_ids = ([pad_token_segment_id] *
                          padding_length) + token_type_ids
    else:
        input_ids += [pad_token] * padding_length
        attention_mask += [0 if mask_padding_with_zero else 1] * padding_length
        token_type_ids += [pad_token_segment_id] * padding_length

    return input_ids, attention_mask, token_type_ids


def dual_process_fn(line, i, tokenizer, args):
    features = []
    cells = line.split("\t")
    if len(cells) == 2:
        # this is for training and validation
        # id, passage = line
        mask_padding_with_zero = True
        pad_token_segment_id = 0
        pad_on_left = False

        text = cells[1].strip()
        input_id_a = tokenizer.encode(
            text, add_special_tokens=True, max_length=args.max_seq_length,truncation=True)
        token_type_ids_a = [0] * len(input_id_a)
        attention_mask_a = [
            1 if mask_padding_with_zero else 0] * len(input_id_a)
        input_id_a, attention_mask_a, token_type_ids_a = pad_ids(
            input_id_a, attention_mask_a, token_type_ids_a, args.max_seq_length, tokenizer.pad_token_id, mask_padding_with_zero, pad_token_segment_id, pad_on_left)
        features += [torch.tensor(input_id_a, dtype=torch.int), torch.tensor(
            attention_mask_a, dtype=torch.bool), torch.tensor(token_type_ids_a, dtype=torch.uint8)]
        qid = int(cells[0])
        features.append(qid)
    else:
        raise Exception(
            "Line doesn't have correct length: {0}. Expected 2.".format(str(len(cells))))
    return [features]


def triple_process_fn(line, i, tokenizer, args):
    features = []
    cells = line.split("\t")
    if len(cells) == 3:
        # this is for training and validation
        # query, positive_passage, negative_passage = line
        mask_padding_with_zero = True
        pad_token_segment_id = 0
        pad_on_left = False

        for text in cells:
            input_id_a = tokenizer.encode(
                text.strip(), add_special_tokens=True, max_length=args.max_seq_length,truncation=True)
            token_type_ids_a = [0] * len(input_id_a)
            attention_mask_a = [
                1 if mask_padding_with_zero else 0] * len(input_id_a)
            input_id_a, attention_mask_a, token_type_ids_a = pad_ids(
                input_id_a, attention_mask_a, token_type_ids_a, args.max_seq_length, tokenizer.pad_token_id, mask_padding_with_zero, pad_token_segment_id, pad_on_left)
            features += [torch.tensor(input_id_a, dtype=torch.int),
                         torch.tensor(attention_mask_a, dtype=torch.bool)]

    else:
        raise Exception(
            "Line doesn't have correct length: {0}. Expected 3.".format(str(len(cells))))
    return [features]

def four_process_fn(line, i, tokenizer, args):
    features = []
    cells = line.split("\t")
    if len(cells) == 4:
        # this is for training and validation
        # query, positive_passage, negative_passage = line
        mask_padding_with_zero = True
        pad_token_segment_id = 0
        pad_on_left = False
        id = cells[-1]
        cells = cells[:-1]
        features += [torch.tensor(int(id), dtype=torch.int)]
        # idx = 0
        for text in cells:
            input_id_a = tokenizer.encode(
                text.lower().strip(), add_special_tokens=True, max_length=args.max_seq_length,truncation=True)
            token_type_ids_a = [0] * len(input_id_a)
            attention_mask_a = [
                1 if mask_padding_with_zero else 0] * len(input_id_a)
            input_id_a, attention_mask_a, token_type_ids_a = pad_ids(
                input_id_a, attention_mask_a, token_type_ids_a, args.max_seq_length, tokenizer.pad_token_id, mask_padding_with_zero, pad_token_segment_id, pad_on_left)
            features += [torch.tensor(input_id_a, dtype=torch.int),
                         torch.tensor(attention_mask_a, dtype=torch.bool)]
            # idx += 1

    else:
        raise Exception(
            "Line doesn't have correct length: {0}. Expected 3.".format(str(len(cells))))
    return [features]


def four_process_fn_add_cross_encoder(line, i, tokenizer, tokenizer_bert,args):
    features = []
    cells = line.split("\t")
    if len(cells) == 4: #q,pos,neg,id
        # this is for training and validation
        # query, positive_passage, negative_passage = line
        mask_padding_with_zero = True
        pad_token_segment_id = 0
        pad_on_left = False
        id = cells[-1]
        cells = cells[:-1]
        features += [torch.tensor(int(id), dtype=torch.int)]
        # idx = 0
        for text in cells:
            input_id_a = tokenizer.encode(
                text.lower().strip(), add_special_tokens=True, max_length=args.max_seq_length,truncation=True)
            token_type_ids_a = [0] * len(input_id_a)
            attention_mask_a = [
                1 if mask_padding_with_zero else 0] * len(input_id_a)
            input_id_a, attention_mask_a, token_type_ids_a = pad_ids(
                input_id_a, attention_mask_a, token_type_ids_a, args.max_seq_length, tokenizer.pad_token_id, mask_padding_with_zero, pad_token_segment_id, pad_on_left)
            features += [torch.tensor(input_id_a, dtype=torch.int),
                         torch.tensor(attention_mask_a, dtype=torch.bool)]
            # idx += 1
        input_bert_pos = tokenizer_bert(cells[0].lower().strip(), cells[1].lower().strip(),  max_length=256,pad_to_max_length=True,truncation=True, return_tensors="pt")
        input_bert_neg = tokenizer_bert(cells[0].lower().strip(), cells[2].lower().strip(),  max_length=256,pad_to_max_length=True,truncation=True, return_tensors="pt")
        features += [input_bert_pos['input_ids'][0],
                     input_bert_pos['attention_mask'][0],
                     input_bert_pos['token_type_ids'][0],
                     input_bert_neg['input_ids'][0],
                     input_bert_neg['attention_mask'][0],
                     input_bert_neg['token_type_ids'][0],
                     ]
    else:
        raise Exception(
            "Line doesn't have correct length: {0}. Expected 3.".format(str(len(cells))))
    return [features]

def four_process_fn_add_reader(line, i, tokenizer, tokenizer_reader,args):
    features = []
    cells = line.split("\t")
    if len(cells) == 4:
        # this is for training and validation
        # query, positive_passage, negative_passage = line
        mask_padding_with_zero = True
        pad_token_segment_id = 0
        pad_on_left = False
        id = cells[-1]
        cells = cells[:-1]
        #features += [torch.tensor(int(id), dtype=torch.int)]
        # idx = 0
        for text in cells:
            input_id_a = tokenizer.encode(
                text.lower().strip(), add_special_tokens=True, max_length=args.max_seq_length,truncation=True)
            token_type_ids_a = [0] * len(input_id_a)
            attention_mask_a = [
                1 if mask_padding_with_zero else 0] * len(input_id_a)
            input_id_a, attention_mask_a, token_type_ids_a = pad_ids(
                input_id_a, attention_mask_a, token_type_ids_a, args.max_seq_length, tokenizer.pad_token_id, mask_padding_with_zero, pad_token_segment_id, pad_on_left)
            features += [torch.tensor(input_id_a, dtype=torch.int),
                         torch.tensor(attention_mask_a, dtype=torch.bool)]
            # idx += 1
        input_id_q = tokenizer_reader.encode(
                cells[0].lower().strip(), add_special_tokens=True, max_length=args.max_seq_length,truncation=True)
        input_id_a = tokenizer_reader.encode(
                cells[1].lower().strip(), add_special_tokens=True, max_length=args.max_seq_length,truncation=True)
        input_id_a = input_id_a[1:]
        max_len = 256
        if len(input_id_a) + len(input_id_q) > max_len:
            pos_origin = input_id_a[:max_len-len(input_id_q)]
        else:
            pos_origin = input_id_a
        pad_len = max(0, max_len - (len(input_id_q) + len(pos_origin)))
        attention_mask_reader = [1] * (len(input_id_q) + len(pos_origin)) + [0] * pad_len
        input_id_reader = input_id_q+pos_origin + [tokenizer.pad_token_id] * pad_len
        input_id_reader = torch.tensor(input_id_reader, dtype=torch.int)
        pad_len_pos_origin = max(0, max_len - (len(pos_origin)))
        input_id_pos_origin = pos_origin + [tokenizer.pad_token_id] * pad_len_pos_origin
        input_id_pos_origin = torch.tensor(input_id_pos_origin, dtype=torch.int)
        attention_mask_reader = torch.tensor(attention_mask_reader, dtype=torch.bool)
        features += [input_id_reader,attention_mask_reader,input_id_pos_origin]
        features += [torch.tensor(len(input_id_q), dtype=torch.int)]
    else:
        raise Exception(
            "Line doesn't have correct length: {0}. Expected 3.".format(str(len(cells))))
    return [features]

def four_process_fn_add_reader_combine(line, i, tokenizer, tokenizer_reader,args):
    features = []
    cells = line.split("\t")
    if len(cells) == 4:
        # this is for training and validation
        # query, positive_passage, negative_passage = line
        mask_padding_with_zero = True
        pad_token_segment_id = 0
        pad_on_left = False
        id = cells[-1]
        cells = cells[:-1]
        features += [torch.tensor(int(id), dtype=torch.int)]
        # idx = 0
        for text in cells:
            input_id_a = tokenizer.encode(
                text.lower().strip(), add_special_tokens=True, max_length=args.max_seq_length,truncation=True)
            token_type_ids_a = [0] * len(input_id_a)
            attention_mask_a = [
                1 if mask_padding_with_zero else 0] * len(input_id_a)
            input_id_a, attention_mask_a, token_type_ids_a = pad_ids(
                input_id_a, attention_mask_a, token_type_ids_a, args.max_seq_length, tokenizer.pad_token_id, mask_padding_with_zero, pad_token_segment_id, pad_on_left)
            features += [torch.tensor(input_id_a, dtype=torch.int),
                         torch.tensor(attention_mask_a, dtype=torch.bool)]
            # idx += 1
        input_id_q = tokenizer_reader.encode(
                cells[0].lower().strip(), add_special_tokens=True, max_length=args.max_seq_length,truncation=True)
        input_id_a = tokenizer_reader.encode(
                cells[1].lower().strip(), add_special_tokens=True, max_length=args.max_seq_length,truncation=True)
        input_id_a = input_id_a[1:]
        max_len = 256
        if len(input_id_a) + len(input_id_q) > max_len:
            pos_origin = input_id_a[:max_len-len(input_id_q)]
        else:
            pos_origin = input_id_a
        pad_len = max(0, max_len - (len(input_id_q) + len(pos_origin)))
        attention_mask_reader = [1] * (len(input_id_q) + len(pos_origin)) + [0] * pad_len
        input_id_reader = input_id_q+pos_origin + [tokenizer.pad_token_id] * pad_len
        input_id_reader = torch.tensor(input_id_reader, dtype=torch.int)
        pad_len_pos_origin = max(0, max_len - (len(pos_origin)))
        input_id_pos_origin = pos_origin + [tokenizer.pad_token_id] * pad_len_pos_origin
        input_id_pos_origin = torch.tensor(input_id_pos_origin, dtype=torch.int)
        attention_mask_reader = torch.tensor(attention_mask_reader, dtype=torch.bool)
        features += [input_id_reader,attention_mask_reader,input_id_pos_origin]
        features += [torch.tensor(len(input_id_q), dtype=torch.int)]
    else:
        raise Exception(
            "Line doesn't have correct length: {0}. Expected 3.".format(str(len(cells))))
    return [features]


def triple2dual_process_fn(line, i, tokenizer, args):
    ret = []
    cells = line.split("\t")
    if len(cells) == 3:
        # this is for training and validation
        # query, positive_passage, negative_passage = line
        # return 2 entries per line, 1 pos + 1 neg
        mask_padding_with_zero = True
        pad_token_segment_id = 0
        pad_on_left = False
        pos_feats = []
        neg_feats = []

        for i, text in enumerate(cells):
            input_id_a = tokenizer.encode(
                text.strip(), add_special_tokens=True, max_length=args.max_seq_length,)
            token_type_ids_a = [0] * len(input_id_a)
            attention_mask_a = [
                1 if mask_padding_with_zero else 0] * len(input_id_a)
            input_id_a, attention_mask_a, token_type_ids_a = pad_ids(
                input_id_a, attention_mask_a, token_type_ids_a, args.max_seq_length, tokenizer.pad_token_id, mask_padding_with_zero, pad_token_segment_id, pad_on_left)
            if i == 0:
                pos_feats += [torch.tensor(input_id_a, dtype=torch.int),
                              torch.tensor(attention_mask_a, dtype=torch.bool)]
                neg_feats += [torch.tensor(input_id_a, dtype=torch.int),
                              torch.tensor(attention_mask_a, dtype=torch.bool)]
            elif i == 1:
                pos_feats += [torch.tensor(input_id_a, dtype=torch.int),
                              torch.tensor(attention_mask_a, dtype=torch.bool), 1]
            else:
                neg_feats += [torch.tensor(input_id_a, dtype=torch.int),
                              torch.tensor(attention_mask_a, dtype=torch.bool), 0]
        ret = [pos_feats, neg_feats]
    else:
        raise Exception(
            "Line doesn't have correct length: {0}. Expected 3.".format(str(len(cells))))
    return ret

