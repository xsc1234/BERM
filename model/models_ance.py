import sys
sys.path += ['../']
import torch
from torch import nn
from transformers import (
    RobertaConfig,
    RobertaModel,
    RobertaForSequenceClassification,
    RobertaForQuestionAnswering,
    RobertaTokenizer,
    BertModel,
    BertTokenizer,
    BertForSequenceClassification,
    BertConfig
)
import torch.nn.functional as F
from data.process_fn import triple_process_fn, triple2dual_process_fn,four_process_fn,four_process_fn_add_reader
#from model.SEED_Encoder import SEEDEncoderConfig, SEEDTokenizer, SEEDEncoderForSequenceClassification,SEEDEncoderForMaskedLM
import joblib
import os
device = "cuda"
marco_path = ''
dic_idf = joblib.load('{}/word_idf_tok_512'.format(marco_path))
<<<<<<< HEAD
=======
## ANCE
>>>>>>> f2c2f4f9d6258c7d0596c526c530f485e3b909d7
# from transformers import DPRReader, DPRReaderTokenizer
# tokenizer = DPRReaderTokenizer.from_pretrained("/data/home/xushicheng/reader_multi")
# reader = DPRReader.from_pretrained("/data/home/xushicheng/reader_multi")
# reader = reader.to(device)
## add reader
from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta")

def compute_bm25(query,doc_stem):
    k1 = 1.5
    b = 0.75
    bm_25_score = 0
    for q_stem in query:
        if q_stem in doc_stem:
            try:
                bm_25_score += dic_idf[q_stem]*(k1+1) / k1*(1-b+b*len(doc_stem)/100)
                #print('good')
            except:
                try:
                    bm_25_score += dic_idf['Ġ'+q_stem] * (k1 + 1) / k1 * (1 - b + b * len(doc_stem) / 100)
                except:
                    #print('no')
                    bm_25_score += 15*(k1+1) / k1*(1-b+b*len(doc_stem)/100)
    return bm_25_score

class EmbeddingMixin:
    """
    Mixin for common functions in most embedding models. Each model should define its own bert-like backbone and forward.
    We inherit from RobertaModel to use from_pretrained 
    """
    def __init__(self, model_argobj):
        if model_argobj is None:
            self.use_mean = False
        else:
            self.use_mean = model_argobj.use_mean
        print("Using mean:", self.use_mean)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d

    def masked_mean_or_first(self, emb_all, mask):
        # emb_all is a tuple from bert - sequence output, pooler
        assert isinstance(emb_all, tuple)
        if self.use_mean:
            return self.masked_mean(emb_all[0], mask)
        else:
            return emb_all[0][:, 0] #sequence output的一个batch中的每个句子的第0个[CLS]表示

    def query_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")

    def body_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")

class EmbeddingMixin_reader:
    """
    Mixin for common functions in most embedding models. Each model should define its own bert-like backbone and forward.
    We inherit from RobertaModel to use from_pretrained
    """
    def __init__(self, model_argobj):
        if model_argobj is None:
            self.use_mean = False
        else:
            self.use_mean = model_argobj.use_mean
        print("Using mean:", self.use_mean)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d

    def masked_mean_or_first(self, emb_all, mask):
        # emb_all is a tuple from bert - sequence output, pooler
        assert isinstance(emb_all, tuple)
        if self.use_mean:
            return self.masked_mean(emb_all[0], mask)
        else:
            return emb_all[0][:, 0] #sequence output的一个batch中的每个句子的第0个[CLS]表示

    def query_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")

    def body_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")

    def query_emb_train(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")

    def body_emb_train(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")

class NLL(EmbeddingMixin):
    def forward(
            self,
            query_ids,
            attention_mask_q,
            input_ids_a=None,
            attention_mask_a=None,
            input_ids_b=None,
            attention_mask_b=None,
            is_query=True):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)

        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)

        logit_matrix = torch.cat([(q_embs * a_embs).sum(-1).unsqueeze(1),
                                  (q_embs * b_embs).sum(-1).unsqueeze(1)], dim=1)  # [B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0 * lsm[:, 0]
        return (loss.mean(),)



class NLL_reader_one_hot_gelu_equal_cls_add_bm25(EmbeddingMixin_reader):
    def forward(
            self,
            train_step,
            query_ids,
            attention_mask_q,
            input_ids_a=None,
            attention_mask_a=None,
            query_len=None,
            input_ids_reader=None,
            attention_mask_reader=None,
            input_ids_b=None,
            attention_mask_b=None,
            is_query=True):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)

        q_embs,_ = self.query_emb_train(query_ids, attention_mask_q)
        a_embs,sequence_output = self.body_emb_train(input_ids_a, attention_mask_a)
        b_embs,_ = self.body_emb_train(input_ids_b, attention_mask_b)
        #sequence_output [bz,max_seq_len,768]
        logit_matrix = torch.cat([(q_embs * a_embs).sum(-1).unsqueeze(1),
                                  (q_embs * b_embs).sum(-1).unsqueeze(1)], dim=1)  # [B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0 * lsm[:, 0]
        juhao_matrix = (input_ids_a == 4).nonzero()
        last_bz = 0
        last_juhao = 0
        avg_vetors = []
        bm25_list = []
        matching_embedding = torch.mul(q_embs,a_embs) #[bs,768]
        gelu = nn.GELU()
        matching_embedding = gelu(matching_embedding)
        loss_div = 0
        loss_equal = 0
        # print(juhao_matrix.shape)
        count = 1
        # print('one sample is start:')
        for i in range(juhao_matrix.shape[0]):
            bz_idx = juhao_matrix[i][0]
            if bz_idx == last_bz:
                now_juhao = juhao_matrix[i][1]
                if now_juhao < last_juhao+1 or now_juhao == last_juhao + 1:
                    sum_vector = torch.sum(sequence_output[bz_idx][last_juhao+1].unsqueeze(0), dim=0, keepdim=True)
                else: #[1,juhao]
                    sum_vector = torch.sum(sequence_output[bz_idx][last_juhao+1:now_juhao], dim=0, keepdim=True)
                avg_vector = torch.div(sum_vector,max(1,now_juhao-last_juhao-1))
                avg_vetors.append(avg_vector)
                query_token = tokenizer.convert_ids_to_tokens(query_ids[bz_idx][:query_len[bz_idx]])
                if now_juhao < last_juhao + 1 or now_juhao == last_juhao + 1:
                    #sequence_token = tokenizer.convert_ids_to_tokens(input_ids_a[bz_idx][last_juhao + 1])
                    sequence_token = tokenizer.convert_ids_to_tokens([4])
                else:
                    sequence_token = tokenizer.convert_ids_to_tokens(input_ids_a[bz_idx][last_juhao+1:now_juhao])

                bm25 = compute_bm25(query_token, sequence_token)
                bm25_list.append(bm25)
                last_juhao = now_juhao

            else:
                avg_vectors = torch.cat(avg_vetors,dim=0)
                logist_dis = torch.matmul(matching_embedding[last_bz].unsqueeze(0),torch.transpose(avg_vectors,0,1)) #(1,句子个数)
                logist_soft = F.softmax(logist_dis,dim=-1)
                target = torch.Tensor(bm25_list).unsqueeze(0).to(logist_dis.device)
                count += 1
                y_target = torch.argmax(target,dim=-1)
                crossentropyloss = nn.CrossEntropyLoss()
                loss_div += crossentropyloss(logist_soft, y_target)


                logist_equal = torch.matmul(a_embs[last_bz].unsqueeze(0),torch.transpose(avg_vectors,0,1))
                logist_equal_log_soft = F.log_softmax(logist_equal, dim=-1)
                target_equal = torch.Tensor([1]*logist_equal_log_soft.shape[-1]).unsqueeze(0).to(logist_dis.device)
                y_target_equal = F.softmax(target_equal,dim=-1)
                loss_equal += F.kl_div(logist_equal_log_soft, y_target_equal, reduction='mean')

                last_juhao = 0
                last_bz = bz_idx
                now_juhao = juhao_matrix[i][1]
                if now_juhao < last_juhao + 1 or now_juhao == last_juhao + 1:
                    sum_vector = torch.sum(sequence_output[bz_idx][last_juhao + 1].unsqueeze(0), dim=0, keepdim=True)
                else:
                    sum_vector = torch.sum(sequence_output[bz_idx][last_juhao + 1:now_juhao], dim=0, keepdim=True)
                avg_vector = torch.div(sum_vector, max(1,now_juhao - last_juhao - 1))
                avg_vetors = [avg_vector]
                #计算bm25
                query_token = tokenizer.convert_ids_to_tokens(query_ids[bz_idx][:query_len[bz_idx]])
                if now_juhao < last_juhao + 1 or now_juhao == last_juhao + 1:
                    sequence_token = tokenizer.convert_ids_to_tokens([4])
                else:
                    sequence_token = tokenizer.convert_ids_to_tokens(input_ids_a[bz_idx][last_juhao+1:now_juhao])
                query_token = [token[1:].lower() if token[0] == 'Ġ' else token.lower() for token in query_token]
                sequence_token = [token[1:].lower() if token[0] == 'Ġ' else token.lower() for token in sequence_token]
                bm25 = compute_bm25(query_token, sequence_token)
                bm25_list = [bm25]
                last_juhao = now_juhao


        weighted_div = 0.1
        weighted_equal = 1.0
        if train_step == 1 or train_step == 300001:
            print('train_step is {}'.format(train_step))
            print('weighted_div:{}'.format(weighted_div))
            print('weighted_equal:{}'.format(weighted_equal))
        return (loss.mean()+ weighted_div*(loss_div / count) + weighted_equal*(loss_equal / count),loss.mean(),(loss_div / count),(loss_equal / count),)


class NLL_MultiChunk(EmbeddingMixin):
    def forward(
            self,
            query_ids,
            attention_mask_q,
            input_ids_a=None,
            attention_mask_a=None,
            input_ids_b=None,
            attention_mask_b=None,
            is_query=True):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)

        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)

        [batchS, full_length] = input_ids_a.size()
        chunk_factor = full_length // self.base_len

        # special handle of attention mask -----
        attention_mask_body = attention_mask_a.reshape(
            batchS, chunk_factor, -1)[:, :, 0]  # [batchS, chunk_factor]
        inverted_bias = ((1 - attention_mask_body) * (-9999)).float()

        a12 = torch.matmul(
            q_embs.unsqueeze(1), a_embs.transpose(
                1, 2))  # [batch, 1, chunk_factor]
        logits_a = (a12[:, 0, :] + inverted_bias).max(dim=-
                                                      1, keepdim=False).values  # [batch]
        # -------------------------------------

        # special handle of attention mask -----
        attention_mask_body = attention_mask_b.reshape(
            batchS, chunk_factor, -1)[:, :, 0]  # [batchS, chunk_factor]
        inverted_bias = ((1 - attention_mask_body) * (-9999)).float()

        a12 = torch.matmul(
            q_embs.unsqueeze(1), b_embs.transpose(
                1, 2))  # [batch, 1, chunk_factor]
        logits_b = (a12[:, 0, :] + inverted_bias).max(dim=-
                                                      1, keepdim=False).values  # [batch]
        # -------------------------------------

        logit_matrix = torch.cat(
            [logits_a.unsqueeze(1), logits_b.unsqueeze(1)], dim=1)  # [B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0 * lsm[:, 0]
        return (loss.mean(),)


class RobertaDot_NLL_LN(NLL, RobertaForSequenceClassification):
    """None
    Compress embedding to 200d, then computes NLL loss.
    """

    def __init__(self, config, model_argobj=None):
        NLL.__init__(self, model_argobj)
        RobertaForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768)
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask,
                                return_dict=False)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)




class RobertaDot_NLL_LN_reader_one_hot_gelu_equal_cls_add_bm25(NLL_reader_one_hot_gelu_equal_cls_add_bm25, RobertaForSequenceClassification):
    """None
    Compress embedding to 200d, then computes NLL loss.
    """

    def __init__(self, config, model_argobj=None):
        NLL_reader_one_hot_gelu_equal_cls_add_bm25.__init__(self, model_argobj)
        RobertaForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768)
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)
        # self.row_wise_ffn = nn.Linear(config.hidden_size,1)
    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask,
                                return_dict=False)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1


    def query_emb_train(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask,
                                return_dict=False)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1,outputs1[0]

    def body_emb_train(self, input_ids, attention_mask):
        return self.query_emb_train(input_ids, attention_mask)

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)


class RobertaDot_CLF_ANN_NLL_MultiChunk(NLL_MultiChunk, RobertaDot_NLL_LN):
    def __init__(self, config):
        RobertaDot_NLL_LN.__init__(self, config)
        self.base_len = 512

    def body_emb(self, input_ids, attention_mask):
        [batchS, full_length] = input_ids.size()
        chunk_factor = full_length // self.base_len

        input_seq = input_ids.reshape(
            batchS,
            chunk_factor,
            full_length //
            chunk_factor).reshape(
            batchS *
            chunk_factor,
            full_length //
            chunk_factor)
        attention_mask_seq = attention_mask.reshape(
            batchS,
            chunk_factor,
            full_length //
            chunk_factor).reshape(
            batchS *
            chunk_factor,
            full_length //
            chunk_factor)

        outputs_k = self.roberta(input_ids=input_seq,
                                 attention_mask=attention_mask_seq)

        compressed_output_k = self.embeddingHead(
            outputs_k[0])  # [batch, len, dim]
        compressed_output_k = self.norm(compressed_output_k[:, 0, :])

        [batch_expand, embeddingS] = compressed_output_k.size()
        complex_emb_k = compressed_output_k.reshape(
            batchS, chunk_factor, embeddingS)

        return complex_emb_k  # size [batchS, chunk_factor, embeddingS]


class HFBertEncoder(BertModel):
    def __init__(self, config):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
        self.init_weights()
    @classmethod
    def init_encoder(cls, args, dropout: float = 0.1):
        cfg = BertConfig.from_pretrained("/data/xushicheng/bert-base")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        return cls.from_pretrained("/data/xushicheng/bert-base", config=cfg)
    def forward(self, input_ids, attention_mask):
        hidden_states = None
        output = super().forward(input_ids=input_ids,attention_mask=attention_mask,return_dict=True)
        return output.pooler_output
    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size


class BiEncoder(nn.Module):
    """ Bi-Encoder model component. Encapsulates query/question and context/passage encoders.
    """
    def __init__(self, args):
        super(BiEncoder, self).__init__()
        cfg = RobertaConfig.from_pretrained('roberta')
        self.question_model = RobertaModel.from_pretrained("roberta",config=cfg)
        self.ctx_model = RobertaModel.from_pretrained("roberta",config=cfg)
    def query_emb(self, input_ids, attention_mask):
        output = self.question_model(input_ids=input_ids,attention_mask=attention_mask,return_dict=True)
        return output.pooler_output
    def body_emb(self, input_ids, attention_mask):
        output = self.ctx_model(input_ids=input_ids,attention_mask=attention_mask,return_dict=True)
        return output.pooler_output
    def forward(self, query_ids, attention_mask_q, input_ids_a = None, attention_mask_a = None, input_ids_b = None, attention_mask_b = None):
        if input_ids_b is None:
            q_embs = self.query_emb(query_ids, attention_mask_q)
            a_embs = self.body_emb(input_ids_a, attention_mask_a)
            return (q_embs, a_embs)
        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)
        logit_matrix = torch.cat([(q_embs*a_embs).sum(-1).unsqueeze(1), (q_embs*b_embs).sum(-1).unsqueeze(1)], dim=1) #[B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0*lsm[:,0]
        return (loss.mean(),)
        

# --------------------------------------------------
ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            RobertaConfig,
        ) if hasattr(conf,'pretrained_config_archive_map')
    ),
    (),
)


default_process_fn = four_process_fn_add_reader


class MSMarcoConfig:
    def __init__(self, name, model, process_fn=default_process_fn, use_mean=True, tokenizer_class=RobertaTokenizer, config_class=RobertaConfig):
        self.name = name
        self.process_fn = process_fn
        self.model_class = model
        self.use_mean = use_mean
        self.tokenizer_class = tokenizer_class
        self.config_class = config_class


configs = [
    MSMarcoConfig(name="rdot_nll",
                model=RobertaDot_NLL_LN,
                use_mean=False,
                ),
    MSMarcoConfig(name="rdot_nll_multi_chunk",
                model=RobertaDot_CLF_ANN_NLL_MultiChunk,
                use_mean=False,
                ),
    MSMarcoConfig(name="dpr",
                model=BiEncoder,
                tokenizer_class=RobertaTokenizer,
                config_class=RobertaConfig,
                use_mean=False,
                ),
    MSMarcoConfig(name="reader_one_hot_gelu_equal_cls_add_bm25",
                  model=RobertaDot_NLL_LN_reader_one_hot_gelu_equal_cls_add_bm25,
                  use_mean=False,
                  ),
]

MSMarcoConfigDict = {cfg.name: cfg for cfg in configs}
