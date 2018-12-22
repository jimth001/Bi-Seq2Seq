import tensorflow as tf
import deep_components
import numpy as np
import nltk
import pickle
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.contrib.seq2seq import  sequence_loss
from tensorflow.contrib.seq2seq import BahdanauAttention,AttentionWrapper
from tensorflow.contrib.seq2seq import BasicDecoder,BeamSearchDecoder,dynamic_decode
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.seq2seq import TrainingHelper,tile_batch

embed_size = 100
encoder_size = 100
MAX_LENGTH = np.int32(20)

class seq2seq:
    def __init__(self,vocab_size,learning_rate,encoder_size,max_length,embedding_size,sos_token,eos_token,unk_token,beam_size=5):
        self.vocab_size=vocab_size
        self.lr=learning_rate
        self.encoder_size=encoder_size
        self.max_length=max_length
        self.embedding_size=embedding_size
        self.SOS_token=sos_token
        self.EOS_token=eos_token
        self.UNK_token=unk_token
        self.beam_search_size=beam_size
        with tf.variable_scope('placeholder_and_embedding'):
            self.query = tf.placeholder(shape=(None, None), dtype=tf.int32)
            self.query_length = tf.placeholder(shape=(None,), dtype=tf.int32)
            self.reply = tf.placeholder(shape=(None, None), dtype=tf.int32)
            self.reply_length = tf.placeholder(shape=(None,), dtype=tf.int32)
            self.decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32)
            self.decoder_target = tf.placeholder(shape=(None, None), dtype=tf.int32)
            self.decoder_length = tf.placeholder(shape=(None,), dtype=tf.int32)
            self.batch_size = tf.placeholder(shape=(),dtype=tf.int32)
            self.embedding_pl = tf.placeholder(dtype=tf.float32, shape=(self.vocab_size, embedding_size),
                                                      name='embedding_source_pl')
            word_embedding = tf.get_variable(name='word_embedding', shape=(self.vocab_size, embedding_size),
                                               dtype=tf.float32, trainable=True)
            self.init_embedding = word_embedding.assign(self.embedding_pl)
            self.max_target_sequence_length = tf.reduce_max(self.decoder_length, name='max_target_len')
            self.mask = tf.sequence_mask(self.decoder_length, self.max_target_sequence_length, dtype=tf.float32,
                                         name='masks')

        with tf.variable_scope("query_encoder"):
            self.query_encoder = deep_components.gru_encoder(word_embedding, self.encoder_size)
            query_out,query_state = self.query_encoder(seq_index=self.query, seq_len=self.query_length)
        with tf.variable_scope("reply_encoder"):
            self.reply_encoder = deep_components.gru_encoder(word_embedding, self.encoder_size)
            reply_out,reply_state = self.reply_encoder(seq_index=self.reply, seq_len=self.reply_length)
        with tf.variable_scope("decoder"):
            combined_encoder_state = tf.concat([query_state, reply_state], axis=1)
            tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(
                combined_encoder_state, multiplier=self.beam_search_size)
            tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
                query_out, multiplier=self.beam_search_size)
            tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
                self.query_length, multiplier=self.beam_search_size)
            decoder_cell=deep_components.AttentionGRUCell(memory=tiled_encoder_outputs,memory_size=self.encoder_size,
                                                          attention_size=self.encoder_size,
                                                          embedding_dims=self.embedding_size,
                                                          rnn_units=self.encoder_size*2)

            '''decoder_gru = GRUCell(self.encoder_size * 2)
            attention_mechanism = BahdanauAttention(
                num_units=self.encoder_size,
                memory=tiled_encoder_outputs,
                memory_sequence_length=tiled_sequence_length)
            attention_cell = AttentionWrapper(decoder_gru, attention_mechanism,
                                              attention_layer_size=self.encoder_size)
            decoder_initial_state_beam = attention_cell.zero_state(
                dtype=tf.float32, batch_size=tf.cast(self.batch_size * self.beam_search_size,dtype=tf.int32)).clone(
                cell_state=tiled_encoder_final_state)'''
            #############################
            #attention_cell=decoder_gru
            #decoder_initial_state_beam = tiled_encoder_final_state
            ##############################
            decode_out_layer = tf.layers.Dense(self.vocab_size, name='output_layer', _reuse=tf.AUTO_REUSE)
        with tf.variable_scope("seq2seq-train"):
            # train
            self.tiled_d_in=tile_batch(self.decoder_inputs,multiplier=self.beam_search_size)
            self.tiled_d_tgt=tile_batch(self.decoder_target,multiplier=self.beam_search_size)
            train_helper=TrainingHelper(tf.contrib.seq2seq.tile_batch(tf.nn.embedding_lookup(word_embedding,self.decoder_inputs),multiplier=self.beam_search_size),
                                        sequence_length=tile_batch(self.decoder_length,multiplier=self.beam_search_size), name="train_helper")
            train_decoder=BasicDecoder(decoder_cell,train_helper,initial_state=tiled_encoder_final_state,output_layer=decode_out_layer)
            self.dec_output, _, self.gen_len = dynamic_decode(train_decoder, impute_finished=True,
                                                   maximum_iterations=self.max_target_sequence_length)
            #self.gen_max_len=tf.reduce_max(self.gen_len)
            #self.padding=tf.zeros(shape=(self.batch_size,self.max_length-self.gen_max_len,self.vocab_size),dtype=tf.float32)
            #self.padding=tile_batch(self.padding,multiplier=self.beam_search_size)
            self.dec_logits = tf.identity(self.dec_output.rnn_output)
            #self.dec_logits = tf.concat((self.dec_logits,self.padding),axis=1)
            self.decoder_target_mask = tile_batch(self.mask,multiplier=self.beam_search_size)
            self.cost = sequence_loss(self.dec_logits,
                                      tile_batch(self.decoder_target,multiplier=self.beam_search_size),
                                      self.decoder_target_mask)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)
        with tf.variable_scope("seq2seq_beam_search_generate"):
            start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.SOS_token
            beam_infer_decoder = BeamSearchDecoder(decoder_cell, embedding=word_embedding, end_token=self.EOS_token,
                                              start_tokens=start_tokens, initial_state=tiled_encoder_final_state,
                                              beam_width=self.beam_search_size, output_layer=decode_out_layer)
            self.bs_outputs, _, _ = dynamic_decode(beam_infer_decoder, maximum_iterations=self.max_length)
        with tf.variable_scope("greedy_generate"):
            decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=word_embedding,
                                                                       start_tokens=start_tokens, end_token=self.EOS_token)
            inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=decoding_helper,
                                                                initial_state=tiled_encoder_final_state,
                                                                output_layer=decode_out_layer)
            self.greedy_outputs,_,_=dynamic_decode(inference_decoder, maximum_iterations=self.max_length)

    def parse_beam_search_result(self,predict_ids):
        '''
                将beam_search返回的结果去掉unk，截断EOS后面得内容
                :param predict_ids: 列表，长度为batch_size，每个元素都是decode_len*beam_size的数组
                :param id2word: vocab字典
                :return:
                '''
        all_result=[]
        for single_predict in predict_ids:
            one_beam_result=[]
            for i in range(self.beam_search_size):
                predict_list = np.ndarray.tolist(single_predict[:, i])
                one_beam_result.append(self.parse_output([predict_list])[0])
            all_result.append(one_beam_result)
        return all_result

    def train(self, sess,train_data, query, query_length, reply, reply_len, decoder_inputs, decoder_target ,decoder_length,batch_size):
        """
        feed data to train seq2seq model.
        :param sess: session
        :param encoder_inputs: encoder inputs
        :param encoder_length: encoder inputs sequence length, (batch, )
        :param decoder_inputs: decoder inputs
        :param decoder_target: decoder target
        :return:
        """
        #res = [self.optimizer, self.cost]
        res = [self.optimizer, self.cost,
               self.dec_logits, self.decoder_target_mask,self.dec_output,self.gen_len,self.tiled_d_in,self.tiled_d_tgt]
        _, cost,dec_logits,decoder_target_mask,dec_out,gen_len,d_in,d_tgt = sess.run(res,
                           feed_dict={self.query: np.array(query),
                                      self.query_length: np.array(query_length),
                                      self.reply: np.array(reply),
                                      self.reply_length: np.array(reply_len),
                                      self.decoder_inputs: np.array(decoder_inputs),
                                      self.decoder_target: np.array(decoder_target),
                                      self.decoder_length: np.array(decoder_length),
                                      self.batch_size: batch_size,
                                      })
        #print([train_data.indices2sentence(x) for x in query])
        #print([train_data.indices2sentence(x) for x in reply])
        #print([train_data.indices2sentence(x) for x in d_in])
        #print([train_data.indices2sentence(x) for x in d_tgt])
        return cost

    def evaluate(self, sess, query,  reply,  decoder_inputs, decoder_target,batch_size=256):
        """
                feed data to train seq2seq model.
                :param sess: session
                :param encoder_inputs: encoder inputs
                :param encoder_length: encoder inputs sequence length, (batch, )
                :param decoder_inputs: decoder inputs
                :param decoder_target: decoder target
                :return:
                """
        data_num=len(query)
        low=0
        total_loss=0.0
        while low<data_num:
            n_samples=min([batch_size,data_num-low])
            batch_q,batch_q_len=padding_batch(copy_list(query[low:low+n_samples]))
            batch_r, batch_r_len = padding_batch(copy_list(reply[low:low + n_samples]))
            batch_d_in,batch_d_len=padding_batch(copy_list(decoder_inputs[low:low+n_samples]))
            batch_d_tgt,batch_d_len=padding_batch(copy_list(decoder_target[low:low+n_samples]))
            cost = sess.run(self.cost,
                           feed_dict={self.query: np.array(batch_q),
                                      self.query_length: np.array(batch_q_len),
                                      self.reply: np.array(batch_r),
                                      self.reply_length: np.array(batch_r_len),
                                      self.decoder_inputs: np.array(batch_d_in),
                                      self.decoder_target: np.array(batch_d_tgt),
                                      self.decoder_length: np.array(batch_d_len),
                                      self.batch_size: n_samples,
                                      })
            total_loss+=cost*n_samples
            low+=n_samples
        return total_loss/data_num

    def generate(self, sess, query, query_length, reply, reply_len,use_beam_search):
        """
        feed data to generate.
        :param sess: session
        :param encoder_inputs: encoder inputs
        :param encoder_length: encoder inputs sequence length,
        :return:
        """
        if query.ndim == 1:
            query = query.reshape((1, -1))
            query_length = query_length.reshape((1,))
        if reply.ndim == 1:
            reply = reply.reshape((1, -1))
            reply_len = reply_len.reshape((1,))
        decoder_inputs = np.asarray([[self.SOS_token]*self.max_length] * len(query), dtype="int32")
        if use_beam_search:
            res = [self.bs_outputs]
        else:
            res = [self.greedy_outputs]
        generate = sess.run(res,
                            feed_dict={self.query: np.array(query),
                                      self.query_length: np.array(query_length),
                                      self.reply: np.array(reply),
                                      self.reply_length: np.array(reply_len),
                                      self.decoder_inputs: decoder_inputs,
                                      self.batch_size:len(query),
                                       })[0]
        if use_beam_search:
            return generate.predicted_ids
        else:
            return generate.sample_id


    def parse_output(self,token_indices):
        res = []
        for one_sen in token_indices:
            sen = []
            for token in one_sen:
                if token != self.EOS_token:  # end
                    if token != self.UNK_token:
                        sen.append(token)
                else:
                    break
            res.append(sen)
        return res

class prepare_data():
    def __init__(self,query_file,reply_file,target_file,embedding_file):
        self.embedding_file=embedding_file
        self.query_file=query_file
        self.reply_file=reply_file
        self.target_file=target_file
        self.embedding,self.vocab_hash=self.load_fasttext_embedding()
        self.SOS_token=np.int32(self.add_term_to_embedding('<SOS>',[0.0]*len(self.embedding[0])))
        self.UNK_token=np.int32(self.add_term_to_embedding('<UNK>',[0.0]*len(self.embedding[0])))
        self.EOS_token=np.int32(self.add_term_to_embedding('<EOS>', [0.0] * len(self.embedding[0])))
        self.index2word=self.gen_index2word_dict()
        self.query=self.process_query_file(max_len=MAX_LENGTH)
        self.reply=self.process_reply_file(max_len=MAX_LENGTH)
        self.target_input,self.target_output=self.process_target_file(max_len=MAX_LENGTH)
        #self.padding_and_get_len(self.target_input,max_len=MAX_LENGTH)
        #self.target_len=self.padding_and_get_len(self.target_output,max_len=MAX_LENGTH)
        print('finish preprocess')

    def padding_and_get_len(self,list,max_len=None):
        len_list=[]
        if max_len is None:
            #现在encoder输入在训练过程中padding，只有decoder的输入输出提前padding到MAX_LENGTH
            pass
        else:
            for i in range(0, len(list)):
                if len(list[i]) < max_len:
                    len_list.append(len(list[i]))
                    list[i] = list[i] + [0] * (max_len - len(list[i]))
                else:
                    len_list.append(max_len)
                    list[i] = list[i][:max_len]
        return len_list

    def process_query_file(self,max_len):
        source=[]
        total_num = 0
        cut_num = 0
        with open(self.query_file,'r',encoding='utf-8') as f:
            for line in f:
                words=nltk.word_tokenize(line.strip())
                if len(words)>max_len:
                    words=words[:max_len]
                    cut_num += 1
                total_num+=1
                source.append(self.sentence2indices(words))
        print('总数量：', total_num)
        print('截断得句子数量', cut_num)
        return source

    def process_reply_file(self,max_len):
        source=[]
        total_num = 0
        cut_num = 0
        with open(self.reply_file,'r',encoding='utf-8') as f:
            for line in f:
                words=nltk.word_tokenize(line.strip())
                if len(words)>max_len:
                    words=words[:max_len]
                    cut_num += 1
                total_num += 1
                source.append(self.sentence2indices(words))
        print('总数量：', total_num)
        print('截断得句子数量', cut_num)
        return source

    def process_target_file(self,max_len):
        target_output = []
        target_input = []
        total_num=0
        cut_num=0
        with open(self.target_file, 'r', encoding='utf-8') as f:
            for line in f:
                words = nltk.word_tokenize(line.strip())
                if len(words)>max_len:
                    words=words[:max_len]
                    cut_num+=1
                total_num+=1
                target_input.append(self.sentence2indices(words,with_sos=True))
                target_output.append(self.sentence2indices(words,with_eos=True))
        print('总数量：',total_num)
        print('截断得句子数量',cut_num)
        return target_input,target_output

    def gen_index2word_dict(self):
        i2d=[]
        tmp=sorted(self.vocab_hash.items(),key=lambda d:d[1])
        for item in tmp:
            i2d.append(item[0])
        return i2d

    def add_term_to_embedding(self,term,vector):
        self.vocab_hash[term]=len(self.vocab_hash)
        self.embedding.append(vector)
        return self.vocab_hash[term]

    def load_fasttext_embedding(self):
        vectors = []
        vocab_hash = {}
        with open(self.embedding_file, 'r', encoding='utf-8') as f:
            first_line = True
            for line in f:
                if first_line:
                    first_line = False
                    continue
                strs = line.strip().split(' ')
                vocab_hash[strs[0]] = len(vectors)
                vectors.append([float(s) for s in strs[1:]])
        return vectors, vocab_hash

    def indices2sentence(self, idxs):
        return " ".join([self.index2word[idx] for idx in idxs])

    def sentence2indices(self, words, with_sos=False,with_eos=False,with_unk=True):
        idxs=[]
        if with_sos:
            idxs.append(self.SOS_token)
        if with_unk:
            idxs += [self.vocab_hash.get(token,self.UNK_token) for token in words]
        else:
            idxs += [self.vocab_hash.get(token) for token in words if token in self.vocab_hash]  # default to <unk>
        if with_eos:
            idxs.append(self.EOS_token)
        return idxs

def preprocess():
    train_data = prepare_data(query_file='./data/train.query',
                              reply_file='./data/train.reply',
                              target_file='./data/train.target',
                              embedding_file='./data/embedding')
    pickle.dump(train_data, open('./data/train.pkl', 'wb'), protocol=True)
    val_data = prepare_data(query_file='./data/val.query',
                              reply_file='./data/val.reply',
                              target_file='./data/val.target',
                            embedding_file='./data/embedding')
    pickle.dump(val_data, open('./data/val.pkl', 'wb'), protocol=True)
    test_data = prepare_data(query_file='./data/test.query',
                              reply_file='./data/test.reply',
                              target_file='./data/test.target',
                            embedding_file='./data/embedding')
    pickle.dump(test_data, open('./data/test.pkl', 'wb'), protocol=True)

def generate_batches(model_path,beam_size=5,output_path='./output/result',batch_size=32,use_beam_search=True):
    test_data = pickle.load(open('./data/test.pkl', 'rb'))
    if not use_beam_search:
        beam_size=1
    nmt = seq2seq(vocab_size=len(test_data.embedding),
                  learning_rate=1e-3, encoder_size=encoder_size, max_length=MAX_LENGTH,
                  embedding_size=embed_size, sos_token=test_data.SOS_token, eos_token=test_data.EOS_token,
                  unk_token=test_data.UNK_token,beam_size=beam_size)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver(max_to_keep=20)
    saver.restore(sess, model_path)
    output=[]
    low_pointer = 0
    data_num=len(test_data.query)
    while low_pointer<data_num:
        n_samples = min([batch_size, data_num - low_pointer])
        query, query_len = padding_batch(copy_list(test_data.query[low_pointer:low_pointer+n_samples]))
        reply, reply_len = padding_batch(copy_list(test_data.reply[low_pointer:low_pointer+n_samples]))
        #print(test_data.indices2sentence(query[0]))
        #print(query_len[0])
        #print(test_data.indices2sentence(reply[0]))
        #print(reply_len[0])
        gen_result=nmt.generate(sess, np.array(query), np.array(query_len),
                     np.array(reply), np.array(reply_len),use_beam_search=use_beam_search)
        if use_beam_search:
            indexs=nmt.parse_beam_search_result(gen_result)
            for sample in indexs:
                for one_beam in sample:
                    output.append(test_data.indices2sentence(one_beam))
            #print('--------------------------------------------------')
            #print(test_data.indices2sentence(indexs[0][0]))
        else:
            indexs=nmt.parse_output(gen_result)
            for sen in indexs:
                output.append(test_data.indices2sentence(sen))
            #print('--------------------------------------------------')
            #print(test_data.indices2sentence(indexs[0]))
        low_pointer+=n_samples
    with open(output_path,'w',encoding='utf-8') as fw:
        for s in output:
            fw.write(s+'\n')

def padding_batch(input_list):
    in_len=[len(i) for i in input_list]
    new_in=pad_sequences(input_list,padding='post')
    return new_in,in_len

def copy_list(list):
    new_list = []
    for l in list:
        if type(l) == type([0]) or type(l) == type(np.array([0])):
            new_list.append(copy_list(l))
        else:
            new_list.append(l)
    return new_list

def train_onehotkey(beam_size=5,batch_size=128,n_epochs=8,batches_per_evaluation=500,previous_model_path=None,continue_train=False):
    train_data=pickle.load(open('./data/train.pkl', 'rb'))
    val_data=pickle.load(open('./data/val.pkl', 'rb'))
    print('build graph')
    nmt=seq2seq(vocab_size=len(train_data.embedding),
                learning_rate=1e-3,encoder_size=encoder_size,max_length=MAX_LENGTH,
                embedding_size=embed_size,sos_token=train_data.SOS_token,eos_token=train_data.EOS_token,
                unk_token=train_data.UNK_token,beam_size=beam_size)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver=tf.train.Saver(max_to_keep=20)
    print('init graph')
    if continue_train:
        saver.restore(sess,previous_model_path)
    else:
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(nmt.init_embedding,feed_dict={nmt.embedding_pl:train_data.embedding})
    epoch = 0
    low_pointer=0
    train_data_num=len(train_data.query)
    total_batch=0
    best_loss=1000000
    print('start train')
    while epoch < n_epochs:
        n_samples=min([batch_size,train_data_num-low_pointer])
        query = train_data.query[low_pointer:low_pointer+n_samples]
        query, query_len = padding_batch(copy_list(query))
        reply = train_data.reply[low_pointer:low_pointer + n_samples]
        reply, reply_len = padding_batch(copy_list(reply))
        decoder_inputs = train_data.target_input[low_pointer:low_pointer+n_samples]
        decoder_inputs,decoder_length=padding_batch(copy_list(decoder_inputs))
        decoder_target = train_data.target_output[low_pointer:low_pointer+n_samples]
        decoder_target, decoder_length = padding_batch(copy_list(decoder_target))
        #print(train_data.indices2sentence(query[1]))
        #print(train_data.indices2sentence(reply[1]))
        #print(train_data.indices2sentence(decoder_inputs[1]))
        #print(train_data.indices2sentence(decoder_target[1]))
        train_loss = nmt.train(sess,train_data,query,query_len,reply,reply_len,decoder_inputs,decoder_target,decoder_length,batch_size=n_samples)
        total_batch +=1
        low_pointer+=n_samples
        if total_batch % 20 == 0:
            print(train_loss)
        if total_batch%batches_per_evaluation==0:
            val_loss=nmt.evaluate(sess,val_data.query,val_data.reply,val_data.target_input,val_data.target_output,batch_size=32)
            print("epoch: {0}/{1}, batch_num: {2} train_loss: {3} val_loss: {4}".format(epoch, n_epochs, total_batch, train_loss, val_loss))
            if val_loss<best_loss:
                best_loss=val_loss
                saver.save(sess,'./model/best.{0}.model'.format(total_batch))
        if low_pointer>=train_data_num:
            low_pointer=0
            epoch+=1
            print('epoch {0} ended'.format(epoch))
            saver.save(sess,'./model/epoch.{0}.model'.format(epoch))
    sess.close()

if __name__=='__main__':
    #preprocess()
    #train_onehotkey(batch_size=32,beam_size=1,n_epochs=10,previous_model_path='./model/epoch.10.model',continue_train=False)
    generate_batches(beam_size=1,output_path='./output/beamsearch_result',model_path='./model/epoch.10.model',batch_size=32,use_beam_search=True)
    print('all work has finished')



