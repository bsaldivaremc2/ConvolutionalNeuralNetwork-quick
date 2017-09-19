import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def create_graph_bools(train_batch,layers,test_batch,mode='train',first_run=True,stddev_n = 0.1, learning_rate = 1e-4,iters=1,model_file='test_model',batch_proc=True, width=256,height=256):
    if mode == 'train':
        test_batch_bool=False
        only_feed_forward=False
        if first_run == True:
            restore_session = False
        else:
            restore_session = True
        save_model = True
    elif mode == 'test':
        test_batch_bool=True
        only_feed_forward=True
        restore_session = True
        save_model = False
    elif mode == 'evaluate':
        test_batch_bool=False
        only_feed_forward=True
        restore_session = False
        save_model = False

    create_graph_return = create_graph(train_batch,layers=layers,test_batch=test_batch,
                width=width,height=height, test_batch_bool=test_batch_bool,only_feed_forward=only_feed_forward,
                restore_session = restore_session, save_model = save_model, stddev_n = stddev_n, 
                learning_rate = learning_rate,iters=iters,model_file=model_file)
    return create_graph_return.copy()



def get_previous_features(i_layer):
    convx_dims = i_layer.get_shape().as_list()
    output_features = 1
    for dim in range(1,len(convx_dims)):
        output_features=output_features*convx_dims[dim]
    return output_features
def create_conv(iDic,input_layer,iName,prev_dic,stddev_n = 0.1,norm_offset=0,norm_scale=1,norm_epsilon=1e-6):
    W_name = "W"+iName
    b_name = "b"+iName
    Z_name = "Z"+iName
    conv_name = "conv"+iName
    relu_name = "relu"+iName
    maxpool_name = "maxpool"+iName
    keep_prob_name = 'keep_prob'+iName
    dropout_name = 'dropout'+iName
    if prev_dic['type']=='CV':
        iDic['input_depth']=prev_dic['depth']
    elif prev_dic['type']=='input_layer':
        if 'prev_channels' in iDic.keys():
            iDic['input_depth']=iDic['prev_channels']
        else:
            iDic['input_depth']=1
    if iDic['type']=='CV':
        iDic['W'] = tf.Variable(tf.truncated_normal([iDic['filter_w'], iDic['filter_w'], iDic['input_depth'], iDic['depth']], stddev=stddev_n),name=W_name)
        iDic['b'] = tf.Variable(tf.constant(stddev_n, shape=[iDic['depth']]),name=b_name)
        iDic['conv']= tf.nn.conv2d(input_layer, iDic['W'], strides=iDic['filter_stride'], padding='SAME',name=conv_name) + iDic['b']
        if 'norm_bool' in iDic.keys():
            if iDic['norm_bool']==True:
                iDic['mean'], iDic['variance'] = tf.nn.moments(iDic['conv'],[0,1,2])
                iDic['norm'] = tf.nn.batch_normalization(iDic['conv'],iDic['mean'],iDic['variance'],norm_offset,norm_scale,norm_epsilon)
                iDic['relu']= tf.nn.relu(iDic['norm'],name=relu_name)
            else:
                iDic['relu']= tf.nn.relu(iDic['conv'],name=relu_name)
        else:
            iDic['relu']= tf.nn.relu(iDic['conv'],name=relu_name)
        if 'max_pooling' in iDic.keys():
            if iDic['max_pooling']==True:
                iDic['max'] = tf.nn.max_pool(iDic['relu'], ksize=iDic['max_pool_mask'],strides=iDic['max_pool_stride'], padding=iDic['padding'])
                iDic['output_label']='max'
            else:
                iDic['output_label']='relu'
        else:
            iDic['output_label']='relu'
    elif iDic['type']=='CV2FC':
        iDic['input_features'] = get_previous_features(input_layer)
        iDic['input_layer'] = tf.reshape(input_layer, [-1, iDic['input_features']])
        iDic['W'] = tf.Variable(tf.truncated_normal([iDic['input_features'], iDic['neurons']], stddev=stddev_n),name=W_name)
        iDic['b'] = tf.Variable(tf.constant(stddev_n, shape=[iDic['neurons']]),name=b_name) 
        iDic['Z'] = tf.matmul(iDic['input_layer'], iDic['W'],name=Z_name) + iDic['b']
        if 'norm_bool' in iDic.keys():
            if iDic['norm_bool']==True:
                iDic['mean'], iDic['variance'] = tf.nn.moments(iDic['Z'],[0])
                iDic['norm'] = tf.nn.batch_normalization(iDic['Z'],iDic['mean'],iDic['variance'],norm_offset,norm_scale,norm_epsilon)
                iDic['relu']= tf.nn.relu(iDic['norm'],name=relu_name)
            else:
                iDic['relu']= tf.nn.relu(iDic['Z'],name=relu_name)
        else:
            iDic['relu']= tf.nn.relu(iDic['Z'],name=relu_name)
        if 'drop_out_bool' in iDic.keys():
            if iDic['drop_out_bool'] == True:
                iDic['keep_prob'] = tf.placeholder(tf.float32,name=keep_prob_name)
                iDic['dropout'] = tf.nn.dropout(iDic['relu'], iDic['keep_prob'],name=dropout_name)
                iDic['output_label']='dropout'
            else:
                iDic['output_label']='relu'
        else:
            iDic['output_label']='relu'
    elif iDic['type']=='FC':
        iDic['input_features'] = get_previous_features(input_layer)
        iDic['W'] = tf.Variable(tf.truncated_normal([iDic['input_features'], iDic['neurons']], stddev=stddev_n),name=W_name)
        iDic['b'] = tf.Variable(tf.constant(stddev_n, shape=[iDic['neurons']]),name=b_name) 
        iDic['Z'] = tf.matmul(input_layer, iDic['W'],name=Z_name) + iDic['b']
        if 'norm_bool' in iDic.keys():
            if iDic['norm_bool']==True:
                iDic['mean'], iDic['variance'] = tf.nn.moments(iDic['Z'],[0])
                iDic['norm'] = tf.nn.batch_normalization(iDic['Z'],iDic['mean'],iDic['variance'],norm_offset,norm_scale,norm_epsilon)
                iDic['relu']= tf.nn.relu(iDic['norm'],name=relu_name)
            else:
                iDic['relu']= tf.nn.relu(iDic['Z'],name=relu_name)
        else:
            iDic['relu']= tf.nn.relu(iDic['Z'],name=relu_name)
        if 'drop_out_bool' in iDic.keys():
            if iDic['drop_out_bool'] == True:
                iDic['keep_prob'] = tf.placeholder(tf.float32,name=keep_prob_name)
                iDic['dropout'] = tf.nn.dropout(iDic['relu'], iDic['keep_prob'],name=dropout_name)
                iDic['output_label']='dropout'
            else:
                iDic['output_label']='relu'
        else:
            iDic['output_label']='relu'
def balance_positive_negative(iNp,iBatchSize=256,v=False):
    negative_examples = iNp[iNp[:,-1]==0]
    positive_examples = iNp[iNp[:,-1]==1]
    mini_batch_size_half = iBatchSize//2
    negative_examples_m = negative_examples.shape[0]
    positive_examples_m = positive_examples.shape[0]
    positive_batches = positive_examples_m//mini_batch_size_half
    positive_batch_residual = positive_examples_m%mini_batch_size_half
    negative_batches = negative_examples_m//mini_batch_size_half
    negative_batch_residual = negative_examples_m%mini_batch_size_half
    balanced_batches = min(positive_batches,negative_batches)

    batch_list = []
    for batch in range(0,balanced_batches):
        start_batch = batch*mini_batch_size_half
        end_batch = (batch+1)*mini_batch_size_half
        _ = np.concatenate((positive_examples[start_batch:end_batch,:],negative_examples[start_batch:end_batch,:]),0)
        xTemp = (_[:,:-1]).astype('float32')
        xTemp=xTemp/255.0
        yTemp = _[:,-1]
        yTemp = np.asarray(pd.get_dummies(yTemp))
        batch_list.append([xTemp,yTemp])
    if v==True:
        print("Pos batches",positive_batches,"Pos res",positive_batch_residual)
        print("Neg batches",negative_batches,"Neg res",negative_batch_residual)
        print("Selected batches number",balanced_batches)
        print("#batches",len(batch_list))
    return batch_list.copy()
def plot_list(iList,figsize=(10,8),title="Loss/Eff",xlabel="Iters",ylabel="Loss/Eff"): 
    plt_loss = np.asarray(iList)
    fig = plt.figure(figsize=figsize)
    plt.plot(np.arange(0,plt_loss.size),plt_loss)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
##############
def create_graph(train_batch,layers,test_batch=None,width=256,height=256,
                 batch_proc=True, test_batch_bool=False,
                 restore_session = False, save_model = False, only_feed_forward=False,
                 stddev_n = 0.1, learning_rate = 1e-4,iters=4,model_file='CNN_model'):
    _x_batch = train_batch[0][0]
    _y_batch = train_batch[0][1]
    class_output = _y_batch.shape[1]
    
    tf.reset_default_graph()

    x_flat = width * height
    x = tf.placeholder(tf.float32, shape=[None, x_flat])
    y_ = tf.placeholder(tf.float32, shape=[None, class_output])
    x_image = tf.reshape(x, [-1,width,height,1])  

    layers.insert(0,{'x_image':x_image,'output_label':'x_image','type':'input_layer'})

    for i in range(1,len(layers)):
        print("Creating layer:",layers[i]['name'])
        create_conv(iDic=layers[i],input_layer=layers[i-1][layers[i-1]['output_label']],iName=layers[i]['name'],prev_dic=layers[i-1])

    FCL_input=layers[-1][layers[-1]['output_label']]
    FCL_input_features = get_previous_features(FCL_input)
    W_FCL = tf.Variable(tf.truncated_normal([FCL_input_features, class_output], stddev=stddev_n))
    b_FCL = tf.Variable(tf.constant(stddev_n, shape=[class_output])) 
    FCL=tf.matmul(FCL_input, W_FCL) + b_FCL
    y_CNN = tf.nn.softmax(FCL)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]))

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_CNN,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init_op = tf.global_variables_initializer()
    
    ###########
    with tf.Session() as s:
        print("Starting session")
        test_keep_prob = 1.0
        s.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if restore_session==True:
            #tf.reset_default_graph()
            saver.restore(s,tf.train.latest_checkpoint('./'))
        dic_to_feed = {x:_x_batch,y_:_y_batch}
        for _layer in layers:
            if 'drop_out_bool' in _layer.keys():
                if _layer['drop_out_bool'] == True:
                    dic_to_feed[_layer['keep_prob']]=test_keep_prob        
        for _ in range(1,len(layers)):
            _layer = layers[_]
            _name = _layer['name']
            _label = _layer['output_label']
            _data_show = _layer[_label]
            r=s.run(_data_show,feed_dict=dic_to_feed)
            print(_name,_label,r.shape)
        _,cross,acc=s.run([train_step,cross_entropy,accuracy],feed_dict=dic_to_feed)
        print("First batch evaluation using the training set, batch 0 ","Loss:",cross,"Accuracy:",acc)

        return_dic = {}
        if only_feed_forward == False:
            train_total_acc = []
            train_total_cross = []
            dic_to_feed = {x:_x_batch,y_:_y_batch}#,_layer['keep_prob']:1}
            for _layer in layers:
                if 'drop_out_bool' in _layer.keys():
                    if _layer['drop_out_bool'] == True:
                        dic_to_feed[_layer['keep_prob']]=_layer['keep_prob_train']
                        #dic_to_feed[_layer['keep_prob']]=1.0
            for itern in range(0,iters):
                if batch_proc == False:
                    #dic_to_feed = {x:_x_batch,y_:_y_batch,_layer['keep_prob']:_layer['keep_prob_train']}
                    _,cross,acc=s.run([train_step,cross_entropy,accuracy],feed_dict=dic_to_feed)
                    print("iter:",itern,"Loss:",cross,"Accuracy:",acc)
                else:
                    train_batch_acc = []
                    train_batch_cross = []
                    for bn,batch_n in enumerate(train_batch):
                        x_batch,y_batch=batch_n[0],batch_n[1]
                        dic_to_feed[x]=x_batch
                        dic_to_feed[y_]=y_batch
                        _,cross,acc=s.run([train_step,cross_entropy,accuracy],feed_dict=dic_to_feed)
                        print("iter:",itern,"batch:",bn,"Loss:",cross,"Accuracy:",acc)
                        train_batch_acc.append(acc)
                        train_batch_cross.append(cross)
                        train_total_acc.append(acc)
                        train_total_cross.append(cross)
                    np_train_batch_acc = np.asarray(train_batch_acc)
                    print("Train batch mean",np_train_batch_acc.mean(),"min:",np_train_batch_acc.min(),"max",np_train_batch_acc.max())
            np_train_acc = np.asarray(train_total_acc[len(train_batch)*(iters-1):])
            print("Train last iter mean",np_train_acc.mean(),"min:",np_train_acc.min(),"max",np_train_acc.max())
            if save_model ==True:
                print("Saving model in:",model_file)
                saving_model = saver.save(s, model_file)
            return_dic['train_acc']=train_total_acc.copy()
            return_dic['train_cross']=train_total_cross.copy()
        else:
            for _layer in layers:
                if 'drop_out_bool' in _layer.keys():
                    if _layer['drop_out_bool'] == True:
                        dic_to_feed[_layer['keep_prob']]=_layer['keep_prob_train']
            if batch_proc == False:
                dic_to_feed[x]=_x_batch
                dic_to_feed[y_]=_y_batch
                #dic_to_feed = {x:_x_batch,y_:_y_batch,_layer['keep_prob']:_layer['keep_prob_train']}
                _,cross,acc=s.run([train_step,cross_entropy,accuracy],feed_dict=dic_to_feed)
                print("Loss:",cross,"Accuracy:",acc)
            else:
                if test_batch_bool == True:
                    feed_batch = test_batch
                    print("Evaluating using test batch")
                else:
                    feed_batch = train_batch
                    print("Evaluating using train batch")
                total_accuracy = []
                total_loss = []
                for bn,batch_n in enumerate(feed_batch):
                    x_batch,y_batch=batch_n[0],batch_n[1]
                    dic_to_feed[x]=x_batch
                    dic_to_feed[y_]=y_batch
                    #dic_to_feed = {x:x_batch,y_:y_batch,_layer['keep_prob']:1}
                    _,cross,acc=s.run([train_step,cross_entropy,accuracy],feed_dict=dic_to_feed)
                    total_accuracy.append(acc)
                    total_loss.append(cross)
                    print("batch:",bn,"Loss:",cross,"Accuracy:",acc)
                np_acc = np.asarray(total_accuracy)
                print("Test Accuracy mean:",np_acc.mean(),"max:",np_acc.max(),"min:",np_acc.min())
                return_dic['test_acc']=total_accuracy.copy()
                return_dic['test_cross']=total_loss.copy()
    print("Done")
    return return_dic.copy()

