import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import smtplib
import socket
from time import time
def create_graph_bools(train_batch,layers,test_batch,mode='train',first_run=True,
	stddev_n = 0.1, learning_rate = 1e-4,iters=1,min_train_loss=1e-8,
    model_file='test_model',batch_proc=True, 
	width=256,height=256,input_mode='CNN',
	reduce_noise=False,rn_shift=0.15,rn_magnitude=0.8,
    get_deconv=False,deconv_layer='CV2',deconv_val = 'conv',
    auto_test={'bool':False,'test_interval':2,'stop_max':True,'max_iters':10,
    'fp_inc':2,'fp_inc':2,'train_rate_stop':1e-8,'restore_session':False}):
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

    train_stats = []
    test_stats = []
    if auto_test['bool']==False:
        create_graph_return = create_graph(train_batch,layers=layers,test_batch=test_batch,
                width=width,height=height, test_batch_bool=test_batch_bool,only_feed_forward=only_feed_forward,
                restore_session = restore_session, save_model = save_model, stddev_n = stddev_n, 
                learning_rate = learning_rate,iters=iters,min_train_loss=min_train_loss,model_file=model_file,input_mode=input_mode,
                reduce_noise=reduce_noise,rn_shift=rn_shift,rn_magnitude=rn_magnitude,
                get_deconv=get_deconv,deconv_layer=deconv_layer,deconv_val = deconv_val)
    else:

        learning_rate_down = learning_rate
        print("Auto test start")
        train_interval = auto_test['max_iters']//auto_test['test_interval']
        def auto_test_module(learning_rate_down=learning_rate,fpx_base=0,fnx_base=0,giter=0,restore=False):
                test_layers = []
                for layer in layers:
                    test_layers.append(layer.copy())
                train_layers = []
                for layer in layers:
                    train_layers.append(layer.copy())

                
                test_batch_bool=False
                save_model = True
                only_feed_forward=False
                if restore==False:
                    if giter ==0:
                        restore_session = False
                    else:
                        restore_session = True
                else:
                    restore_session = True
                iters=auto_test['test_interval']
                create_graph_return = create_graph(train_batch,layers=train_layers,test_batch=test_batch,
                    width=width,height=height, test_batch_bool=test_batch_bool,only_feed_forward=only_feed_forward,
                    restore_session = restore_session, save_model = save_model, stddev_n = stddev_n, 
                    learning_rate = learning_rate_down,iters=iters,min_train_loss=min_train_loss,model_file=model_file,input_mode=input_mode,
                    reduce_noise=reduce_noise,rn_shift=rn_shift,rn_magnitude=rn_magnitude,
                    get_deconv=get_deconv,deconv_layer=deconv_layer,deconv_val = deconv_val)
            #{'tp':tpo,'tn':tno,'fp':fpo,'fn':tno,'sensitivity':o_sensitivity,'specificity':o_specificity}

                local_test_layers = []
                for layer in test_layers:
                    local_test_layers.append(layer.copy())
                test_batch_bool=True
                only_feed_forward=True
                restore_session = True
                save_model = False
                create_graph_return_test = create_graph(train_batch,layers=test_layers,test_batch=test_batch,
                    width=width,height=height, test_batch_bool=test_batch_bool,only_feed_forward=only_feed_forward,
                    restore_session = restore_session, save_model = save_model, stddev_n = stddev_n, 
                    learning_rate = learning_rate_down,iters=iters,min_train_loss=min_train_loss,model_file=model_file,input_mode=input_mode,
                    reduce_noise=reduce_noise,rn_shift=rn_shift,rn_magnitude=rn_magnitude,
                    get_deconv=get_deconv,deconv_layer=deconv_layer,deconv_val = deconv_val)
                fpx_test = create_graph_return_test['stats']['fp']
                fnx_test = create_graph_return_test['stats']['fn']
                if  giter==0:
                    fpx_base=fpx_test
                    fnx_base=fnx_test
                else:
                    fpx_diff = fpx_test - fpx_base
                    fnx_diff = fnx_test - fnx_base
                    print("previous fp:",fpx_base,"new fp:",fpx_test)
                    print("previous fn:",fnx_base,"new fn:",fnx_test)
                    if (fpx_diff > auto_test['fp_inc']) or (fnx_diff > auto_test['fn_inc']):
                        learning_rate_down = learning_rate_down/10
                        print("New learning_rate:",learning_rate_down)
                        fpx_base=fpx_test
                        fnx_base=fnx_test   
                return [learning_rate_down,fpx_base,fnx_base,create_graph_return_test['stats'].copy(),create_graph_return_test['training_message']]
            
        def loop_stat(list_stat,lr):
            st=""
            for stx,lrx in zip(list_stat,lr):
                st+=str(stx)+" | learning rate: "+str(lrx)+"\n"
            return st
        
        learning_rate_list=[]
        err_message=""
        giter=0
        fpx_base,fnx_base=0,0
        st = time()
        if auto_test['stop_max']==True:
            for giter in range(giter,train_interval):
                print("Global iter:",giter,"/",train_interval)
                print(learning_rate_down,auto_test['train_rate_stop'])
                learning_rate_down,fpx_base,fnx_base,stats,train_msg = auto_test_module(learning_rate_down,fpx_base,fnx_base,giter=giter,restore=auto_test['restore_session'])
                if train_msg=='min_train_loss':
                    print("End by training loss")
                    err_message="End by training loss"
                    break;
                if learning_rate_down < auto_test['train_rate_stop']:
                    print("End by learning rate")
                    break;
                test_stats.append(stats)
                learning_rate_list.append(learning_rate_down)
                if (stats['tp']==0) and (stats['fp']==0):
                    print("Error")
                    err_message="Loss: nan found"
                    break;
        else:
            while (learning_rate_down>auto_test['train_rate_stop']):
                print("Global iter:",giter)
                print(learning_rate_down,auto_test['train_rate_stop'])
                learning_rate_down,fpx_base,fnx_base,stats,train_msg = auto_test_module(learning_rate_down,fpx_base,fnx_base,giter=giter,restore=auto_test['restore_session'])
                test_stats.append(stats)
                learning_rate_list.append(learning_rate_down)
                giter+=1
                if train_msg=='min_train_loss':
                    print("End by training loss")
                    err_message="End by training loss"
                    break;
                if (stats['tp']==0) and (stats['fp']==0):
                    print("Error")
                    err_message="Loss: nan found"
                    break;
        email_keys = ['send_email_bool','email_origin','email_pass','email_destination']
        time_taken = np.ceil((time() - st)/60)
        if bool(list(filter(lambda x: x in list(auto_test.keys()),email_keys))):
            if auto_test['send_email_bool']==True:
                content="stats:\n "+loop_stat(test_stats,learning_rate_list)+" \n "+err_message+"\n"
                #send_mail(email_origin,email_destination,email_pass,subject="Test report",content="Test")
                email_org = auto_test['email_origin']
                email_pass = auto_test['email_pass']
                email_dest = auto_test['email_destination']
                email_subj = "Model "+model_file+" evaluation completed: "
                email_subj += socket.gethostname()+" Max-iter "+str(auto_test['stop_max'])
                email_subj += "iters: "+str(giter)+ " Minutes taken: "+str(time_taken)
                send_mail(email_org,email_dest,email_pass,subject=email_subj,content=content)
        create_graph_return={'stats':10}
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
        vc_input_features = get_previous_features(input_layer)
        iDic['input_layer'] = tf.reshape(input_layer, [-1, vc_input_features])

        if 'x2_bool' in iDic.keys():
            if iDic['x2_bool'] == True:
                iDic['x2'] = tf.placeholder(tf.float32, shape=[None, iDic['x2_features']])
                input_layer_mod = tf.concat([iDic['input_layer'],iDic['x2']],1)
            else:
                input_layer_mod = iDic['input_layer']
        else:
            input_layer_mod = iDic['input_layer']
        iDic['input_features'] = get_previous_features(input_layer_mod)

        iDic['W'] = tf.Variable(tf.truncated_normal([iDic['input_features'], iDic['neurons']], stddev=stddev_n),name=W_name)
        iDic['b'] = tf.Variable(tf.constant(stddev_n, shape=[iDic['neurons']]),name=b_name) 
        iDic['Z'] = tf.matmul(input_layer_mod, iDic['W'],name=Z_name) + iDic['b']
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
        if 'x2_bool' in iDic.keys():
            if iDic['x2_bool'] == True:
                iDic['x2'] = tf.placeholder(tf.float32, shape=[None, iDic['x2_features']])
                input_layer_mod = tf.concat([input_layer,iDic['x2']],1)
            else:
                input_layer_mod = input_layer
        else:
            input_layer_mod = input_layer
        iDic['input_features'] = get_previous_features(input_layer_mod)
        iDic['W'] = tf.Variable(tf.truncated_normal([iDic['input_features'], iDic['neurons']], stddev=stddev_n),name=W_name)
        iDic['b'] = tf.Variable(tf.constant(stddev_n, shape=[iDic['neurons']]),name=b_name) 
        iDic['Z'] = tf.matmul(input_layer_mod, iDic['W'],name=Z_name) + iDic['b']
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
                 stddev_n = 0.1, learning_rate = 1e-4,iters=4,min_train_loss=1e-8,
                 model_file='CNN_model',
                 input_mode='CNN',wf=0.8,ws=0.3,batch_size=8,
                 reduce_noise=False,rn_shift=0.15,rn_magnitude=0.8,
                 get_deconv=True,deconv_layer='CV2',deconv_val = 'conv'):
    _x_batch = train_batch[0][0]
    _y_batch = train_batch[0][1]
    class_output = _y_batch.shape[1]
    
    tf.reset_default_graph()

    x_flat = width * height
    x = tf.placeholder(tf.float32, shape=[None, x_flat])
    y_ = tf.placeholder(tf.float32, shape=[None, class_output])

##################Add Noise

    amp = tf.constant(255.0)
    epsilon_e = 1e-4
    Wf = tf.Variable(rn_magnitude)
    Ws = tf.Variable(rn_shift)
    pow_lim = 20.0
    gen_f = pow_lim/amp

##########################





    input_layer_dic = {'type':'input_layer'}
    if input_mode=='CNN':
        x_image = tf.reshape(x, [-1,width,height,1])
        #input_layer_dic['output_label']='x_image'
        #input_layer_dic['x_image']=x_image
        
        if reduce_noise==True:
            
            f1 = tf.constant([[1.0,0,-1.0],[1.0,0,-1.0],[1.0,0,-1.0]])
            f2 = tf.transpose(f1)
            f3 = tf.constant([[1.0,0,1.0],[0,0,0],[-1.0,0,-1.0]])
            f4 = tf.transpose(f3)
            f1r = tf.reshape(f1,[3,3,1,1])
            f2r = tf.reshape(f2,[3,3,1,1])
            f3r = tf.reshape(f3,[3,3,1,1])
            f4r = tf.reshape(f4,[3,3,1,1])
            cv1 = tf.abs(tf.nn.conv2d(x_image, f1r, strides=[1,1,1,1], padding='SAME'))
            cv2 = tf.abs(tf.nn.conv2d(x_image, f2r, strides=[1,1,1,1], padding='SAME'))
            cv3 = tf.abs(tf.nn.conv2d(x_image, f3r, strides=[1,1,1,1], padding='SAME'))
            cv4 = tf.abs(tf.nn.conv2d(x_image, f4r, strides=[1,1,1,1], padding='SAME'))
            cvM = cv1+cv2+cv3+cv4
            cvMn = cvM/(tf.reduce_max(cvM)-tf.reduce_min(cvM))
            powf = gen_f*Wf*(amp*cvMn-amp*Ws+epsilon_e)
            noise_red = tf.divide(1.0,tf.add(1.0,tf.exp(-powf)))
            input_layer_dic['output_label']='x_image'
            input_layer_dic['x_image']=noise_red
        else:
            input_layer_dic['output_label']='x_image'
            input_layer_dic['x_image']=x_image
        
    elif input_mode=='noise':
        #powf = gen_f*Wf*(amp*x-amp*Ws+epsilon_e)
        #noise_red = tf.divide(1.0,tf.add(1.0,tf.exp(-powf)))
        x_image = tf.reshape(noise_red, [-1,width,height,1])
        input_layer_dic['output_label']='x_image'
        input_layer_dic['x_image']=x_image
    else:
        input_layer_dic['output_label']='x'
        input_layer_dic['x']=x

    layers.insert(0,input_layer_dic)

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
    yi = tf.argmax(y_,1)
    yp = tf.argmax(y_CNN,1)

    tpi = yp*yi
    tp = tf.reduce_sum(tf.cast(tf.greater(tpi,0),tf.int32))
    
    fni = yi-yp
    fn = tf.reduce_sum(tf.cast(tf.greater(fni,0),tf.int32))
    
    sensitivity = tp/(fn+tp)
    
    tni = yi+yp
    tn = tf.reduce_sum(tf.cast(tf.equal(tni,0),tf.int32))
    
    fpi = yp - yi
    fp = tf.reduce_sum(tf.cast(tf.greater(fpi,0),tf.int32))
    
    specificity = tn/(tn+fp)
    accuracy = (tn+tp)/(tn+tp+fn+fp)
    correct_prediction = tf.equal(tf.argmax(y_CNN,1), tf.argmax(y_,1))
    
    acc_no_mean = tf.cast(correct_prediction, tf.float32)

    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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
            if 'x2_bool' in _layer.keys():
                if _layer['x2_bool'] == True:
                    dic_to_feed[_layer['x2']]=hist_batch(_x_batch,bins=_layer['x2_features'])
        for _ in range(1,len(layers)):
            _layer = layers[_]
            _name = _layer['name']
            _label = _layer['output_label']
            _data_show = _layer[_label]

            r=s.run(_data_show,feed_dict=dic_to_feed)
            print(_name,_label,r.shape)
        _,cross,acc=s.run([train_step,cross_entropy,accuracy],feed_dict=dic_to_feed)
        print("First batch evaluation using the training set, batch 0 ","Loss:",cross,"Accuracy:",acc)

        return_dic = {'training_message':None}
        if only_feed_forward == False:
            train_total_acc = []
            train_total_cross = []
            dic_to_feed = {x:_x_batch,y_:_y_batch}#,_layer['keep_prob']:1}
            for _layer in layers:
                if 'drop_out_bool' in _layer.keys():
                    if _layer['drop_out_bool'] == True:
                        dic_to_feed[_layer['keep_prob']]=_layer['keep_prob_train']
                        #dic_to_feed[_layer['keep_prob']]=1.0
                if 'x2_bool' in _layer.keys():
                    if _layer['x2_bool'] == True:
                        dic_to_feed[_layer['x2']]=hist_batch(_x_batch,bins=_layer['x2_features'])
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
                        for _layer in layers:
                            if 'x2_bool' in _layer.keys():
                                if _layer['x2_bool'] == True:
                                    dic_to_feed[_layer['x2']]=hist_batch(x_batch,bins=_layer['x2_features'])
                        _,cross,acc=s.run([train_step,cross_entropy,accuracy],feed_dict=dic_to_feed)
                        print("iter:",itern,"batch:",bn,"Loss:",cross,"Accuracy:",acc)
                        train_batch_acc.append(acc)
                        train_batch_cross.append(cross)
                        train_total_acc.append(acc)
                        train_total_cross.append(cross)
                        #if cross<min_train_loss:
                        if str(cross) == "nan":
                            break;
                    if  str(cross) != "nan":
                        np_train_batch_acc = np.asarray(train_batch_acc)
                        print("Train batch mean",np_train_batch_acc.mean(),"min:",np_train_batch_acc.min(),"max",np_train_batch_acc.max())
                #if cross<min_train_loss:
                if str(cross) == "nan":
                    print("Stopped by learning convergence. Loss=nan")
                    return_dic['training_message']="min_train_loss"
                    break;
            if  str(cross) != "nan":
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
                        dic_to_feed[_layer['keep_prob']]=test_keep_prob
            if batch_proc == False:
                cross,acc=s.run([cross_entropy,accuracy],feed_dict=dic_to_feed)
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
                global_acc = []
                pred_list = []
                tf_pn = []
                for bn,batch_n in enumerate(feed_batch):
                    x_batch,y_batch=batch_n[0],batch_n[1]
                    dic_to_feed[x]=x_batch
                    dic_to_feed[y_]=y_batch
                    for _layer in layers:
                        if 'x2_bool' in _layer.keys():
                            if _layer['x2_bool'] == True:
                                dic_to_feed[_layer['x2']]=hist_batch(x_batch,bins=_layer['x2_features'])
                    cross,acc,gacc,y_pred,tpo,tno,fpo,fno=s.run([cross_entropy,accuracy,acc_no_mean,y_CNN,tp,tn,fp,fn],feed_dict=dic_to_feed)
                    #get_deconv=True
                    #deconv_layer='CV2'
                    #deconv_val = 'conv'
                    if get_deconv==True:
                        for _layer in layers:
                            if 'name' in _layer.keys():
                                if deconv_layer == _layer['name']:
                                    return_dic['deconv']=s.run(_layer[deconv_val],feed_dict=dic_to_feed)
                    total_accuracy.append(acc)
                    total_loss.append(cross)
                    global_acc.append(gacc)
                    pred_list.append(y_pred)
                    tf_pn.append([tpo,tno,fpo,fno])
                    print("batch:",bn,"Loss:",cross,"Accuracy:",acc)
                np_acc = np.asarray(total_accuracy)
                tf_pn_np = np.asarray(tf_pn)
                tpo,tno,fpo,fno = np.sum(tf_pn_np,0).tolist()
                print("tp",tpo,"tn",tno,"fp",fpo,"fn",fno)
                o_sensitivity = tpo/(fno+tpo+0.00001)
                o_specificity = tno/(tno+fpo+0.00001)
                print("Test Accuracy mean:",np_acc.mean(),"min:",np_acc.min(),"max:",np_acc.max(),"Sensitivity:",o_sensitivity,"Specificity:",o_specificity)
                return_dic['test_acc']=total_accuracy.copy()
                return_dic['test_g_acc']=global_acc.copy()
                return_dic['y_pred']=pred_list.copy()
                return_dic['test_cross']=total_loss.copy()
                return_dic['stats']={'tp':tpo,'tn':tno,'fp':fpo,'fn':fno,'sensitivity':o_sensitivity,'specificity':o_specificity}
    print("Done")
    return return_dic.copy()
## Some utils:
def send_mail(email_origin,email_destination,email_pass,subject="Test report",content="Test"):
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    #Next, log in to the server
    server.login(email_origin,email_pass)
    msg = "Subject:"+subject+" \n\n "+content+"\n" # The /n separates the message from the headers
    server.sendmail(email_origin,email_destination, msg)
def hist_batch(iNp,bins=256):
    rows = iNp.shape[0]
    for _ in range(0,rows):
        if _ == 0:
            hist_np = np.histogram(iNp[_,:],bins=bins)[0]
            hist_np = np.reshape(hist_np,(1,hist_np.shape[0]))
        else:
            hist_np_t = np.histogram(iNp[_,:],bins=bins)[0]
            hist_np_t = np.reshape(hist_np_t,(1,hist_np_t.shape[0]))
            hist_np = np.concatenate((hist_np,hist_np_t),0)
    return hist_np.copy()
def border_filter_pad_batch(iNp):
    rows = iNp.shape[0]
    cols = iNp.shape[1]
    sq_shape = int(np.sqrt(cols))
    for _ in range(0,rows):
        imgx_r = np.reshape(iNp[_,:],(sq_shape,sq_shape))
        if _ == 0:
            output = np.reshape(borderFilterPad(imgx_r),(1,cols))
        else:
            output_t = np.reshape(borderFilterPad(imgx_r),(1,cols))
            output = np.concatenate((output,output_t),0)
    return output.copy()
def borderFilterPad(np_img):
 np_imgx=np_img.copy().astype('float')
 npimgs=np_imgx.shape
 imgx=npimgs[0]
 imgy=npimgs[1]
 dx_img=np.zeros((imgx,imgy))
 for y in range(1,imgy-1):
  for x in range(1,imgx-1):
   dx_img[x,y]=np.abs(np_imgx[x+1,y]-np_imgx[x-1,y]) + np.abs(np_imgx[x,y+1]-np_imgx[x,y-1]) + np.abs(np_imgx[x+1,y+1]-np_imgx[x-1,y-1]) + np.abs(np_imgx[x-1,y+1]-np_imgx[x+1,y+1])
 total_diff = ( dx_img*255 )/( dx_img.max() - dx_img.min() )
 return total_diff.copy()