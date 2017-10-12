
# Create Convolutional Neural Networks in tensorflow with ease
With this library **cnn_modeling.py** you can create "n" layers of convolutional neural networks **(CNN)** very quickly.  You just need to creat a dictionary for every layer with its hyperparameters and add them to a list. Afterwards use this list as input of the library to create a graph.
## Features/Constraints:
* All the activation functions are ReLU
* Enable/Disable Batch normalization before ReLU activation for every layer.
* Enable/Disable max pooling 
* Stack convolutional layers
* Stack Fully connected  layers 
* Save/restore the model to a given file name
* Get the training/test scores.
* Get the Weights and biases after training/testing 
* Mini-batch processing.
* Enable/Disable dropout for the Fully connected layers. (Keep_ prob is set to one during the test time)
* Adam optimizer used.
* An additional function to transform a numpy array with shape **m** by **n+1** to a mini-batch list of the desired size with balanced clases. But, **Important constrain:** for binary classification only. (This constrain is only for this last additional function).
* Worked for grayscale images. If you want to test if with 3 colors or more channels images see the notes at the end of this page.  
* Create Deep Neural Networks without convolution layers by selecting a desired mode.  
* Update the learning rate based on the performance of the trained model in the dev-set: There is an option to automate the training-testing processes. After a desired amount of iterations, period **T**, the model is tested in the testing-batch. The amount of False Positives (**fp**) and False Negatives (**fn**) are acquired at this **T0**. At **T1** the **fp1** and **tn1** are compared with **fn0** and **fp0**. If **fp1**-**fp0** > an **allowed fp increase (AFpI)**  or **fn1**-**fn0**>**fn (AFnI)** the learning rate is decreased by a factor of 10. If a minimum learning rate is reached the learning process stops.   
* Automate the training-testing process, for **n** desired iterations and decrease the learning-rate as previously described.
* For the automated training-testing set up receive an e-mail when the learning process ends. It supports gmail accounts so far.  
**Miscellaneous**:  
* Feed to the Fully connected layers a histogram of the input layer in parallel with the convolutional layers.  
* Reduce noise in a pre-processing step.  
* Visualize the image result of the convolutions or max-pooling

### Future improvements:
* Exponential weighted average for batch mean/std for the testing time.
* Predictions for a single example or batch/mini-batch input.
	
## Demo:
```python
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import cnn_modeling 
import imp
imp.reload(cnn_modeling)
```


    <module 'cnn_modeling' from '/home/.../cnn_modeling.py'>



```python
numpy_dir = "NumpyData/"
complete_train = numpy_dir+'images_complete_train_256x256_7494.npy'
complete_test = numpy_dir+'images_complete_test_256x256_4394.npy'
```


```python
help(cnn_modeling)
```

    Help on module cnn_modeling:
    
    NAME
        cnn_modeling
    
    FUNCTIONS
        balance_positive_negative(iNp, iBatchSize=256, v=False)
        
        create_conv(iDic, input_layer, iName, prev_dic, stddev_n=0.1, norm_offset=0, norm_scale=1, norm_epsilon=1e-06)
        
        create_graph(train_batch, layers, test_batch=None, width=256, height=256, batch_proc=True, test_batch_bool=False, restore_session=False, save_model=False, only_feed_forward=False, stddev_n=0.1, learning_rate=0.0001, iters=4, model_file='CNN_model')
        
        get_previous_features(i_layer)
        
        plot_list(iList, figsize=(10, 8), title='Loss/Eff', xlabel='Iters', ylabel='Loss/Eff')
    
    FILE
        /home/.../cnn_modeling.py

Here we transform a nmpy array with the shape **m** x **n+1**. m rows as examples and n features +1 column for the class. We create balanced batches with the size **iBatchsize=256**, with 128 positive (1) and 128 negative (0). 
```python
train_numpy = np.load(complete_train)
train_batch = cnn_modeling.balance_positive_negative(iNp=train_numpy,iBatchSize=256)
print("Done",len(train_batch))
```
    Done 19

```python
test_numpy = np.load(complete_test)
test_batch = cnn_modeling.balance_positive_negative(iNp=test_numpy,iBatchSize=256)
print("Done",len(test_batch))
```

    Done 15


For this demo we will take just 3 mini-batches from the train and test sets.
```python
mini_train_batch = train_batch[:4]
mini_test_batch = test_batch[:4]
print(len(mini_train_batch),len(mini_test_batch))
```

    4 4


## Testing a feed forward without mini-batch

We make this test first to evaluate if the graph is created without errors


```python
CV1 = { 'type':'CV', 'depth':8, 'filter_w':11, 'filter_stride':[1,2,2,1], 'norm_bool':True,
    'max_pooling':True, 'max_pool_mask':[1,3,3,1], 'max_pool_stride':[1,1,1,1], 'padding':'SAME',
       'name':'CV1'} 

CV2 ={'type':'CV',  'depth':8, 'filter_w':11, 'filter_stride':[1,2,2,1], 'norm_bool':True,
      'max_pooling':True, 'max_pool_mask':[1,3,3,1], 'max_pool_stride':[1,1,1,1], 'padding':'SAME',
      'name':'CV2' } 

CV2FC={'type':'CV2FC', 'neurons':1024, 'norm_bool':True, 'name':'CV2FC'} 
FC1 = { 'type':'FC', 'neurons':1024, 'norm_bool':True, 'name':'FC1'} 
FC2 = { 'type':'FC', 'neurons':1024, 'norm_bool':True, 'name':'FC2',
       'drop_out_bool':True, 'keep_prob_train':0.5} 

layers = [CV1,CV2,CV2FC,FC1,FC2]


stats_dic = cnn_modeling.create_graph(mini_train_batch,layers=layers,test_batch=mini_test_batch,
            width=256,height=256, 
             batch_proc=False, 
            test_batch_bool=False, 
            only_feed_forward=True,
             restore_session = False, save_model = False, 
             stddev_n = 0.1, learning_rate = 1e-4,iters=1,model_file='test_model')
```

    Creating layer: CV1
    Creating layer: CV2
    Creating layer: CV2FC
    Creating layer: FC1
    Creating layer: FC2
    Starting session
    CV1 max (256, 128, 128, 8)
    CV2 max (256, 64, 64, 8)
    CV2FC relu (256, 1024)
    FC1 relu (256, 1024)
    FC2 dropout (256, 1024)
    First batch test  Loss: 0.870873 Accuracy: 0.570312
    Loss: 0.700965 Accuracy: 0.6875
    Done


## Testing a feed forward with mini-batch processing 

As the previous test, we evaluate here if our batch processing is done correctly


```python
CV1 = { 'type':'CV', 'depth':8, 'filter_w':11, 'filter_stride':[1,2,2,1], 'norm_bool':True,
    'max_pooling':True, 'max_pool_mask':[1,3,3,1], 'max_pool_stride':[1,1,1,1], 'padding':'SAME',
       'name':'CV1'} 

CV2 ={'type':'CV',  'depth':8, 'filter_w':11, 'filter_stride':[1,2,2,1], 'norm_bool':True,
      'max_pooling':True, 'max_pool_mask':[1,3,3,1], 'max_pool_stride':[1,1,1,1], 'padding':'SAME',
      'name':'CV2' } 

CV2FC={'type':'CV2FC', 'neurons':1024, 'norm_bool':True, 'name':'CV2FC'} 
FC1 = { 'type':'FC', 'neurons':1024, 'norm_bool':True, 'name':'FC1'} 
FC2 = { 'type':'FC', 'neurons':1024, 'norm_bool':True, 'name':'FC2',
       'drop_out_bool':True, 'keep_prob_train':0.5} 

layers = [CV1,CV2,CV2FC,FC1,FC2]


stats_dic = cnn_modeling.create_graph(mini_train_batch,layers=layers,test_batch=mini_test_batch,
            width=256,height=256, 
             batch_proc=True, 
            test_batch_bool=False, 
            only_feed_forward=True,
             restore_session = False, save_model = False, 
             stddev_n = 0.1, learning_rate = 1e-4,iters=1,model_file='test_model')
```

    Creating layer: CV1
    Creating layer: CV2
    Creating layer: CV2FC
    Creating layer: FC1
    Creating layer: FC2
    Starting session
    CV1 max (256, 128, 128, 8)
    CV2 max (256, 64, 64, 8)
    CV2FC relu (256, 1024)
    FC1 relu (256, 1024)
    FC2 dropout (256, 1024)
    First batch test  Loss: 0.799795 Accuracy: 0.660156
    Evaluating using train batch
    batch: 0 Loss: 0.339041 Accuracy: 0.84375
    batch: 1 Loss: 0.931047 Accuracy: 0.609375
    batch: 2 Loss: 0.77167 Accuracy: 0.640625
    batch: 3 Loss: 0.698236 Accuracy: 0.671875
    Accuracy mean: 0.691406 max: 0.84375 min: 0.609375
    Done


## Start the training and saving the model for later training

For the first training we need to specify **save_model=True** but having **restore_session=False**. It is required to specify the file name where the model will be saved in by specifying the variable **model_file**. For this example we set up **2** iterations for the mini-batch training.


```python
CV1 = { 'type':'CV', 'depth':8, 'filter_w':11, 'filter_stride':[1,2,2,1], 'norm_bool':True,
    'max_pooling':True, 'max_pool_mask':[1,3,3,1], 'max_pool_stride':[1,1,1,1], 'padding':'SAME',
       'name':'CV1'} 

CV2 ={'type':'CV',  'depth':8, 'filter_w':11, 'filter_stride':[1,2,2,1], 'norm_bool':True,
      'max_pooling':True, 'max_pool_mask':[1,3,3,1], 'max_pool_stride':[1,1,1,1], 'padding':'SAME',
      'name':'CV2' } 

CV2FC={'type':'CV2FC', 'neurons':1024, 'norm_bool':True, 'name':'CV2FC'} 
FC1 = { 'type':'FC', 'neurons':1024, 'norm_bool':True, 'name':'FC1'} 
FC2 = { 'type':'FC', 'neurons':1024, 'norm_bool':True, 'name':'FC2',
       'drop_out_bool':True, 'keep_prob_train':0.5} 

layers = [CV1,CV2,CV2FC,FC1,FC2]


stats_dic = cnn_modeling.create_graph(mini_train_batch,layers=layers,test_batch=mini_test_batch,
            width=256,height=256, 
             batch_proc=True, 
            test_batch_bool=False, 
            only_feed_forward=False,
             restore_session = False, save_model = True, 
             stddev_n = 0.1, learning_rate = 1e-4,iters=2,model_file='test_model')
```

    Creating layer: CV1
    Creating layer: CV2
    Creating layer: CV2FC
    Creating layer: FC1
    Creating layer: FC2
    Starting session
    CV1 max (256, 128, 128, 8)
    CV2 max (256, 64, 64, 8)
    CV2FC relu (256, 1024)
    FC1 relu (256, 1024)
    FC2 dropout (256, 1024)
    First batch test  Loss: 2.66882 Accuracy: 0.5
    iter: 0 batch: 0 Loss: 1.72455 Accuracy: 0.601562
    iter: 0 batch: 1 Loss: 2.3825 Accuracy: 0.558594
    iter: 0 batch: 2 Loss: 2.53418 Accuracy: 0.570312
    iter: 0 batch: 3 Loss: 2.02975 Accuracy: 0.589844
    Train batch mean 0.580078 min: 0.558594 max 0.601562
    iter: 1 batch: 0 Loss: 1.21905 Accuracy: 0.710938
    iter: 1 batch: 1 Loss: 1.80638 Accuracy: 0.625
    iter: 1 batch: 2 Loss: 1.73706 Accuracy: 0.632812
    iter: 1 batch: 3 Loss: 1.67835 Accuracy: 0.605469
    Train batch mean 0.643555 min: 0.605469 max 0.710938
    Train last iter mean 0.643555 min: 0.605469 max 0.710938
    Saving model in: test_model
    Done


The function returns the value for the cross entropy and train accuracy during the training


```python
print(stats_dic.keys())
```

    dict_keys(['train_acc', 'train_cross'])


To visualize the progress it is possible to use the function output and use **cnn_modeling.plot_list** to plot it.


```python
cnn_modeling.plot_list(stats_dic['train_acc'], figsize=(8, 6), title='Train accuracy', xlabel='Iters', ylabel='Acc')
cnn_modeling.plot_list(stats_dic['train_cross'], figsize=(8, 6), title='Test Cross entropy', xlabel='Iters', ylabel='Loss')
```


![png](output_18_0.png)



![png](output_18_1.png)


## Restore the model and continue training

The only difference with the previous step is to set **restore_session=True**


```python
CV1 = { 'type':'CV', 'depth':8, 'filter_w':11, 'filter_stride':[1,2,2,1], 'norm_bool':True,
    'max_pooling':True, 'max_pool_mask':[1,3,3,1], 'max_pool_stride':[1,1,1,1], 'padding':'SAME',
       'name':'CV1'} 

CV2 ={'type':'CV',  'depth':8, 'filter_w':11, 'filter_stride':[1,2,2,1], 'norm_bool':True,
      'max_pooling':True, 'max_pool_mask':[1,3,3,1], 'max_pool_stride':[1,1,1,1], 'padding':'SAME',
      'name':'CV2' } 

CV2FC={'type':'CV2FC', 'neurons':1024, 'norm_bool':True, 'name':'CV2FC'} 
FC1 = { 'type':'FC', 'neurons':1024, 'norm_bool':True, 'name':'FC1'} 
FC2 = { 'type':'FC', 'neurons':1024, 'norm_bool':True, 'name':'FC2',
       'drop_out_bool':True, 'keep_prob_train':0.5} 

layers = [CV1,CV2,CV2FC,FC1,FC2]


stats_dic_restore = cnn_modeling.create_graph(mini_train_batch,layers=layers,test_batch=mini_test_batch,
            width=256,height=256, 
             batch_proc=True, 
            test_batch_bool=False, 
            only_feed_forward=False,
             restore_session = True, save_model = True, 
             stddev_n = 0.1, learning_rate = 1e-4,iters=2,model_file='test_model')
```

    Creating layer: CV1
    Creating layer: CV2
    Creating layer: CV2FC
    Creating layer: FC1
    Creating layer: FC2
    Starting session
    INFO:tensorflow:Restoring parameters from ./test_model
    CV1 max (256, 128, 128, 8)
    CV2 max (256, 64, 64, 8)
    CV2FC relu (256, 1024)
    FC1 relu (256, 1024)
    FC2 dropout (256, 1024)
    First batch test  Loss: 0.574261 Accuracy: 0.785156
    iter: 0 batch: 0 Loss: 0.621952 Accuracy: 0.800781
    iter: 0 batch: 1 Loss: 1.17566 Accuracy: 0.667969
    iter: 0 batch: 2 Loss: 1.31136 Accuracy: 0.648438
    iter: 0 batch: 3 Loss: 1.08492 Accuracy: 0.683594
    Train batch mean 0.700195 min: 0.648438 max 0.800781
    iter: 1 batch: 0 Loss: 0.449774 Accuracy: 0.824219
    iter: 1 batch: 1 Loss: 0.981902 Accuracy: 0.726562
    iter: 1 batch: 2 Loss: 0.920269 Accuracy: 0.757812
    iter: 1 batch: 3 Loss: 0.740904 Accuracy: 0.75
    Train batch mean 0.764648 min: 0.726562 max 0.824219
    Train last iter mean 0.764648 min: 0.726562 max 0.824219
    Saving model in: test_model
    Done


Add the previous stats to the new one


```python
print(len(stats_dic_restore['train_acc']))
stats_dic_restore['train_acc'].extend(stats_dic['train_acc'])
stats_dic_restore['train_cross'].extend(stats_dic['train_acc'])
print(len(stats_dic_restore['train_acc']))
```

    8
    16


Visualize the result


```python
cnn_modeling.plot_list(stats_dic_restore['train_acc'], figsize=(8, 6), title='Train accuracy', xlabel='Iters', ylabel='Acc')
cnn_modeling.plot_list(stats_dic_restore['train_cross'], figsize=(8, 6), title='Test Cross entropy', xlabel='Iters', ylabel='Loss')
```


![png](output_25_0.png)



![png](output_25_1.png)


## Restore the trained model and test it in the dev/test mini-batch set 

Use the previous setting but change **test_batch_bool=True**, **save_model=False** and **only_feed_forward=True**


```python
CV1 = { 'type':'CV', 'depth':8, 'filter_w':11, 'filter_stride':[1,2,2,1], 'norm_bool':True,
    'max_pooling':True, 'max_pool_mask':[1,3,3,1], 'max_pool_stride':[1,1,1,1], 'padding':'SAME',
       'name':'CV1'} 

CV2 ={'type':'CV',  'depth':8, 'filter_w':11, 'filter_stride':[1,2,2,1], 'norm_bool':True,
      'max_pooling':True, 'max_pool_mask':[1,3,3,1], 'max_pool_stride':[1,1,1,1], 'padding':'SAME',
      'name':'CV2' } 

CV2FC={'type':'CV2FC', 'neurons':1024, 'norm_bool':True, 'name':'CV2FC'} 
FC1 = { 'type':'FC', 'neurons':1024, 'norm_bool':True, 'name':'FC1'} 
FC2 = { 'type':'FC', 'neurons':1024, 'norm_bool':True, 'name':'FC2',
       'drop_out_bool':True, 'keep_prob_train':0.5} 

layers = [CV1,CV2,CV2FC,FC1,FC2]


stats_dic_restore = cnn_modeling.create_graph(mini_train_batch,layers=layers,test_batch=mini_test_batch,
            width=256,height=256, 
             batch_proc=True, 
            test_batch_bool=True, 
            only_feed_forward=True,
             restore_session = True, save_model = False, 
             stddev_n = 0.1, learning_rate = 1e-4,iters=2,model_file='test_model')
```

    Creating layer: CV1
    Creating layer: CV2
    Creating layer: CV2FC
    Creating layer: FC1
    Creating layer: FC2
    Starting session
    INFO:tensorflow:Restoring parameters from ./test_model
    CV1 max (256, 128, 128, 8)
    CV2 max (256, 64, 64, 8)
    CV2FC relu (256, 1024)
    FC1 relu (256, 1024)
    FC2 dropout (256, 1024)
    First batch test  Loss: 0.200401 Accuracy: 0.929688
    Evaluating using test batch
    batch: 0 Loss: 1.48273 Accuracy: 0.589844
    batch: 1 Loss: 1.41957 Accuracy: 0.613281
    batch: 2 Loss: 1.29926 Accuracy: 0.621094
    batch: 3 Loss: 1.3216 Accuracy: 0.625
    Accuracy mean: 0.612305 max: 0.625 min: 0.589844
    Done


## Try to improve the model: Add layers, modify hyperparameters

In this part we will modify some hyperparameters aiming to improve the dev/test performance.
### Training:


```python
CV1 = { 'type':'CV', 'depth':32, 'filter_w':11, 'filter_stride':[1,4,4,1], 'norm_bool':True,
    'max_pooling':True, 'max_pool_mask':[1,3,3,1], 'max_pool_stride':[1,2,2,1], 'padding':'SAME',
       'name':'CV1'} 

CV2 ={'type':'CV',  'depth':32, 'filter_w':5, 'filter_stride':[1,1,1,1], 'norm_bool':True,
      'max_pooling':True, 'max_pool_mask':[1,3,3,1], 'max_pool_stride':[1,2,2,1], 'padding':'SAME',
      'name':'CV2' } 
CV3 ={'type':'CV',  'depth':128, 'filter_w':3, 'filter_stride':[1,2,2,1], 'norm_bool':True,
      'name':'CV3' } 
CV4 ={'type':'CV',  'depth':128, 'filter_w':3, 'filter_stride':[1,2,2,1], 'norm_bool':True,
      'name':'CV4' }
CV5 ={'type':'CV',  'depth':64, 'filter_w':3, 'filter_stride':[1,2,2,1], 'norm_bool':True,
      'name':'CV5' }

CV2FC={'type':'CV2FC', 'neurons':1024, 'norm_bool':True, 'name':'CV2FC'} 
FC1 = { 'type':'FC', 'neurons':1024, 'norm_bool':True, 'name':'FC1'} 
FC2 = { 'type':'FC', 'neurons':1024, 'norm_bool':True, 'name':'FC2'} 
FC3 = { 'type':'FC', 'neurons':1024, 'norm_bool':True, 'name':'FC3',
       'drop_out_bool':True, 'keep_prob_train':0.5} 

layers = [CV1,CV2,CV3,CV4,CV5,CV2FC,FC1,FC2,FC3]


stats_dic_improve = cnn_modeling.create_graph(mini_train_batch,layers=layers,test_batch=mini_test_batch,
            width=256,height=256, 
             batch_proc=True, 
            test_batch_bool=False, 
            only_feed_forward=False,
             restore_session = False, save_model = True, 
             stddev_n = 0.1, learning_rate = 1e-4,iters=20,model_file='test_model')
```

    Creating layer: CV1
    Creating layer: CV2
    Creating layer: CV3
    Creating layer: CV4
    Creating layer: CV5
    Creating layer: CV2FC
    Creating layer: FC1
    Creating layer: FC2
    Creating layer: FC3
    Starting session
    CV1 max (256, 32, 32, 32)
    CV2 max (256, 16, 16, 32)
    CV3 relu (256, 8, 8, 128)
    CV4 relu (256, 4, 4, 128)
    CV5 relu (256, 2, 2, 64)
    CV2FC relu (256, 1024)
    FC1 relu (256, 1024)
    FC2 relu (256, 1024)
    FC3 dropout (256, 1024)
    First batch test  Loss: 1.18043 Accuracy: 0.414062
    iter: 0 batch: 0 Loss: 0.941256 Accuracy: 0.636719
    iter: 0 batch: 1 Loss: 1.38965 Accuracy: 0.546875
    iter: 0 batch: 2 Loss: 1.28251 Accuracy: 0.542969
    iter: 0 batch: 3 Loss: 1.33823 Accuracy: 0.617188
    Train batch mean 0.585938 min: 0.542969 max 0.636719
    iter: 1 batch: 0 Loss: 0.550499 Accuracy: 0.769531
    iter: 1 batch: 1 Loss: 1.01165 Accuracy: 0.683594
    iter: 1 batch: 2 Loss: 0.963572 Accuracy: 0.640625
    iter: 1 batch: 3 Loss: 1.02371 Accuracy: 0.664062
    Train batch mean 0.689453 min: 0.640625 max 0.769531
  
  
    ... [some iterations later] ...
    
    
    Train batch mean 0.982422 min: 0.976562 max 0.992188
    iter: 19 batch: 0 Loss: 0.0344171 Accuracy: 1.0
    iter: 19 batch: 1 Loss: 0.0420092 Accuracy: 0.988281
    iter: 19 batch: 2 Loss: 0.042245 Accuracy: 0.980469
    iter: 19 batch: 3 Loss: 0.0493034 Accuracy: 0.992188
    Train batch mean 0.990234 min: 0.980469 max 1.0
    Train last iter mean 0.990234 min: 0.980469 max 1.0
    Saving model in: test_model
    Done

### Testing:

```python
CV1 = { 'type':'CV', 'depth':32, 'filter_w':11, 'filter_stride':[1,4,4,1], 'norm_bool':True,
    'max_pooling':True, 'max_pool_mask':[1,3,3,1], 'max_pool_stride':[1,2,2,1], 'padding':'SAME',
       'name':'CV1'} 

CV2 ={'type':'CV',  'depth':32, 'filter_w':5, 'filter_stride':[1,1,1,1], 'norm_bool':True,
      'max_pooling':True, 'max_pool_mask':[1,3,3,1], 'max_pool_stride':[1,2,2,1], 'padding':'SAME',
      'name':'CV2' } 
CV3 ={'type':'CV',  'depth':128, 'filter_w':3, 'filter_stride':[1,2,2,1], 'norm_bool':True,
      'name':'CV3' } 
CV4 ={'type':'CV',  'depth':128, 'filter_w':3, 'filter_stride':[1,2,2,1], 'norm_bool':True,
      'name':'CV4' }
CV5 ={'type':'CV',  'depth':64, 'filter_w':3, 'filter_stride':[1,2,2,1], 'norm_bool':True,
      'name':'CV5' }

CV2FC={'type':'CV2FC', 'neurons':1024, 'norm_bool':True, 'name':'CV2FC'} 
FC1 = { 'type':'FC', 'neurons':1024, 'norm_bool':True, 'name':'FC1'} 
FC2 = { 'type':'FC', 'neurons':1024, 'norm_bool':True, 'name':'FC2'} 
FC3 = { 'type':'FC', 'neurons':1024, 'norm_bool':True, 'name':'FC3',
       'drop_out_bool':True, 'keep_prob_train':0.5} 

layers = [CV1,CV2,CV3,CV4,CV5,CV2FC,FC1,FC2,FC3]


stats_dic_test = cnn_modeling.create_graph(mini_train_batch,layers=layers,test_batch=mini_test_batch,
            width=256,height=256, 
             batch_proc=True, 
            test_batch_bool=True, 
            only_feed_forward=True,
             restore_session = True, save_model = False, 
             stddev_n = 0.1, learning_rate = 1e-4,iters=20,model_file='test_model')
```

    Creating layer: CV1
    Creating layer: CV2
    Creating layer: CV3
    Creating layer: CV4
    Creating layer: CV5
    Creating layer: CV2FC
    Creating layer: FC1
    Creating layer: FC2
    Creating layer: FC3
    Starting session
    INFO:tensorflow:Restoring parameters from ./test_model
    CV1 max (256, 32, 32, 32)
    CV2 max (256, 16, 16, 32)
    CV3 relu (256, 8, 8, 128)
    CV4 relu (256, 4, 4, 128)
    CV5 relu (256, 2, 2, 64)
    CV2FC relu (256, 1024)
    FC1 relu (256, 1024)
    FC2 relu (256, 1024)
    FC3 dropout (256, 1024)
    First batch test  Loss: 0.0101463 Accuracy: 1.0
    Evaluating using test batch
    batch: 0 Loss: 0.905623 Accuracy: 0.664062
    batch: 1 Loss: 1.15409 Accuracy: 0.644531
    batch: 2 Loss: 0.834898 Accuracy: 0.699219
    batch: 3 Loss: 0.885531 Accuracy: 0.65625
    Accuracy mean: 0.666016 max: 0.699219 min: 0.644531
    Done


We saw a 6% improvement from the previous test. From 2 to 5 convolutional layers and an increase in the filters/depth on them.

#### Visualization of the learning process


```python
cnn_modeling.plot_list(stats_dic_improve['train_acc'], figsize=(8, 6), title='Train accuracy', xlabel='Iters', ylabel='Acc')
cnn_modeling.plot_list(stats_dic_improve['train_cross'], figsize=(8, 6), title='Train Cross entropy', xlabel='Iters', ylabel='Loss')
```


![png](output_34_0.png)



![png](output_34_1.png)


Through the creation of the model the dictionary with the hyperparameters gets modified. The weights and biases are stored in the keys **'W'** and **'b'** of every layer.


```python
print(CV1.keys())
with tf.Session() as s:
    s.run(tf.global_variables_initializer())
    r = s.run(CV1['W'])
    print(r.shape)
    print(r)
    
```

    dict_keys(['max_pool_stride', 'b', 'variance', 'max_pooling', 'depth', 'filter_w', 'output_label', 'type', 'padding', 'norm_bool', 'W', 'norm', 'max_pool_mask', 'filter_stride', 'mean', 'input_depth', 'max', 'conv', 'name', 'relu'])
    (11, 11, 1, 32)
    [[[[ 0.01258293  0.04532569  0.00288749 ..., -0.03397996 -0.05494995
         0.03610544]]
    
      [[ 0.12260102 -0.02733466  0.00377944 ...,  0.06139905 -0.10509678
        -0.06860053]]
 
        ... Some weights later ...
 
    
      [[-0.07153799  0.01371634  0.05449439 ..., -0.00618845 -0.11393668
        -0.118617  ]]
    
      [[-0.08060057  0.12104277  0.14032906 ..., -0.10620737 -0.13509519
        -0.12730974]]]]
  
  

## Simpler training-testing 

The option has two possibilities:  
* **train**: Uses the **batch_train** for training and will compute the learning_step process (E.g. Gradient descent). When the option **first_run** is set to **True**, no model is loaded. At the end of the iterations the model will be saved in the **model_file** variable. If **first_run** is set to **False**, the **model_file** model will be loaded for the next specified iterations.  
* **test**: Uses the **batch_test** for an only forward process. It will load the **model_file** specified.  



```python
d1,d2 = 32,32

CV1 ={'type':'CV',  'depth':d1, 'filter_w':3, 'filter_stride':[1,1,1,1], 'norm_bool':True,
      'name':'CV1' } 
CV2 ={'type':'CV',  'depth':d2, 'filter_w':3, 'filter_stride':[1,1,1,1], 'norm_bool':True,
      'max_pooling':True, 'max_pool_mask':[1,2,2,1], 'max_pool_stride':[1,2,2,1], 'padding':'SAME',
      'name':'CV2' }

CV2FC={'type':'CV2FC', 'neurons':128, 'norm_bool':True, 'name':'CV2FC',
      'drop_out_bool':True, 'keep_prob_train':0.5} 

layers = [CV1,CV2,CV2FC]

mode = 'train'
first_run = True
iters = 1
model_file = 'cnn_model_01'
batch_train = group1_batch
batch_test = group2_batch
iMode='CNN'

st = time()
stats_dic = cnn_modeling.create_graph_bools(batch_train,layers,batch_test,mode=mode,
            first_run=first_run, model_file=model_file,
            stddev_n = 0.1, learning_rate = 0.001,iters=iters,
            batch_proc=True, width=256,height=256,input_mode=iMode)
print("Minutes taken", np.ceil((time() - st)/60))

```

## Create a Deep Neural Network without Convolutional layers

Change the **input_mode** from **CNN** to **DNN** 


```python
FC1 = { 'type':'FC', 'neurons':1024, 'norm_bool':True, 'name':'FC1',
      'drop_out_bool':True, 'keep_prob_train':1,
      'x2_bool':False,'x2_features':256} 
FC2 = { 'type':'FC', 'neurons':1024, 'norm_bool':True, 'name':'FC2',
       'drop_out_bool':True, 'keep_prob_train':0.5} 
layers = [FC1,FC2]

mode = 'train'
first_run = True
iters = 1
model_file = 'dnn_model_01'
batch_train = group1_batch
batch_test = group2_batch
iMode='DNN'

st = time()
stats_dic = cnn_modeling.create_graph_bools(batch_train,layers,batch_test,mode=mode,
            first_run=first_run, model_file=model_file,
            stddev_n = 0.1, learning_rate = 0.001,iters=iters,
            batch_proc=True, width=256,height=256,input_mode=iMode)
print("Minutes taken", np.ceil((time() - st)/60))
```

## Automate traing-testing with decreasing learning-rate

Update the learning rate based on the performance of the trained model in the dev-set: There is an option to automate the training-testing processes. After a desired amount of iterations, period **T**, the model is tested in the testing-batch. The amount of False Positives (**fp**) and False Negatives (**fn**) are acquired at this **T0**. At **T1** the **fp1** and **tn1** are compared with **fn0** and **fp0**. If **fp1**-**fp0** > an **allowed fp increase (AFpI)**  or **fn1**-**fn0**>**fn (AFnI)** the learning rate is decreased by a factor of 10. If a minimum learning rate is reached the learning process stops.  

Create a dictionary for the **auto_test** variable. Enable this mode by setting **bool** to true.  
* **test_interval** defines the period **T** to evaluate how well the model is performing on the dev/test set.  
*  **stop_max** if set to **True**, the learning process will stop when the number of iterations reach the **max_iters** value or if the **train_rate_stop** is reached. If set to **False** it only will stop when **train_rate_stop** is reached.  
* **fp_inc** and **fn_inc** is the **Allowed Increase** for **fp** and **fn** respectively. If this values are overpassed, the learning rate will decrease by a factor of 10 and the last **fp** and **fn** taken in this update will be consired as the base for further evaluations.  
* **train_rate_stop** is the minimun value that the learning-rate could decrease.  
* **send_email_bool** if set to **True** the system will send an e-mail with the account **emain_origin** and passoword **email_pass** to **email_destination**.  


```python
d1,d2 = 32,32

CV1 ={'type':'CV',  'depth':d1, 'filter_w':3, 'filter_stride':[1,1,1,1], 'norm_bool':True,
      'name':'CV1' } 
CV2 ={'type':'CV',  'depth':d2, 'filter_w':3, 'filter_stride':[1,1,1,1], 'norm_bool':True,
      'max_pooling':True, 'max_pool_mask':[1,2,2,1], 'max_pool_stride':[1,2,2,1], 'padding':'SAME',
      'name':'CV2' }

CV2FC={'type':'CV2FC', 'neurons':128, 'norm_bool':True, 'name':'CV2FC',
      'drop_out_bool':True, 'keep_prob_train':0.5} 

layers = [CV1,CV2,CV2FC]

mode = 'train'
first_run = True
iters = 1
model_file = 'cnn_model_01'
batch_train = group1_batch
batch_test = group2_batch
iMode='CNN'

auto_test={'bool':True,'test_interval':6,'stop_max':False,
           'max_iters':8,'fp_inc':2,'fn_inc':2,'train_rate_stop':0.00000001,
           'restore_session':False,
          'send_email_bool':True,'email_origin':'origin.email@gmail.com',
           'email_pass':'your.email.password',
           'email_destination':'destination.email@gmail.com'}

st = time()
stats_dic = cnn_modeling.create_graph_bools(batch_train,layers,batch_test,mode=mode,
            first_run=first_run, model_file=model_file,
            stddev_n = 0.1, learning_rate = 0.001,iters=iters,
            batch_proc=True, width=256,height=256,input_mode=iMode,
            auto_test=auto_test)
print("Minutes taken", np.ceil((time() - st)/60))

```

The email will have the structure:  
**Subject**: Model evaluation completed: **hostname of the server/computer** Max-iter **max_iter value**   
**Content**:  
stats:  
 {'tn': 2147, 'specificity': 0.8439465375631031, 'fp': 397, 'fn': 545, 'sensitivity': 0.785770437162852, 'tp': 1999} | learning rate: 0.0001  
{'tn': 2150, 'specificity': 0.8451257828414867, 'fp': 394, 'fn': 532, 'sensitivity': 0.790880500035847, 'tp': 2012} | learning rate: 0.0001  
...   
{'tn': 2166, 'specificity': 0.8514150909928652, 'fp': 378, 'fn': 492, 'sensitivity': 0.8066037704142933, 'tp': 2052} | learning rate: 1.0000000000000002e-08  
...  
{'tn': 2167, 'specificity': 0.8518081727523263, 'fp': 377, 'fn': 493, 'sensitivity': 0.8062106886548321, 'tp': 2051} | learning rate: 1.0000000000000002e-08  
{'tn': 2167, 'specificity': 0.8518081727523263, 'fp': 377, 'fn': 494, 'sensitivity': 0.805817606895371, 'tp': 2050} | learning rate: 1.0000000000000003e-09  

#### Enable the gmail security for low security applications before having the email send working

Before using the email send feature, test the following lines:  
  
import smtplib  
server = smtplib.SMTP('smtp.gmail.com:587')  
server.starttls()  
**Next, log in to the server** 
server.login("your.email@gmail.com", "your.password.")  
  
**Send a test email**  
msg = "Subject:test \n\n Hello! \n test context\n" # The \n\n separates the message from the headers  
server.sendmail("your.email@gmail.com", "destination.email@gmail.com", msg)  


You are expected to see an error message after its excecution. Check your gmail inbox and you will get a **security warning**. Read the description, click on **enable access for less secure applications** and follow the instructions. Once done, try the previous lines again and if you don't get any error, your account is ready to be used. (I don't use my main email account for this).
  
## Miscellaneuos features:  
### Feed a histogram of the input layer (image) in parallel with the convolutional layers  

![png](20171012-115122.png)

You can feed a histogram of the input layer to a Fully Connected Layer by setting the option **x2_bool** to **True** and defining the number of bins for the histogram. **x2_features=256** will feed a histogram of 256 bins  

```python  

CV2FC={'type':'CV2FC', 'neurons':128, 'norm_bool':True, 'name':'CV2FC',
      'drop_out_bool':True, 'keep_prob_train':0.5} 
FC1 = { 'type':'FC', 'neurons':1024, 'norm_bool':True, 'name':'FC1',
      'drop_out_bool':True, 'keep_prob_train':1,
      'x2_bool':False,'x2_features':256} 
FC2 = { 'type':'FC', 'neurons':1024, 'norm_bool':True, 'name':'FC2',
       'drop_out_bool':True, 'keep_prob_train':0.5} 

layers = [CV1,CV2,CV2FC,FC1,FC2]

```
### Pre-processing: Explicity convolution for border detection  
I have Included a previous step prior to the convolutional layers. A manual convolution of the images against 4 filters. After the convolutions with these filters the result is added and normalized to remark every possible border in these directions  
```python  
F1 = tf.constant([[1.0,0,-1.0],[1.0,0,-1.0],[1.0,0,-1.0]])
F2 = tf.transpose(F1)
F3 = tf.constant([[1.0,0,1.0],[0,0,0],[-1.0,0,-1.0]])
F4 = tf.transpose(F3)

CVM = CV1+CV2+CV3+CV4
CVMn = 255.0*CVM/(tf.reduce_max(CVM)-tf.reduce_min(CVM))

```  

Then, this result goes against an upper-pass filter to transform upper values of a threshold to 255 and those below to zero. The functions is described in: https://github.com/bsaldivaremc2/image_classification_synthetic_rotated_sample_creator  

To activate this feature add to the **cnn_modeling.create_graph_bools** function the variables:  
* **reduce_noise**: To **True**  
* **rn_shift**: is the threshold value between 0 and 1. a value of 0.5 means that values above 127 will be set to 255 and below to 0.   
* **rn_magnitude**: from 0 to 1, is the factor that will transform the value of each pixel, if set closer to 1 a transformation towards 255 for values over the threshold will be set. Lower values will provide lighter transformations.

```python  
cnn_modeling.create_graph_bools(...,reduce_noise=False,rn_shift=0.03,rn_magnitude=0.8,...)  

```
  
### Visualize the result of a given Convolution Layer against an input mini-batch set of images  
If you want to see how it looks like the image after the convolution of a given layer, including the max-pooling layer set the option **get_deconv**=**True** . Also specify which layer you want to see in the variable **deconv_layer='CV4'**. Since every layer has a **norm**, **relu**, **conv**, **max** operations, you need to specify which of these you want to be returned. To see the convolution layers result set **deconv_val = 'conv'** or to see the max pooling if enabled **deconv_val = 'max'**.  
Use this in a **test** mode.  
The function will return a key **stats_dic['deconv']** which holds the result.  
```python  

stats_dic = cnn_modeling.create_graph_bools(batch_train,layers,batch_test,mode=test,...
            get_deconv=False,deconv_layer='CV4',deconv_val = 'max')  

r = stats_dic['deconv']
image_to_see = 20 #Number of image in the mini-batch

print(r.shape)
for filterx in range(0,2): #See the first two filters
    deconv = r[image_to_see,:,:,filterx]
    deconv_t = np.reshape(deconv,(r.shape[1],r.shape[2]))
    plt.imshow(deconv_t,cmap="gray")
    plt.show()
            
```  

### Important indications:
* As may have seen there is a structure Convolutional Layer **(CV)**, CV2FC and Fully Connected **(FC)**. For the moment it is required to have this structure and have a translation layer from the CV layers to the FC. This CV2FC is a FC layer with a reshape step in the first part.
* You can enable max pooling by setting **max_pooling=True**, but you need to specify the hyperparameters shown. For the moment there is no default values 
* Every layer requires a unique name. During the saving and restoring of the model I saw some issues when variables lack a name. 
* The **create_graph** function requires that the train and test batches be a list of **"bn"** mini-batches where the mini-batch[0] is for x and [1] for y. x should be of shape **mini-batch size** by **n features** and y **mini-batch size** by **nc classes**
* The layers were tested with grayscale images. if the input batches have 3 channels (r,g,b) or others add to the first **CV** dictionary the key **'prev_channels':3**  and see what happens. By default it is set to **1** . I haven't tested for more channels so if you do it and works let me know.