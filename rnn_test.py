import tensorflow as tf
import numpy as np
import os
import cv2
"""
测试RNN对字符分割的效果
"""
n_steps=20   #步长，可以理解为一张图片或一句话训练时需要的状态数
n_inputs=20  #输入参数，例如一行像素值
image_size=20
n_neurons=200  #隐含层中包括的神经元数量
n_outputs=15   #输出神经元个数
learning_rate=0.001

def RNN():
    X=tf.placeholder(tf.float32,[None,n_steps,n_inputs])
    Y=tf.placeholder(tf.int64,[None])

    #定义隐含层神经元
    basic_cell=tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
    #outputs=[batchsize,step,num_neurons],outputs包括所有步骤的输出状态,  states=[batchsize,num_neurons]表示最后一步的输出状态
    outputs,states=tf.nn.dynamic_rnn(basic_cell,X,dtype=tf.float32)

    logits=tf.contrib.layers.fully_connected(states,n_outputs,activation_fn=None)

    loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,logits=logits))
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op=optimizer.minimize(loss)

    predict_label=tf.argmax(logits,1)
    accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1),Y),tf.float32))
    
    return {"X":X,
           "Y":Y,
           "loss":loss,
           "train":training_op,
           "accuracy":accuracy,
           "outputs":outputs,
           "states":states}

def LSTM(type='LSTM'):
    X=tf.placeholder(tf.float32,[None,n_steps,n_inputs])
    Y=tf.placeholder(tf.int64,[None])

    #lstm_cell=tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons,use_peepholes=True)
    if type=='LSTM':
        lstm_cell = tf.contrib.rnn.LSTMCell(num_units=n_neurons, use_peepholes=True) 
        #state形状为[2，batch_size, cell.output_size],state是由Ct 和 ht组成的
        outputs,states=tf.nn.dynamic_rnn(lstm_cell,X,dtype=tf.float32)
        top_layer_h_state=states[-1]
    else:
        gru_cell = tf.contrib.rnn.GRUCell(num_units=n_neurons)
        outputs,states=tf.nn.dynamic_rnn(gru_cell,X,dtype=tf.float32)
        top_layer_h_state=states

    #layers.dense表示全链接
    logits=tf.layers.dense(top_layer_h_state,n_outputs,name='softmax')
    loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,logits=logits))
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op=optimizer.minimize(loss)

    accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1),Y),tf.float32))
    
    return {"X":X,
           "Y":Y,
           "loss":loss,
           "train":training_op,
           "accuracy":accuracy,
           "outputs":outputs,
           "states":states}

class DataProcess:
    def __init__(self,train_data_path):
        self.train_images_labels=[]
        self.count=0
        self.epoch=0
        self.train_data_path=train_data_path
        images_name=os.listdir(self.train_data_path)
        #print(len(images_name))
        for image_name in images_name:
            image_path=os.path.join(self.train_data_path,image_name)
            #print(image_path)
            label=int(image_name.split('_')[0])
            #print(label)
            self.train_images_labels.append({"image":image_path,"label":label})
        np.random.shuffle(self.train_images_labels)
        print(len(self.train_images_labels))

    def read_batch(self,batch_size,image_channel=1):
        images=np.zeros([batch_size,image_size,image_size])
        labels=[]
        num=0
        while num<batch_size:
            image_path=self.train_images_labels[self.count]["image"]
            labels.append(self.train_images_labels[self.count]["label"])
            self.count+=1
            image=cv2.imread(image_path,1)
            image=cv2.resize(image,(image_size,image_size))
            if image_channel==1:
                image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) 
            #print(np.shape(image))
            image=np.reshape(image,[-1,image_size,image_size])
            images[num,:]=image
            num+=1
            if self.count>=len(self.train_images_labels):
                np.random.shuffle(self.train_images_labels)
                self.count=0
                self.epoch+=1
        
        return images,labels

def train(data_path,restore=False,max_iter=30000):
    dataReader=DataProcess(data_path)
    rnn=LSTM(type='GRU')
    model_name='rnn'
    batch_size=128
    checkpoint_dir='./rnn_checkpoint/'
    #tf.reset_default_graph()
    print("begin train...")
    with tf.Session() as sess:
        saver=tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        
        start_step=0
        if restore:
            ckpt=tf.train.latest_checkpoint(checkpoint_dir)
            if ckpt:
                saver.restore(sess,ckpt)
                print("restore from the checkpoint {0}".format(ckpt))
                start_step += int(ckpt.split('-')[-1])
        iter_num=0
        while iter_num<=max_iter:
            train_images_batch,train_labels_batch=dataReader.read_batch(batch_size=batch_size)
            
            
            _,_loss,_accuracy,_outputs,_states=sess.run([rnn["train"],rnn["loss"],
                                            rnn["accuracy"],rnn["outputs"],rnn["states"]],
                                            feed_dict={rnn["X"]:train_images_batch,rnn["Y"]:train_labels_batch})
            #print(np.shape(_outputs))
            #print(np.shape(_states))
            print("step:" ,start_step, "   loss:",_loss,"   accuracy:",_accuracy)
            
            iter_num+=1
            start_step+=1
            if start_step%500==1:
                print("save the ckpt of {0}".format(start_step))
                saver.save(sess,os.path.join(checkpoint_dir,model_name),global_step=start_step)

if __name__=="__main__":
    data_path=r'D:\micr\Temp_samples\b_train'
    train(data_path=data_path,restore=False)

                                    

        







