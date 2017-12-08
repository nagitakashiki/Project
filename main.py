import tensorflow as tf
import os
import shutil
import numpy as np

from PIL import Image

"""
実行例

ディレクトリ構造が以下の場合
~/Station_samples/0梅郷駅西口/(省略)
                  1梅郷駅東口/(省略)
                  2東京駅/(省略)
                  3柏駅/(省略)
                  4池袋駅/(省略)
                  5運河駅西口/(省略)
                  6運河駅東口/(省略)
  stations_train.tfrecords
  stations_test.tfrecords
  main.py

$python3
>>>import main
>>>m = main.main()
>>>m.train(n_class=7,size_image=56,model='cnn1')

実行後
~/logs/Project/train/(tfeventsファイル)
  save_files/(ckptファイル群)
  Station_samples/(省略)
  stations_train.tfrecords
  stations_test.tfrecords
  main.py

>>>m.test(n_class=7,size_image=56,model='cnn1')

実行後
~/logs/Project/train/(tfeventsファイル)
               test/(tfeventsファイル)
  save_files/(ckptファイル群)
  Station_samples/(省略)
  stations_train.tfrecords
  stations_test.tfrecords
  main.py
"""

"""
実行例2(実際に画像を与えて結果を返す関数の使い方)

前提
トレーニングを20000回行い、ディレクトリ構造が
~/save_files
　main.py
 0梅郷駅西口/(省略)
 1梅郷駅東口/(省略)
となっていることを前提とします。

(1)identification
画像を一枚投げて結果を返す関数
$python3
>>>import main
>>>m = main.main()
>>>m.identification(n_class=クラス数,size_image=画像サイズ,model='cnn1',img=与える画像)
(2)stepidentification
100ステップ毎の学習データを用いて画像を一枚投げて結果を返す関数
$python3
>>>import main
>>>m = main.main()
>>>m.stepidentification(n_class=クラス数,size_image=画像サイズ,model='cnn1',img=与える画像)
(3)listidentification
あるディレクトリ内に保存されている画像すべてを投げて結果を返す関数
これに限りディレクトリ名に使われている数字をラベルとして用いた、正誤判定を行っているため、与えるディレクトリは
0梅郷駅西口/(省略)
1梅郷駅東口/(省略)
みたいな名前になっていることが前提
$python3
>>>import main
>>>m = main.main()
>>>m.listidentification(n_class=クラス数,size_image=画像サイズ,model='cnn1',dir=与えるディレクトリ)
"""
class main(object):

    #tfrecordsファイルから画像データと対応するラベルを取得する
    def input(self,rec_name,IMAGE_SIZE,BATCH_SIZE):
        file_name_queue = tf.train.string_input_producer(rec_name)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(file_name_queue)
        # デシリアライズ
        features = tf.parse_single_example(
            serialized_example,
            features={
                "label": tf.FixedLenFeature([], tf.int64),
                "image": tf.FixedLenFeature([], tf.string),
                "height": tf.FixedLenFeature([], tf.int64),
                "width": tf.FixedLenFeature([], tf.int64),
                "depth": tf.FixedLenFeature([], tf.int64),
            })
        #画像の読み込み
        img = tf.reshape(tf.decode_raw(features["image"], tf.uint8),
                                   tf.stack([IMAGE_SIZE, IMAGE_SIZE, 3]))
        img = tf.cast(img, tf.float32)
        images, labels = tf.train.shuffle_batch(
            [img, tf.cast(features['label'], tf.int32)],
            batch_size=BATCH_SIZE,capacity=500,min_after_dequeue=100
        )
        images = tf.image.resize_images(images, [IMAGE_SIZE, IMAGE_SIZE])

        return images,labels

    #dirで指定されたパスが存在しない場合ディレクトリ作成
    def make_dir(self,dir,format=False):
        if not os.path.exists(dir):
            os.makedirs(dir)
        if format and os.path.exists(dir):
            shutil.rmtree(dir)

    #tensorboardのサマリに追加する
    def variable_summaries(self,var):

        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    #重みベクトルを初期化して返す
    def _variable_with_weight_decay(self,name,shape,stddev,wd):

        var = tf.get_variable(name, shape=shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    #畳込み層
    def l_conv(self,element,size,chanel,kinds,l_name):

        with tf.variable_scope(l_name) as scope:
            kernel = self._variable_with_weight_decay('weights', shape=[size, size, chanel, kinds], stddev=0.01, wd=0.0)
            conv = tf.nn.conv2d(element, kernel, strides=[1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[kinds], initializer=tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            conv = tf.nn.relu(bias, name=scope.name)
            self.variable_summaries(conv)
            return conv

    #全結合層
    def l_full(self,element,con_from,con_to,l_name):

        with tf.variable_scope(l_name) as scope:
            weights = self._variable_with_weight_decay('weights', shape=[con_from, con_to], stddev=0.01, wd=0.005)
            biases = tf.get_variable('biases', shape=[con_to], initializer=tf.constant_initializer(0.0))
            fc = tf.nn.relu(tf.nn.bias_add(tf.matmul(element, weights), biases), name=scope.name)
            self.variable_summaries(fc)
            return fc

    #CNNモデルの定義
    def cnn1(self,images,n_class,keep_prob):
        #畳込み、プーリング1
        conv1 = self.l_conv(images,5,3,32,"conv1")
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        #畳込み、プーリング2
        conv2 = self.l_conv(pool1,5,32,64,"conv2")
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        shape = pool2.get_shape().as_list()
        dim = (shape[1]**2)*shape[3]
        reshape = tf.reshape(pool2,[-1,dim])
        #全結合層1
        fc1 = self.l_full(reshape,dim,1024,"fc1")
        #ドロップアウト層
        fc1_drop = tf.nn.dropout(fc1, keep_prob)
        #全結合層2
        
        with tf.variable_scope("fc2") as scope:
            weights = tf.get_variable("weight", shape=[1024, n_class], initializer=tf.truncated_normal_initializer(stddev=0.01))
            biases = tf.get_variable("biases", shape=[n_class], initializer=tf.constant_initializer(0.0))
            fc2 = tf.nn.bias_add(tf.matmul(fc1_drop, weights), biases, name=scope.name)
            self.variable_summaries(fc2)
        return fc2

    #VGGモデルの定義
    def vgg1(self,images,n_class,keep_prob):
        #畳込み1_1、畳込み1_2、プーリング1
        conv1_1 = self.l_conv(images,3,3,64,"conv1_1")
        conv1_2 = self.l_conv(conv1_1,3,64,64,"conv1_2")
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        #畳込み2_1、畳込み2_2、プーリング2
        conv2_1 = self.l_conv(pool1,3,64,128,"conv2_1")
        conv2_2 = self.l_conv(conv2_1,3,128,128,"conv2_2")
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
        #畳込み3_1、畳込み3_2、畳込み3_3、プーリング3
        conv3_1 = self.l_conv(pool2,3,128,256,"conv3_1")
        conv3_2 = self.l_conv(conv3_1,3,256,256,"conv3_2")
        conv3_3 = self.l_conv(conv3_2,3,256,256,"conv3_3")
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
        #畳込み4_1、畳込み4_2、畳込み4_3、プーリング4
        conv4_1 = self.l_conv(pool3,3,256,512,"conv4_1")
        conv4_2 = self.l_conv(conv4_1,3,512,512,"conv4_2")
        conv4_3 = self.l_conv(conv4_2,3,512,512,"conv4_3")
        pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

        shape = pool4.get_shape().as_list()
        dim = (shape[1]**2)*shape[3]
        reshape = tf.reshape(pool4,[-1,dim])
        #全結合層1
        fc1 = self.l_full(reshape,dim,2048,"fc1")
        #全結合層2
        fc2 = self.l_full(fc1,2048,1024,"fc2")
        #全結合層3
        fc3 = self.l_full(fc2,1024,256,"fc3")
        #ドロップアウト層
        fc3_drop = tf.nn.dropout(fc3, keep_prob)
        #全結合層4
        with tf.variable_scope("fc4") as scope:
            weights = tf.get_variable("weight", shape=[256, n_class], initializer=tf.truncated_normal_initializer(stddev=0.01))
            biases = tf.get_variable("biases", shape=[n_class], initializer=tf.constant_initializer(0.0))
            fc4 = tf.nn.bias_add(tf.matmul(fc3_drop, weights), biases, name=scope.name)
            self.variable_summaries(fc4)
        return fc4

    #モデルの呼び出し
    def model(self,images,n_class,keep_prob,model):
        if model == 'cnn1':
            return self.cnn1(images,n_class,keep_prob)
        elif model == 'vgg1':
            return self.vgg1(images,n_class,keep_prob)

    #トレーニングを行う
    def train(self,n_class,size_image,model,log_dir='logs/Project/train'):

        sess = tf.InteractiveSession()

        self.make_dir(log_dir,False)
        #画像とラベルとドロップアウト層のパラメータのプレイスホルダを生成
        x = tf.placeholder(tf.float32, shape=[None,size_image,size_image,3])
        y_ = tf.placeholder(tf.float32, shape=[None,n_class])
        keep_prob = tf.placeholder(tf.float32)
        #画像をモデルにかける
        y_conv = self.model(x,n_class,keep_prob,model)
        #損失関数よりlossを取得
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
            tf.add_to_collection('losses', cross_entropy)
            error=tf.add_n(tf.get_collection('losses'), name='total_loss')
            self.variable_summaries(error)
        #確率的勾配降下法により重みを最適化
        with tf.name_scope('accuracy'):
            train_step = tf.train.AdamOptimizer(1e-4).minimize(error)
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.variable_summaries(accuracy)
        #サマリをマージする
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log_dir, sess.graph)

        saver = tf.train.Saver(max_to_keep=500)
        #画像データと対応するラベルを取得
        record=[["stations_train"]]
        img,lab=self.input(list(map(lambda x:x[0]+".tfrecords", record)),size_image,50)
        lab = tf.one_hot(lab,n_class)
        lab = tf.cast(lab, tf.float32)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #スレッドを利用して並列処理
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for i in range(20000):
                #バッチサイズ分のデータを格納
                batch = sess.run([img,lab])
                if (i+1) % 100 == 0:
                    #100ステップ毎のパラメータをtensorboardに書き出す
                    run_options  = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary = sess.run(merged,
                        feed_dict={x: batch[0], y_:batch[1], keep_prob: 1.0},
                        options=run_options,
                        run_metadata=run_metadata)
                    writer.add_summary(summary, i)
                    #100ステップ毎の正答率をprintし、パラメータ情報を保存する
                    train_accuracy = accuracy.eval(feed_dict={
                        x: batch[0], y_: batch[1], keep_prob: 1.0})
                    print('step %d, training accuracy %g' % ((i+1), train_accuracy))
                    saver.save(sess, "save_files/model.ckpt", global_step=(i+1))
                #トレーニング
                train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            coord.request_stop()
            coord.join(threads)
            writer.close()

    #テストを行う
    def test(self,n_class,size_image,model,log_dir='logs/Project/test'):

        sess = tf.InteractiveSession()
        #画像とラベルとドロップアウト層のパラメータのプレイスホルダを生成
        x = tf.placeholder(tf.float32, shape=[None,size_image,size_image,3])
        y_ = tf.placeholder(tf.float32, shape=[None,n_class])
        keep_prob = tf.placeholder(tf.float32)
        #画像をモデルにかける
        y_conv = self.model(x,n_class,keep_prob,model)
        #正答率の計算
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.variable_summaries(accuracy)
        #サマリをマージする
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log_dir, sess.graph)

        saver = tf.train.Saver()
        #画像データと対応するラベルを取得
        record=[["stations_test"]]
        img,lab=self.input(list(map(lambda x:x[0]+".tfrecords", record)),size_image,50)
        lab = tf.one_hot(lab,n_class)
        lab = tf.cast(lab, tf.float32)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #スレッドを利用して並列処理
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for i in range(200):
                #ckptファイルからパラメータ情報を復元
                saver.restore(sess, "save_files/model.ckpt-"+str((i+1)*100))
                #バッチサイズ分のデータを格納
                batch = sess.run([img,lab])
                #100ステップ毎のパラメータをtensorboardに書き出す
                run_options  = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary = sess.run(merged,
                    feed_dict={x: batch[0], y_:batch[1], keep_prob: 1.0},
                    options=run_options,
                    run_metadata=run_metadata)
                writer.add_summary(summary, i)
                #100ステップ毎の正答率をprintする
                print('step %d, test accuracy %g' % ((i+1)*100, accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})))
            coord.request_stop()
            coord.join(threads)

    #ディレクトリのラベルの判別用
    def pull_num(self,str):
        i=1
        for i in range(len(str)):
            try:
                int(str[i])
            except ValueError:
                return(int(str[0:i]))
            i+=1
        return int(str)


    #画像を一枚与えたときの結果を返す
    def identification(self,n_class,size_image,model,img):
        
        if not os.path.exists("save_files"):
            print('Please train')
        else:
            sess = tf.InteractiveSession()
            image = [np.array(Image.open(img).convert("RGB").resize((size_image, size_image)))]
            
            x=tf.placeholder(tf.float32,shape=[None,size_image,size_image,3])
            keep_prob=tf.placeholder(tf.float32)
            y_conv=self.model(x,n_class,keep_prob,model)
            
            y_=tf.nn.softmax(y_conv)

            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                
                saver.restore(sess,"save_files/model.ckpt-20000")
                result = np.round(sess.run(y_,feed_dict={x: image,keep_prob: 1.0}),3)
                stationnum = sess.run(tf.argmax(result,1))
                print('station {0} ,\n station number is {1}'.format(result,stationnum))
            

    #一枚与えられた画像に対し100ステップごとの学習結果を適用した結果を返す
    def stepidentification(self,n_class,size_image,model,img):
        
        if not os.path.exists("save_files"):
            print('Please train')
        else:
            sess = tf.InteractiveSession()
            image = [np.array(Image.open(img).convert("RGB").resize((size_image, size_image)))]

            x=tf.placeholder(tf.float32,shape=[None,size_image,size_image,3])
            keep_prob=tf.placeholder(tf.float32)
            y_conv=self.model(x,n_class,keep_prob,model)
            
            y_=tf.nn.softmax(y_conv)

            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                for i in range(200):
                    saver.restore(sess,"save_files/model.ckpt-"+str((i+1)*100))
                    result = np.round(sess.run(y_,feed_dict={x: image,keep_prob: 1.0}),3)
                    stationnum = sess.run(tf.argmax(result,1))
                    print('step {0} {1} ,\n station number is {2}'.format(((i+1)*100),result,stationnum))
                
    #ディレクトリ内の写真すべてをなげて結果を返す
    def listidentification(self,n_class,size_image,model,dir):
        if not os.path.exists("save_files"):
            print('Please train')
        else:
            if not os.path.exists(dir):
                print('No directry')
            else:

                imglist=os.listdir(dir)
                label=np.array(self.pull_num(dir))
                labels=[label]*(len(imglist))
                
                x=tf.placeholder(tf.float32,shape=[None,size_image,size_image,3])
                keep_prob=tf.placeholder(tf.float32)
                y_conv=self.model(x,n_class,keep_prob,model)
            
                y_=tf.nn.softmax(y_conv)

                saver = tf.train.Saver()
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    saver.restore(sess,"save_files/model.ckpt-20000")

                    colorlist=[np.array(Image.open(dir+"/"+img).convert("RGB").resize((size_image, size_image))) for img in imglist]

                    #for img in imglist:
                     #   image = np.array(Image.open(dir+"/"+img).convert("RGB").resize((size_image, size_image)))
                      #  colorlist.append(image)

                    result = np.round(sess.run(y_,feed_dict={x: colorlist,keep_prob: 1.0}),3)
                    stationnum = sess.run(tf.argmax(result,1))

                    for re,snum in zip(result,stationnum):
                        print('station {0} ,\n station number is {1}'.format(re,snum))
                        print('result {0}'.format(sess.run(tf.equal(label,snum))))
                    print(sess.run(tf.reduce_mean(tf.cast(tf.equal(labels,stationnum),tf.float32))))


    def liststepidentification(self,n_class,size_image,model,dir,labels,epoch):
        
        if not os.path.exists("save_files"):
            print('Please train')
        else:
            sess = tf.InteractiveSession()

            dirlist = os.listdir(dir)
            denominbator=0
            childlist = []
            for i in dirlist:
                samplelist=os.listdir(dir+'/'+i)
                childlist.append(list(map(lambda x:i+'/'+x ,samplelist)))
                denominbator=denominbator+len(samplelist)
            #testlabel = [[labels[i]]*len(childlist[i]) for i in range(len(dirlist))]
            testlabel = [[[j]*len(childlist[i])for j in labels[i]] for i in range(len(dirlist))]
            print(testlabel)
            #print(testlabel)

            x=tf.placeholder(tf.float32,shape=[None,size_image,size_image,3])
            keep_prob=tf.placeholder(tf.float32)
            y_conv=self.model(x,n_class,keep_prob,model)
            
            y_=tf.nn.softmax(y_conv)

            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                for i in range(epoch//100):
                    saver.restore(sess,"save_files/model.ckpt-"+str((i+1)*100))
                    print('step'+str((i+1)*100))
        
                    correctans=0
                    for sdir,slabel,name in zip(childlist,testlabel,dirlist):
                        colorlist=[np.array(Image.open(dir+"/"+img).convert("RGB").resize((size_image, size_image))) for img in sdir]

                        result = np.round(sess.run(y_,feed_dict={x: colorlist,keep_prob: 1.0}),3)
                        stationnum = sess.run(tf.argmax(result,1))
                    
                        childanses=(tf.equal(stationnum,slabel))
                    
                        childanses=list(sess.run((tf.cast(childanses,tf.float32))))
                    
                        childans=(tf.add_n(childanses))
                        correctans=correctans+tf.reduce_sum(childans)
                    
                        print(' {0} accurancy: {1}'.format(name,sess.run(
                        tf.reduce_mean(childans))))
                
                    print(' All accurancy: {0}'.format(sess.run(correctans/denominbator)))