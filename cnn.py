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
  stations_train.tfrecords
  stations_test.tfrecords
  cnn.py

$python3
>>>import cnn
>>>c = cnn.cnn()
>>>c.train(n_class=5,size_image=56)

実行後
~/logs/Project/train/(tfeventsファイル)
  save_files/(ckptファイル群)
  Station_samples/(省略)
  stations_train.tfrecords
  stations_test.tfrecords
  cnn.py

>>>c.test(n_class=5,size_image=56)

実行後
~/logs/Project/train/(tfeventsファイル)
               test/(tfeventsファイル)
  save_files/(ckptファイル群)
  Station_samples/(省略)
  stations_train.tfrecords
  stations_test.tfrecords
  cnn.py
"""

class cnn(object):

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

    #モデルの定義
    def model(self,images,n_class,keep_prob):
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

    #トレーニングを行う
    def train(self,n_class,size_image,log_dir='logs/Project/train'):

        sess = tf.InteractiveSession()

        self.make_dir(log_dir,False)
        #画像とラベルとドロップアウト層のパラメータのプレイスホルダを生成
        x = tf.placeholder(tf.float32, shape=[None,size_image,size_image,3])
        y_ = tf.placeholder(tf.float32, shape=[None,n_class])
        keep_prob = tf.placeholder(tf.float32)
        #画像をモデルにかける
        y_conv = self.model(x,n_class,keep_prob)
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
    def test(self,n_class,size_image,log_dir='logs/Project/test'):

        sess = tf.InteractiveSession()
        #画像とラベルとドロップアウト層のパラメータのプレイスホルダを生成
        x = tf.placeholder(tf.float32, shape=[None,size_image,size_image,3])
        y_ = tf.placeholder(tf.float32, shape=[None,n_class])
        keep_prob = tf.placeholder(tf.float32)
        #画像をモデルにかける
        y_conv = self.model(x,n_class,keep_prob)
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

    def identification(self,n_class,size_image,img):
        
        if not os.path.exists("save_files"):
            print('Please train')
        else:
            sess = tf.InteractiveSession()
            image = [np.array(Image.open(img).convert("RGB").resize((size_image, size_image)))]
            #image = tf.cast(image,tf.float32)
            #image = tf.reshape(image,tf.stack([1,size_image,size_image,3]))
            #print(image)

            x=tf.placeholder(tf.float32,shape=[None,size_image,size_image,3])
            keep_prob=tf.placeholder(tf.float32)
            y_conv=self.model(x,n_class,keep_prob)
            
            y_=tf.nn.softmax(y_conv)

            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                #for i in range(200):
                saver.restore(sess,"save_files/model.ckpt-20000")
                result = np.round(sess.run(y_,feed_dict={x: image,keep_prob: 1.0}),3)
                stationnum = sess.run(tf.argmax(result,1))
                print('station {0} ,\n station number is {1}'.format(result,stationnum))

                

    def listidentification(self,n_class,size_image,dir):
        if not os.path.exists("save_files"):
            print('Please train')
        else:
            if not os.path.exists(dir):
                print('No directry')
            else:
                imglist=os.listdir(dir)
                x=tf.placeholder(tf.float32,shape=[None,size_image,size_image,3])
                keep_prob=tf.placeholder(tf.float32)
                y_conv=self.model(x,n_class,keep_prob)
            
                y_=tf.nn.softmax(y_conv)

                saver = tf.train.Saver()
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    saver.restore(sess,"save_files/model.ckpt-20000")

                    for img in imglist:

                        image = [np.array(Image.open(dir+"/"+img).convert("RGB").resize((size_image, size_image)))]

                        result = np.round(sess.run(y_,feed_dict={x: image,keep_prob: 1.0}),3)
                        stationnum = sess.run(tf.argmax(result,1))
                        print('station {0} ,\n station number is {1}'.format(result,stationnum))

    