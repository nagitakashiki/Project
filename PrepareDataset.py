#conding:utf-8
import http.client
import json
import re
import requests
import math
import urllib
import os
import cv2
import glob
import shutil
import numpy as np
import random
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

"""
実行例
ディレクトリ構造が以下の場合
~/Stations/0西口/(mp4ファイル群)
           1東口/(mp4ファイル群)
           2東京駅/(mp4ファイル群)
           3柏駅/(mp4ファイル群)
           4池袋駅/(mp4ファイル群)
           PrepareDataset.py
$python3
>>>import PrepareDataset as PD
>>>pd = PD.PrepareDataset()
>>>pd.capture(video="Stations/2東京駅/MOV_0094.mp4",dir="Station_samples/2東京駅",fname="東京駅00",fps=1)
>>>pd.split_sample(dir="Station_samples")
>>>pd.make_samples(width=224,height=224)
現在file=format=Trueにするとバグる模様.今後修正予定
実行後
~/Station_samples/0梅郷駅西口/(pingファイル群)
                  1梅郷駅西口/(pingファイル群)
                  2東京駅/(pingファイル群)
                  3柏駅/(pingファイル群)
                  4池袋駅/(pingファイル群)
  samples/test/0梅郷駅西口/(pingファイル群)
               1梅郷駅西口/(pingファイル群)
               2東京駅/(pingファイル群)
               3柏駅/(pingファイル群)
               4池袋駅/(pingファイル群)
          train/0梅郷駅西口/(pingファイル群)
                1梅郷駅西口/(pingファイル群)
                2東京駅/(pingファイル群)
                3柏駅/(pingファイル群)
                4池袋駅/(pingファイル群)
  Stations/(省略)
  stations_test.tfrecords
  stations_train.tfrecords
  PrepareDataset.py
"""

class PrepareDataset(object):

    #dirで指定されたパスが存在しない場合ディレクトリ作成
    def make_dir(self,dir,format=False):
        if not os.path.exists(dir):
            os.makedirs(dir)
        if format and os.path.exists(dir):
            shutil.rmtree(dir)

    #フレーム画像をキャプチャする
    def capture(self,video,dir,fname,fps=30,width=224,height=224,file_format=False):
        self.make_dir(dir,file_format)
        cap = cv2.VideoCapture(video)
        v_fps = round(cap.get(cv2.CAP_PROP_FPS))
        interval = round(v_fps/fps)
        i = 0
        print('capturing from video...')
        while(cap.isOpened()):
            flag,frame = cap.read()
            if (flag==False):
                break
            resized_frame = cv2.resize(img,(width,height))
            cv2.imwrite(os.path.join(dir,fname+str(i).zfill(5)+'.png'), resized_frame)
            i += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, interval*i)

        print('Done.')
        cap.release()

    def resize_dir(self,dir_img,dir_save,width=224,height=224,file_format=False):
        self.make_dir(dir_save,file_format)
        list_img = sorted(glob.glob(dir_img+'/*'))
        print('resizing images...')
        for f_img in list_img:
            img = cv2.imread(f_img)
            img_resize = cv2.resize(img,(width,height))
            cv2.imwrite(os.path.join(dir_save,os.path.splitext(os.path.basename(f_img))[0]+'.png'), img_resize)
        print('Done.')

    def convert_to_grayscale(self,dir_img,dir_save,file_format=False):
        self.make_dir(dir,file_format)
        list_img = sorted(glob.glob(dir_img+'/*'))
        print('converting to grayscale...')
        for f_img in list_img:
            img = cv2.imread(f_img)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(os.path.join(dir_save,os.path.splitext(os.path.basename(f_img))[0]+'_gray.png'), img_gray)
        print('Done.')

    def detect_edge(self,dir_img,dir_save,minVal,maxVal,file_format=False,sobel=3):
        self.make_dir(dir,file_format)
        list_img = sorted(glob.glob(dir_img+'/*'))
        print('detecting edge...')
        for f_img in list_img:
            img = cv2.imread(f_img)
            img_edge = cv2.Canny(img,minVal,maxVal,sobel)
            cv2.imwrite(os.path.join(dir_save,os.path.splitext(os.path.basename(f_img))[0]+'_edge.png'), img_edge)
        print('Done.')

    #一括操作
    def prepare_from_video(self,video,dir,fname,minVal,maxVal,file_format=False,fps=30,sobel=3):
        self.capture(video,dir,fname,file_format,fps)
        self.convert_to_grayscale(dir,dir+'_gray',file_format)
        self.detect_edge(dir,dir+'_edge',minVal,maxVal,file_format,sobel)

    #トレーニング/テストデータの分割
    def split_sample(self,dir,train_size=0.8,file_format=False,output_file="./samples"):
        samples=[]
        labels=[]
        files = os.listdir(dir)             #dir直下のディレクトリリストを取得

        self.make_dir(output_file,file_format)   #出力ディレクトリの生成
        self.make_dir(output_file+"/train",file_format)
        self.make_dir(output_file+"/test",file_format)

        for i in files:
            tmp=os.listdir(dir+"/"+i)
            samples+=list(map(lambda x:dir+"/"+i+"/"+x,tmp))#画像パスリストの取得
            labels+=[i]*len(tmp)                            #ラベルリストの生成

            #トレーニング,テストデータ格納ディレクトリの生成
            self.make_dir(output_file+"/train/"+i,file_format)
            self.make_dir(output_file+"/test/"+i,file_format)

        #トレーニング,テストデータを分割
        samples_train, samples_eval, labels_train, labels_eval = train_test_split(samples, labels, train_size=train_size)

        for i in range(len(samples_train)):#トレーニングデータのコピー
            shutil.copy(samples_train[i], output_file+"/train/"+labels_train[i])

        for i in range(len(samples_eval)):#テストデータのコピー
            shutil.copy(samples_eval[i], output_file+"/test/"+labels_eval[i])

    def make_TFR(self,rec_file_name,img_data):#画像→TFRecord
        with tf.python_io.TFRecordWriter(rec_file_name) as writer:
            for img_name,label in img_data:
                img_enc = img_name.encode('utf-8')
                record = tf.train.Example(features=tf.train.Features(feature={ #画像のパラメータ設定
                    "label": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[label])),
                    "image": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[img_enc])),
                }))
                writer.write(record.SerializeToString())            #書き込み

    def pull_num(self,str):
        i=1
        for i in range(len(str)):
            try:
                int(str[i])
            except ValueError:
                return(int(str[0:i]))
            i+=1
        return int(str)

    def image_lister(self,dir):
        img_data=[]
        files = os.listdir(dir)#dir直下のディレクトリ名リストを取得
        for i in files:
            img_files = os.listdir(dir+"/"+i)#ディレクトリ内部の画像名リストを取得
            label = self.pull_num(i)#ディレクトリ名をラベルに変換
            img_data += list(map(lambda x:[dir+"/"+i+"/"+x,label],img_files))#リストに追加
        return img_data

    def make_samples(self,width=0,height=0):
        img_data_train = self.image_lister("samples/train")
        record_file_train = './stations_train.tfrecords'
        self.make_TFR(record_file_train,img_data_train)
        img_data_test = self.image_lister("samples/test")
        record_file_test = './stations_test.tfrecords'
        self.make_TFR(record_file_test,img_data_test)

    def inflation(self,dir,file_format=False):
        self.crop(dir,dir,file_format)
        self.gamma(dir,dir,file_format)
        #self.blur(dir,dir+'_inf',file_format,1)
        self.affine(dir,dir,file_format,2)
        #self.enlarging(dir,dir+'_inf',file_format,1)
        #self.reducing(dir,dir+'_inf',file_format,1)

        #files = os.listdir(dir+'_inf')
        #for i in files:
        #    shutil.move(dir+'_inf/'+str(i), dir)
        #shutil.rmtree(dir+'_inf')

    def crop(self,dir_img,dir_save,file_format=False,num=3,rate=0.9):
        self.make_dir(dir_save,file_format)
        list_img = sorted(glob.glob(dir_img+'/*'))
        print('running crop...')
        for f_img in list_img:
            img = cv2.imread(f_img)
            height,width=img.shape[0:2]
            for i in range(1,num+1):
                hs=random.randint(0,round(height*(1-rate)))
                ws=random.randint(0,round(width*(1-rate)))
                reducing_img = img[hs:hs+round(height*rate),ws:ws+round(width*rate)]
                # 結果を出力
                cv2.imwrite(os.path.join(dir_save,os.path.splitext(os.path.basename(f_img))[0]+str(i).zfill(2)+'_red.png'), reducing_img)
        print('Done.')

    def gamma(self,dir_img,dir_save,file_format=False,num=5,lower=0.5,upper=2.0):
        # self.make_dir(dir_save)
        list_img = sorted(glob.glob(dir_img+'/*'))
        print('running gamma...')
        for f_img in list_img:
            img = cv2.imread(f_img)
            for i in range(1,num+1):
                high=True
                if random.randint(0,1)==1:
                    high=False
                # ガンマ変換ルックアップテーブル
                LUT = np.arange(256, dtype = 'uint8')
                tmp=1
                if high:
                    tmp = random.uniform(1,upper)
                else:
                    tmp=random.uniform(lower,1)
                for j in range(256):
                    LUT[j] = 255 * pow(float(j)/255, 1.0/tmp)
                # 代入
                gamma_img = cv2.LUT(img, LUT)
                cv2.imwrite(os.path.join(dir_save,os.path.splitext(os.path.basename(f_img))[0]+str(i).zfill(2)+'_gmh.png'), gamma_img)
        print('Done.')

    def blur(self,dir_img,dir_save,file_format=False,x=3,y=3):
        self.make_dir(dir_save,file_format)
        list_img = sorted(glob.glob(dir_img+'/*'))
        print('running blur...')
        for f_img in list_img:
            img = cv2.imread(f_img)
            for i in range(1,x+1):
                for j in range(1,y+1):
                    average_square = (2*i,2*j)
                    # 代入
                    blur_img = cv2.blur(img,average_square)
                    # 結果を出力
                cv2.imwrite(os.path.join(dir_save,os.path.splitext(os.path.basename(f_img))[0]+str(i).zfill(2)+'_blr.png'), blur_img)
        print('Done.')

    def affine(self,dir_img,dir_save,file_format=False,num=5,max_angle=10):
        self.make_dir(dir_save,file_format)
        list_img = sorted(glob.glob(dir_img+'/*'))
        print('running affine...')
        for f_img in list_img:
            img = cv2.imread(f_img)
            for i in range(1,num+1):
                # 画像のサイズ取得
                size = tuple([img.shape[1], img.shape[0]])
                # 回転の中心座標（画像の中心）
                center = tuple([int(size[0]/2), int(size[1]/2)])
                # 回転角度・拡大率
                angle=random.randint(-max_angle,max_angle)
                scale = 1+0.030*abs(angle)
                # 回転行列の計算
                R = cv2.getRotationMatrix2D(center, angle, scale)
                # アフィン変換
                affine_img = cv2.warpAffine(img, R, size, flags=cv2.INTER_CUBIC)
                # 結果を出力
                cv2.imwrite(os.path.join(dir_save,os.path.splitext(os.path.basename(f_img))[0]+str(i).zfill(2)+'_afl.png'), affine_img)
        print('Done.')

    def enlarging(self,dir_img,dir_save,file_format=False,num=3):
        self.make_dir(dir_save,file_format)
        list_img = sorted(glob.glob(dir_img+'/*'))
        print('running enlarging...')
        for f_img in list_img:
            img = cv2.imread(bf_img)
            for i in range(1,num+1):
                # リサイズ
                enlarging_img = cv2.resize(img, None, fx=1+0.05*i, fy=1+0.05*i, interpolation = cv2.INTER_LINEAR)
                # 結果を出力
                cv2.imwrite(os.path.join(dir_save,os.path.splitext(os.path.basename(f_img))[0]+str(i).zfill(2)+'_enl.png'), enlarging_img)
        print('Done.')

    def reducing(self,dir_img,dir_save,file_format=False,num=3):
        self.make_dir(dir_save,file_format)
        list_img = sorted(glob.glob(dir_img+'/*'))
        print('running reducing...')
        for f_img in list_img:
            img = cv2.imread(f_img)
            for i in range(1,num+1):
                # リサイズ
                reducing_img = cv2.resize(img, None, fx=1-0.05*i, fy=1-0.05*i, interpolation = cv2.INTER_AREA)
                # 結果を出力
                cv2.imwrite(os.path.join(dir_save,os.path.splitext(os.path.basename(f_img))[0]+str(i).zfill(2)+'_red.png'), reducing_img)
        print('Done.')


    def make_img_path(self,save_dir_path, url, cnt):
        save_img_path = os.path.join(save_dir_path, save_dir)
        make_dir(save_img_path)

        file_ext = os.path.splitext(url)[-1]
        if file_ext.lower() in ('.jpg'):
            full_path = os.path.join(save_img_path, '{0:03d}'.format(cnt) + file_ext.lower())
            return full_path
        else:
            raise ValueError('Not applicable file extension')


    def download_image(self,url, timeout=10):
        response = requests.get(url, allow_redirects=True, timeout=timeout)
        if response.status_code != 200:
            error = Exception("HTTP status: " + response.status_code)

        content_type = response.headers["content-type"]
        if 'image' not in content_type:
            error = Exception("Content-Type: " + content_type)
            raise error

        return response.content


    def save_image(self,filename, image):
        with open(filename, "wb") as fout:
            fout.write(image)

    def imgclorer(self,keyword,num_imgs_required,num_imgs_per_transaction,APIKEY):
        #APIKEY = 'xxxxxxxxxx'
        save_dir_path = './'
        #keyword = '東京駅'
        save_dir = keyword
        make_dir(save_dir_path)
        #num_imgs_required = 1000
        #num_imgs_per_transaction = 25
        offset_count = math.floor(num_imgs_required / num_imgs_per_transaction)

        url_list = []
        correspondence_table = {}

        headers = {
        # Request headers
            'Content-Type': 'multipart/form-data',
            'Ocp-Apim-Subscription-Key': APIKEY,
        }

        for offset in range(offset_count):
            params = urllib.parse.urlencode({
                'q': keyword,
                'mkt': 'ja-JP',
                'count': num_imgs_per_transaction,
                'offset': offset * num_imgs_per_transaction
            })

            try:
                conn = http.client.HTTPSConnection('api.cognitive.microsoft.com')
                conn.request("GET", "/bing/v7.0/images/search?%s" % params, "{body}", headers)
                response = conn.getresponse()
                data = response.read()

                conn.close()
            except Exception as err:
                print("[Errno {0}] {1}".format(err.errno, err.strerror))

            else:
                decode_res = data.decode('utf-8')
                data = json.loads(decode_res)

                for values in data['value']:
                    unquoted_url = urllib.parse.unquote(values['contentUrl'])
                    url_list.append(unquoted_url)

        for i, url in enumerate(url_list):
            try:
                img_path = make_img_path(save_dir_path, url, i)
                image = download_image(url)
                save_image(img_path, image)
                print('saved image... {}'.format(url))
            except KeyboardInterrupt:
                break
            except Exception as err:
                print("%s" % (err))
