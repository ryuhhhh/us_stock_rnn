# 使用するデータ
## 特徴量
* 「横断的」 かつ 「時系列」
  - ある銘柄のある期間の値上がり率を取得(時系列)
    - 期間候補は1日ごとを5回(1週間)、1週間ごとを4回(1ヵ月)
  - それを"その期間"の他銘柄との値で標準化する(横断的)
    - 全期間合わせて標準化しない
    - その期間における"相対的な強さ"が分かる。
  - 取れれば同じように出来高も
  - 教師が率なので訓練も率の方が有力か

|  取得データ名  |  頻度  |
| ---- | ---- |
|  値上がり率(%)  |  1日毎 x 10 |
|  値上がり率(%)  |  1週間ごと x 4 |
|  1次近似の傾き  |  1週間ごと x 4  |
|  出来高  | 1日毎 x 10 |
|  出来高  | 1週間ごと x 4 |

## 教師
* 10日以内に10%あがったか
* 10日以内に5%あがったか

## ポイント
* seq2seq
  - input:過去N日 => output:N-1から明日
* ベースライン指標
  - 平均二乗誤差を使用
  ```
  y_pred = X_valid[:,-1]
  np.mean(keras.losses.mean_squared_error(y_valid,y_pred))
  ```
  - 全結合ネットワークを利用
  ```
  model = keras.models.Sequential([
      keras.layers.Flatten(input_shape=[50, 1]),
      keras.layers.Dense(1)
  ])
  # X_trainは70000,50,1
  model.compile(loss="mse", optimizer="adam")
  history = model.fit(X_train, y_train, epochs=20,
                      validation_data=(X_valid, y_valid))
  model.evaluate(X_valid, y_valid)

  ```
* 再帰層3重で1つ先を予測
  ```
  model = keras.models.Sequential([
    # どんな長さでもよいのでNone
    # return_sequenses=Trueでその層は全ての出力を返す
    ## そうしないとN,T,DではなくN,D(最後の値だけ入る)になる
    # 出力層は1つだけでいいのでなし
    keras.layers.SimpleRNN(20,return_sequenses=True,input_shape=[None,1]),
    keras.layers.SimpleRNN(20,return_sequenses=True),
    keras.layers.SimpleRNN(1)
  ])
  optimizer = keras.optimizers.Adam(lr=0.005)
  model.compile(loss="mse", optimizer=optimizer)
  # y_trainは[N,1]
  history = model.fit(X_train, y_train, epochs=20,
                      validation_data=(X_valid, y_valid))
  ```

* 再帰層3重で10個先を予測
  ```
  model = keras.models.Sequential([
    # どんな長さでもよいのでNone
    # return_sequenses=Trueでその層は全ての出力を返す
    ## そうしないとN,T,DではなくN,D(最後の値だけ入る)になる
    # 出力層は1つだけでいいのでなし
    keras.layers.SimpleRNN(20,return_sequenses=True,input_shape=[None,1]),
    keras.layers.SimpleRNN(20),
    keras.layers.SimpleRNN(""10"")
  ])
  optimizer = keras.optimizers.Adam(lr=0.005)
  model.compile(loss="mse", optimizer=optimizer)
  # y_trainは[N,""10""]
  history = model.fit(X_train, y_train, epochs=20,
                      validation_data=(X_valid, y_valid))
  ```

* 再帰層3重で10個先を予測(最後の再帰層の出力だけではなくすべての再帰層の出力を使用)
  ```
  各出力には10個先の値までを用いる
  出力は10
  ```

* 層正規化
  - 特徴量次元で正規化 -> N,T,D の Dで正規化 -> 個々の時間で個々のインスタンスのために正規化
  - 入力と隠れ状態の線形結合(Wx・x+Wh・h+b)の直後で行われる
  ```
  class LNSimpleRNNCell(keras.layers.Layer):
      def __init__(self, units, activation="tanh", **kwargs):
          super().__init__(**kwargs)
          self.state_size = units
          self.output_size = units
          self.simple_rnn_cell = keras.layers.SimpleRNNCell(units,
                                                            activation=None)
          self.layer_norm = LayerNormalization()
          self.activation = keras.activations.get(activation)
      def call(self, inputs, states):
          # statesには前時間の2つの状態をもらう(LSTMでは短期状態と長期状態)
          # 線形結合を行う(SimpleRNNCellでは1つしか出力しないのでoutputs=new_states[0]となる)
          outputs, new_states = self.simple_rnn_cell(inputs, states)
          # outputsに対して層正規化
          norm_outputs = self.activation(self.layer_norm(outputs))
          # 出力は2種類,1つ目は出力,2つ目は隠れ状態
          return norm_outputs, [norm_outputs]
  ```
  ```
  model = keras.models.Sequential([
      keras.layers.RNN(LNSimpleRNNCell(20), return_sequences=True,
                       input_shape=[None, 1]),
      keras.layers.RNN(LNSimpleRNNCell(20), return_sequences=True),
      keras.layers.TimeDistributed(keras.layers.Dense(10))
  ])
  model.compile(loss="mse",
                optimizer="adam",
                metrics=[last_time_step_mse])
  history = model.fit(X_train, Y_train, epochs=20,validation_data=(X_valid, Y_valid))
    ```
* ドロップアウト
  - dropout と recurrent_dropoutがある

* LSTMやGRU(?)
 - 長い時系列(100単位時間以上など)だと苦戦する
 - 畳み込みの出番となる

* 畳み込みで情報量削減
  ```
  model = keras.models.Sequential([
      # Conv1Dで入力からスカラを生成 => どんなに長い時系列でも1つの20次元のシーケンスとなる
      keras.layers.Conv1D(filters=20, kernel_size=4, strides=2, padding="valid",input_shape=[None, 1]),
      # 中間層の次元は20
      keras.layers.GRU(20, return_sequences=True),
      keras.layers.GRU(20, return_sequences=True),
      keras.layers.TimeDistributed(keras.layers.Dense(10))
  ])
  ```

* WaveNet(畳み込み層重ね) => 音声合成タスクなど長い時系列に使用
  ```
  model = keras.models.Sequential([
  model.add(keras.layers.InputLayer(input_shape=[None,1]))
  for rate in (1,2,4,8) * 2:
    # causalで未来をのぞき見しないために,先頭にゼロパディングを追加
    # dilation_rateで飛ばす数を指定
    model.add(keras.layers.Conv1D(filters=20,
              kernel_size=2,
              padding='causal',
              activation='relu',
              dilation_rate=rate
              ))
  # 出力
  model.add(keras.layers.Conv1D(filters=10,kernel_size=1))
  model.compile(loss="mse",
                optimizer="adam",
                metrics=[last_time_step_mse])
  history = model.fit(X_train, Y_train, epochs=20,validation_data=(X_valid, Y_valid))
  ```