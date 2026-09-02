import mnist  # MNIST 手書き数字データセット。28×28 グレースケール画像と 0〜9 のラベル
import numpy as np  # 正規化、交差エントロピー、勾配ベクトルの計算に使う
from conv import Conv3x3  # 局所特徴を抽出する畳み込み層
from maxpool import MaxPool2  # 空間解像度を下げ、弱い平行移動不変性を入れるプーリング層
from softmax import Softmax  # 特徴を 10 クラスの確率に変換する出力層

# フル MNIST だとフルスクラッチ実装が遅いので、学習・テストとも先頭 1000 枚だけ使う
# 枚数を増やせば精度は上がりやすいが、計算時間も増える
train_images = mnist.train_images()[:1000]  # 学習画像。形は (1000, 28, 28)、画素は 0〜255
train_labels = mnist.train_labels()[:1000]  # 学習ラベル。0〜9 の整数（one-hot にはしない）
test_images = mnist.test_images()[:1000]  # テスト画像。学習に使わないデータで汎化を測る
test_labels = mnist.test_labels()[:1000]  # テストラベル

conv = Conv3x3(8)                  # 28×28×1 → 26×26×8。8 枚の 3×3 フィルタ、valid padding で 2 画素縮む
pool = MaxPool2()                  # 26×26×8 → 13×13×8。2×2 max pool で空間サイズが半分
softmax = Softmax(13 * 13 * 8, 10) # 13×13×8 → 10。平坦化 1352 次元を 10 クラスへ全結合 + softmax

def forward(image, label):  # 1 枚の画像について順伝播し、損失と正誤を返す
  '''
  CNN の順伝播を完了し、交差エントロピー損失と正誤（0 or 1）を計算する。
  アーキテクチャは Conv → Pool → Softmax の 3 段。
  image は 2 次元配列（28×28）。
  label は正解の数字 0〜9。
  '''
  # 画素を [0,255] から [-0.5, 0.5] へ線形変換する。平均を 0 付近にすると勾配が安定しやすい
  # これは前処理の定石で、学習対象のパラメータではない
  out = conv.forward((image / 255) - 0.5)  # 正規化画像を畳み込み、局所特徴マップを得る
  out = pool.forward(out)  # 特徴マップをダウンサンプルし、位置の細かいずれを吸収する
  out = softmax.forward(out)  # 特徴を 10 次元の確率分布 p に変換する

  # 交差エントロピー L = -log(p_label)。正解クラスの確率だけを使い、自然対数
  loss = -np.log(out[label])  # p_label が 1 に近いほど損失は 0 に近づく
  acc = 1 if np.argmax(out) == label else 0  # 最大確率のクラスが正解なら 1（ハード予測の正誤）

  return out, loss, acc  # 確率ベクトル、損失、正誤。学習では確率も勾配の起点に使う

def train(im, label, lr=.005):  # 1 サンプルの SGD 更新（バッチサイズ 1 のオンライン学習）
  '''
  1 枚の画像で順伝播→損失の勾配→逆伝播→パラメータ更新まで行う。
  交差エントロピーと softmax を組み合わせ、∂L/∂p から層を逆順に辿る。
  戻り値は交差エントロピー損失と正誤。
  im は 2 次元配列、label は数字、lr は学習率 η。
  '''
  # 順伝播で予測確率 p と損失を得る（この時点では重みはまだ更新しない）
  out, loss, acc = forward(im, label)  # out が softmax の出力 p

  # 損失 L = -log(p_c) の、確率 p に対する勾配。c は正解クラス
  gradient = np.zeros(10)  # ∂L/∂p_k は k≠c なら 0
  gradient[label] = -1 / out[label]  # ∂L/∂p_c = -1/p_c。ここが誤差逆伝播の最初の信号

  # 逆伝播: 出力層 → プーリング → 畳み込み、の順に勾配を流し、学習可能な層はここで SGD 更新する
  gradient = softmax.backprop(gradient, lr)  # ∂L/∂p から ∂L/∂z、さらに ∂L/∂x（プーリング出力）へ。W,b も更新
  gradient = pool.backprop(gradient)  # max の勝者画素へだけ勾配をコピー。学習パラメータはない
  gradient = conv.backprop(gradient, lr)  # 特徴マップ勾配からフィルタ勾配を積み、カーネルを更新

  return loss, acc  # この 1 ステップの損失と正誤（ログ用）

print('MNIST CNN initialized!')  # 層の初期化が終わったことを表示

# 学習データを 3 エポック回す。1 エポック＝全学習サンプルを 1 周
for epoch in range(3):  # 同じ 1000 枚を 3 回見せる。回数を増やすと過学習のリスクも出る
  print('--- Epoch %d ---' % (epoch + 1))  # 人間向けのエポック番号は 1 始まり

  # 毎エポックでサンプル順をシャッフルする。SGD が同じ順序の相関に引っ張られないようにする
  permutation = np.random.permutation(len(train_images))  # 0..N-1 のランダムな並べ替え
  train_images = train_images[permutation]  # 画像を同じ順で並べ替える
  train_labels = train_labels[permutation]  # ラベルも対応を崩さないよう同じ permutation で並べ替える

  # 1 枚ずつ train() を呼び、オンライン SGD で更新する
  loss = 0  # 直近 100 ステップの損失合計（ログ用）
  num_correct = 0  # 直近 100 ステップの正解数（ログ用）
  for i, (im, label) in enumerate(zip(train_images, train_labels)):  # 画像とラベルをペアで 1 枚ずつ取り出す
    if i % 100 == 99:  # 100 枚ごと（i=99,199,...）に平均損失と精度を表示
      print(  # 学習の進みを監視する（損失が下がるか、精度が上がるか）
        '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %  # 直近 100 枚の平均損失と正解率
        (i + 1, loss / 100, num_correct)  # i+1 が処理済み枚数。num_correct は 100 枚中の正解数＝パーセント
      )  # 100 ステップ分の平均交差エントロピーと、百分率表示した正解数を出力
      loss = 0  # 次の 100 枚に向けて累積をリセット
      num_correct = 0  # 正解数もリセット

    l, acc = train(im, label)  # この 1 枚で順伝播・逆伝播・更新を実行
    loss += l  # ログ用に損失を足す
    num_correct += acc  # 正解なら 1 が足される

# 学習に使っていないテストデータで、更新なしの順伝播だけ行い汎化を評価する
print('\n--- Testing the CNN ---')  # 評価フェーズの開始
loss = 0  # テスト損失の合計
num_correct = 0  # テスト正解数
for im, label in zip(test_images, test_labels):  # テスト 1000 枚を 1 枚ずつ評価（重みは動かさない）
  _, l, acc = forward(im, label)  # forward のみ。train と違い backprop しない
  loss += l  # テスト損失を累積
  num_correct += acc  # テスト正解を累積

num_tests = len(test_images)  # テスト枚数（1000）。平均を取る分母
print('Test Loss:', loss / num_tests)  # 1 枚あたりの平均交差エントロピー
print('Test Accuracy:', num_correct / num_tests)  # 正解率（0〜1）。ランダムなら約 0.1
