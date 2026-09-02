import numpy as np  # 畳み込みは局所領域とフィルタの内積の繰り返しなので、配列演算に NumPy を使う

'''
この実装では入力を 2 次元配列（高さ×幅）と仮定している。
MNIST は 28×28 のグレースケールなのでチャンネル次元がなく、CNN の最初の層としてそのまま使える。
理論上の畳み込み層は入力も (H, W, C) の 3 次元テンソルで、フィルタは (K_h, K_w, C_in, C_out) になる。
より深い CNN で Conv を重ねるなら、入力を 3 次元にしてチャンネル方向の和も取る必要がある。
'''

class Conv3x3:  # 3×3 カーネルによる畳み込み層。局所受容野と重み共有を実装する
  # フィルタサイズ 3×3 の畳み込み層（ストライド 1、valid padding）

  def __init__(self, num_filters):  # 出力チャンネル数（フィルタ枚数）を受け取る
    self.num_filters = num_filters  # 何種類の特徴検出器を持つか。これが出力の深さになる

    # filters の形は (num_filters, 3, 3)。各スライスが 1 枚の 3×3 カーネル
    # 9 で割るのはフィルタ要素数でスケールし、初期値の分散を抑えるため（Xavier 初期化に近い）
    self.filters = np.random.randn(num_filters, 3, 3) / 9  # 学習対象のカーネル。全位置で同じ重みを使う（重み共有）

  def iterate_regions(self, image):  # 入力画像上を 3×3 の局所受容野で走査するジェネレータ
    '''
    valid padding（パディングなし）で、フィルタが画像内に収まる全ての 3×3 領域を生成する。
    ストライド 1 なので、隣接する領域は 1 画素だけずれる。
    image は 2 次元配列（H, W）。
    '''
    h, w = image.shape  # 入力の空間サイズ。出力は (h-3+1, w-3+1) = (h-2, w-2) になる

    for i in range(h - 2):  # 縦方向。カーネル高さ 3 がはみ出さない範囲だけ回す（valid）
      for j in range(w - 2):  # 横方向も同様。これが特徴マップ上の位置 (i, j) に対応する
        im_region = image[i:(i + 3), j:(j + 3)]  # 位置 (i, j) を左上とする 3×3 パッチ（局所受容野）
        yield im_region, i, j  # パッチと、それが写る出力座標を返す

  def forward(self, input):  # 順伝播。入力画像から特徴マップを作る
    '''
    与えた入力で畳み込みの順伝播を行う。
    各位置で「パッチ ⊙ フィルタ」の総和を取り、フィルタごとの特徴マップを積む。
    戻り値の形は (h-2, w-2, num_filters)。
    input は 2 次元配列。
    '''
    self.last_input = input  # 逆伝播で ∂L/∂W = X * ∂L/∂Y を計算するため、入力 X を保存する

    h, w = input.shape  # 入力空間サイズ
    output = np.zeros((h - 2, w - 2, self.num_filters))  # 特徴マップを 0 初期化。深さはフィルタ枚数

    for im_region, i, j in self.iterate_regions(input):  # 全空間位置について局所パッチを取り出す
      output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))  # パッチと全フィルタの要素積を 3×3 で足す＝相互相関（実装上の畳み込み）

    return output  # (H', W', F) の特徴マップ。各チャンネルが異なるフィルタの応答

  def backprop(self, d_L_d_out, learn_rate):  # 逆伝播。損失からフィルタへの勾配を求め、SGD で更新する
    '''
    畳み込み層の逆伝播。
    d_L_d_out は出力特徴マップと同じ形の ∂L/∂Y。
    線形和 Y_{i,j,f} = Σ_{u,v} X_{i+u,j+v} W_{f,u,v} より、∂L/∂W_f = Σ_{i,j} (∂L/∂Y_{i,j,f}) X_{patch(i,j)}。
    learn_rate は SGD の学習率 η。
    '''
    d_L_d_filters = np.zeros(self.filters.shape)  # 各フィルタの勾配を蓄積する配列。形はフィルタと同じ (F, 3, 3)

    for im_region, i, j in self.iterate_regions(self.last_input):  # 順伝播と同じパッチを再生成する（保存した入力から）
      for f in range(self.num_filters):  # フィルタごとに勾配を足す。チャンネルは独立
        d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region  # スカラー ∂L/∂Y_{i,j,f} をパッチに掛けて W_f へ加算（連鎖律）

    # 確率的勾配降下法: W ← W - η ∂L/∂W
    self.filters -= learn_rate * d_L_d_filters  # 学習したエッジ検出器などを少しずつ動かす

    # この CNN では Conv3x3 が最初の層なので、入力への勾配 ∂L/∂X は不要。
    # 手前に層がある場合は、フィルタを 180° 回転して ∂L/∂Y と full 畳み込みし、∂L/∂X を返す必要がある。
    return None  # 入力勾配は返さない（先頭層のため）
