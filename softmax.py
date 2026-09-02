import numpy as np  # 全結合の行列積と softmax の指数・正規化に NumPy を使う

class Softmax:  # 全結合層 + softmax 活性化。特徴ベクトルをクラス確率に変換する
  # 標準的な全結合層に、多クラス分類用の softmax を載せた出力層

  def __init__(self, input_len, nodes):  # 入力次元（平坦化後の特徴数）とクラス数を受け取る
    # input_len で割って初期値の分散を抑える（入力次元に対する Xavier 的なスケーリング）
    self.weights = np.random.randn(input_len, nodes) / input_len  # W の形は (入力次元, クラス数)。線形変換 z = xW + b の重み
    self.biases = np.zeros(nodes)  # バイアス b。最初は 0 で、各クラスのロジットを平行移動する

  def forward(self, input):  # 順伝播。特徴マップ → ベクトル → ロジット → 確率
    '''
    softmax 層の順伝播。
    1) 特徴マップを 1 次元に平坦化（Keras の Flatten に相当）
    2) 全結合でロジット z = xW + b を計算
    3) softmax で p_k = exp(z_k) / Σ_j exp(z_j) とし、クラス確率にする
    戻り値は長さがクラス数の 1 次元配列。
    input は任意形状（ここではプーリング後の (13,13,8)）。
    '''
    self.last_input_shape = input.shape  # 逆伝播で入力勾配を元の特徴マップ形状に戻すために保存

    input = input.flatten()  # (H, W, C) を長さ H*W*C のベクトルにする。空間構造はこの層では使わない
    self.last_input = input  # ∂L/∂W = x^T (∂L/∂z) に使うため、平坦化後の x を保存

    input_len, nodes = self.weights.shape  # 重み行列の形状から入力次元とクラス数を取る（計算自体には未使用）

    totals = np.dot(input, self.weights) + self.biases  # ロジット z。各クラスへの線形スコア（まだ確率ではない）
    self.last_totals = totals  # softmax のヤコビアン ∂p/∂z を逆伝播で使うため、z を保存

    exp = np.exp(totals)  # exp(z_k)。softmax の分子。値を非負にして大小関係を保つ
    return exp / np.sum(exp, axis=0)  # 合計 1 に正規化して確率分布 p を得る（多クラスの出力層）

  def backprop(self, d_L_d_out, learn_rate):  # 逆伝播。交差エントロピーの勾配を全結合のパラメータと入力へ分解する
    '''
    softmax 層の逆伝播。
    d_L_d_out は ∂L/∂p。交差エントロピー L = -log(p_c) なら、c 番目だけ -1/p_c で他は 0。
    softmax の微分は ∂p_i/∂z_k = p_i (δ_{ik} - p_k) なので、それを連鎖律で ∂L/∂z に合成する。
    その後、z = xW + b の線形関係から ∂L/∂W, ∂L/∂b, ∂L/∂x を求める。
    learn_rate は SGD の学習率 η。
    '''
    # 交差エントロピーでは ∂L/∂p の非零成分は正解クラスの 1 つだけ。その成分だけで十分
    for i, gradient in enumerate(d_L_d_out):  # i がクラス番号、gradient が ∂L/∂p_i
      if gradient == 0:  # 不正解クラスは ∂L/∂p_i = 0 なので、ヤコビアンを掛けても寄与しない
        continue  # 正解クラスの項だけ処理する

      # e^{z}（softmax の分子を再計算）
      t_exp = np.exp(self.last_totals)  # 順伝播の exp(z)。p = t_exp / S と書ける

      # Σ_k e^{z_k}（softmax の分母）
      S = np.sum(t_exp)  # 正規化定数。∂p/∂z の式に S が現れる

      # 出力 p_i の、全ロジット z に対する勾配（softmax ヤコビアンの第 i 行）
      d_out_d_t = -t_exp[i] * t_exp / (S ** 2)  # k ≠ i のとき ∂p_i/∂z_k = -e^{z_i} e^{z_k} / S^2
      d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)  # k = i のとき ∂p_i/∂z_i = e^{z_i}(S - e^{z_i}) / S^2

      # ロジット z の、重み・バイアス・入力に対する勾配（線形層 z = xW + b）
      d_t_d_w = self.last_input  # ∂z_k/∂W_{jk} = x_j。つまり ∂z/∂W は入力ベクトル x
      d_t_d_b = 1  # ∂z_k/∂b_k = 1
      d_t_d_inputs = self.weights  # ∂z/∂x = W。入力への勾配は重みを通して伝わる

      # 損失のロジットに対する勾配 ∂L/∂z = (∂L/∂p_i) * (∂p_i/∂z)（連鎖律）
      d_L_d_t = gradient * d_out_d_t  # 長さがクラス数のベクトル。ここから先は普通の全結合の逆伝播

      # 損失の重み・バイアス・入力に対する勾配
      d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]  # ∂L/∂W = x^T (∂L/∂z)。外形積で (入力次元, クラス数) になる
      d_L_d_b = d_L_d_t * d_t_d_b  # ∂L/∂b = ∂L/∂z（バイアスは 1 を掛けるだけ）
      d_L_d_inputs = d_t_d_inputs @ d_L_d_t  # ∂L/∂x = W (∂L/∂z)。次にプーリング層へ渡す勾配

      # SGD: W ← W - η ∂L/∂W、b ← b - η ∂L/∂b
      self.weights -= learn_rate * d_L_d_w  # どの特徴がどのクラスを支持するかを更新
      self.biases -= learn_rate * d_L_d_b  # クラスごとの事前的な偏りを更新

      return d_L_d_inputs.reshape(self.last_input_shape)  # 平坦化を元に戻し、プーリング出力と同じ (H, W, C) の勾配にする
