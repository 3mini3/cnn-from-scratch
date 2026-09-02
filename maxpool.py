import numpy as np  # プーリングは領域ごとの max なので、配列の軸方向の最大値計算に NumPy を使う

class MaxPool2:  # 2×2 の最大値プーリング層。空間方向のダウンサンプリングを行う
  # プールサイズ 2、ストライド 2（非重複）の Max Pooling

  def iterate_regions(self, image):  # 特徴マップを重ならない 2×2 ブロックに分割する
    '''
    非重複の 2×2 領域を順に生成する。
    ストライド＝プールサイズなので、理論上のダウンサンプリング率がちょうど 1/2 になる。
    image は (H, W, C) の 3 次元配列（畳み込みの出力特徴マップ）。
    '''
    h, w, _ = image.shape  # 空間サイズとチャンネル数。プーリングはチャンネルごとに独立に行う
    new_h = h // 2  # 出力高さ。2 で割って切り捨て（奇数なら最後の 1 行は捨てる）
    new_w = w // 2  # 出力幅も同様

    for i in range(new_h):  # 出力の行インデックス。入力では 2i から始まるブロックに対応
      for j in range(new_w):  # 出力の列インデックス
        im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]  # 入力上の 2×2×C ブロック（非重複の局所領域）
        yield im_region, i, j  # ブロックと、それが写る出力座標

  def forward(self, input):  # 順伝播。各 2×2 の最大値だけを残して解像度を半分にする
    '''
    Max Pooling の順伝播。
    各チャンネル・各 2×2 窓で max を取り、平行移動に対して鈍感な（弱い平行移動不変性の）表現にする。
    戻り値の形は (H/2, W/2, C)。
    input は (H, W, C)。
    '''
    self.last_input = input  # 逆伝播で「どの画素が max だったか」を復元するため、入力を保存する

    h, w, num_filters = input.shape  # 入力の空間サイズとチャンネル数（フィルタ枚数）
    output = np.zeros((h // 2, w // 2, num_filters))  # 半分の解像度の特徴マップを用意する

    for im_region, i, j in self.iterate_regions(input):  # 非重複 2×2 ブロックを走査
      output[i, j] = np.amax(im_region, axis=(0, 1))  # 高さ・幅方向の max。チャンネル軸は残す（チャンネルごとに別の勝者）

    return output  # ダウンサンプル後の特徴マップ。最も強い応答だけが次層へ進む

  def backprop(self, d_L_d_out):  # 逆伝播。max を取った画素にだけ勾配を流す（勝者独占）
    '''
    Max Pooling の逆伝播。
    max はほぼ至るところで微分可能で、∂max/∂x_k は「x_k が最大なら 1、そうでなければ 0」。
    したがって ∂L/∂X は、勝者画素に ∂L/∂Y をコピーし、他は 0 にするルーティングになる。
    d_L_d_out はプーリング出力と同じ形の ∂L/∂Y。
    '''
    d_L_d_input = np.zeros(self.last_input.shape)  # 入力と同じ形の勾配。最初は全て 0（敗者画素の勾配）

    for im_region, i, j in self.iterate_regions(self.last_input):  # 順伝播と同じ 2×2 ブロックを再生成
      h, w, f = im_region.shape  # ブロックの高さ・幅・チャンネル数（通常は 2, 2, C）
      amax = np.amax(im_region, axis=(0, 1))  # チャンネルごとの最大値。これが順伝播で通した値

      for i2 in range(h):  # ブロック内の行（0 or 1）
        for j2 in range(w):  # ブロック内の列（0 or 1）
          for f2 in range(f):  # チャンネル。プーリングはチャンネル間で混ぜない
            # この画素がチャンネル f2 の max なら、出力勾配をここへコピーする（argmax の劣勾配）
            if im_region[i2, j2, f2] == amax[f2]:  # 同点なら両方に勾配が流れる（実装上の劣勾配の取り方）
              d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]  # 出力座標 (i,j) の勾配を入力の勝者位置へ配置

    return d_L_d_input  # 畳み込み層へ渡す ∂L/∂X。空間サイズはプーリング前に戻っている
