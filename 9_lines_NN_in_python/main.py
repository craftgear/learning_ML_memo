# -*- coding: utf-8 -*-

# The original code is here https://github.com/miloharper/simple-neural-network

from numpy import exp, array, random, dot

class NeuralNetwork():

    """Docstring for NeuralNetwork. """

    def __init__(self):
        # ランダム関数にシードを与え、毎回同じ数を生成するようにする
        random.seed(1)

        # 三つの入力と一つの出力を持つ一層のニューロンをモデル化する
        # ランダムな三つの数値をウェイトとして3行1列の行列に割り当てる
        # ウェイトの値は中央値が0で−1から1の値を取る
        self.synaptic_weights = 2 * random.random((3,1)) -1

    # アクティベーション関数は、sigmoid関数を使う。この関数はS字カーブを描く。
    # 入力値にウェイトを書けた値を合計したものをこの関数に与え、0から1の間に標準化する
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # sigmoid関数の派生関数。
    # sigmoidカーブの傾き。
    # 現在のウェイトについてどれほど信頼が置けるかを示す
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # 試行錯誤を通じてニューラルネットを訓練する。
    # 毎回ウェイトの値を調整する
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            # 訓練用データセットをニューラルネット(一層)に渡す
            output = self.think(training_set_inputs)

            # 誤差を計算する。（ニューラルネットの計算結果と、期待する結果の差異)
            error = training_set_outputs - output

            # 誤差に入力値をかけ、次にSigmoidカーブの傾きをかける。
            # これによってより信頼度の低いウェイトがより多くの修正を受ける
            # つまり、入力値がゼロの時、ウェイトに変化はない
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # ウェイトの調整
            self.synaptic_weights += adjustment

    # ニューラルネットワークは思考できる
    def think(self, inputs):
        # 一層のニューロンに入力値を渡す
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

if __name__ == "__main__":

    # 一層のニューラルネットワークを初期化
    neural_network = NeuralNetwork()

    print "Random starting synaptic weights: "
    print neural_network.synaptic_weights

    # 訓練用データセットは4つ。それぞれに3つの入力値があり、出力はひとつ
    training_set_inputs = array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
    training_set_outputs = array([[0,1,1,0]]).T

    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print "New synaptic weights after training: "
    print neural_network.synaptic_weights

    # 別の入力値でニューラルネットワークを試してみる
    print "Considering new situation [1, 0, 0] -> ?:"
    print neural_network.think(array([1,0,0]))

