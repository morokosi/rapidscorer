# rapidscorer

RapidScorer
https://ai.stanford.edu/~wzou/kdd_rapidscorer.pdf

の実験を再現したい

- XGboostではなく、LightGBMの評価
- MSLR-WEB10Kデータセットのみでの評価
- QuickScorer, Vectorized-QuickScorer, LightGBM, RapidScorer
- SSE, AVX256, AVX512の評価

## 評価用モデルの作成

- MSLR-WEB10K/ ディレクトリのデータセットをもとに、lightgbmモデルを学習する
- pythonで実装

## LightGBMの推論時間評価

- Cで実装
- lightgbmのpredictを呼び出す
- 推論用のデータセットを読み込み、1行あたりの推論時間を評価するプログラムを作成

## QuickScorer, Vectorized-QuickScorerの実装

https://iris.unive.it/bitstream/10278/3703670/8/paper.pdf

- 論文を元に、QuickScorer / Vectorized-QuickScorerをCで実装
- 1行あたりの推論時間を評価するプログラムを作成
- LightGBMと推論結果が一致することを確認するプログラムを作成

## RapidScorerの実装

- Cで実装
- 1行あたりの推論時間を評価するプログラムを作成
- LightGBMと推論結果が一致することを確認するプログラムを作成


