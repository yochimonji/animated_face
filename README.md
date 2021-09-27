# Human to Animated Face

2021年度実践プログラミングゼミ3班のHuman to Animated Faceのリポジトリです。
リアルタイムで人の顔をアニメの顔に変換する予定でした。

## 実行方法
- cuda driver が入ったubuntu (ほかの環境は試してません)
- 以下のコマンドを実行  
```
git clone https://github.com/yochimonji/human_to_animated_face.git
cd human_to_animated_face
conda env create -f env.yml
conda activate face2anime
```
- [この記事の設定](https://qiita.com/gotta_dive_into_python/items/e17fa9048ceecc55df67)
- ```python main.py```で実行
