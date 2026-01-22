# tc2025_shysgmt






### 環境構築

#### python3.12.3をダウンロード


インストールしたPythonがどこにあるか確認（このパスをメモしておく）
C:\Users\User1> where python 
C:\Users\User1\AppData\Local\Programs\Python\Python312\python.exe
```
where python 
```

#### poetryをインストール
```
C:\Users\User1> pip install poetry
```

#### .venvの作成

プロジェクトフォルダに移動する。
```
C:\Users\User1> cd code
C:\Users\Sugimoto\Code> cd tc2025_shysgmt
C:\Users\User1\Code\tc2025_shysgmt>
```

プロジェクトフォルダ内に.venvを作成するように設定する
```
C:\Users\User1\Code\tc2025_shysgmt> poetry config virtualenvs.in-project true --local
```
pythonを指定して.venvを作成する（pythonが入っているフォルダを指定）
<br>例：C:\Users\User1\AppData\Local\Programs\Python\Python312\python.exe
<br>（<u>User1のところは自分のパソコンに合わせる</u>）
```
C:\Users\User1\Code\tc2025_shysgmt>poetry env use C:\Users\User1\AppData\Local\Programs\Python\Python312\python.exe
```

環境を作成する。
```
C:\Users\User1\Code\tc2025_shysgmt> poetry install
```
ipykernelをインストールする

```
C:\Users\User1\Code\tc2025_shysgmt> poetry run python -m ipykernel install --user --name tc2025_shysgmt
```
