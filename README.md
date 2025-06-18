# tc2025_shysgmt






### 環境構築

#### Power shellでpoetoryをインストール 
```
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python
```

#### コマンドラインで環境構築

プロジェクトフォルダ内に.venvを作成するように設定する
```
C:\Users\User1\Code\tc2025_shysgmt> poetry config virtualenvs.in-project true --local
```

```
C:\Users\User1\Code\tc2025_shysgmt>poetry env use C:\Users\User1\AppData\Local\Programs\Python\Python312\python.exe
```


```
C:\Users\User1\Code\tc2025_shysgmt> poetry install
```


```
C:\Users\User1\Code\tc2025_shysgmt> poetry run pip install ipykernel
C:\Users\User1\Code\tc2025_shysgmt> poetry run python -m ipykernel install --user --name illusion-of-control
```
