# API ResNet101 Soja no Render

Esta API usa FastAPI para classificar imagens de folhas de soja com o checkpoint treinado.

## Arquivos principais

- `main.py`: API de inferencia.
- `requirements.txt`: dependencias do Render.
- `render.yaml`: configuracao de deploy.
- `train_resnet101.py`: script de treino.
- `predict_resnet101.py`: predicao local por linha de comando.

## Endpoints

- `GET /health`: status da API e do modelo.
- `GET /classes`: lista das classes.
- `POST /predict`: recebe uma imagem via `multipart/form-data`.
- `POST /predict-base64`: recebe imagem em base64.
- `GET /docs`: Swagger UI automatico do FastAPI.

## Rodar localmente

```powershell
py -m pip install -r requirements.txt
py -m uvicorn main:app --reload
```

Abra:

```text
http://127.0.0.1:8000/docs
```

## Testar localmente via PowerShell

```powershell
$img = "caminho\para\imagem.jpg"
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict?top_k=3" -Method Post -Form @{ file = Get-Item $img }
```

## Deploy no Render

No Render, crie um Web Service apontando para o repositorio.

Config padrao:

```text
Build Command: pip install -r requirements.txt
Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
Health Check Path: /health
```

Se o Render estiver rodando `uvicorn LabLeafAPI:app`, tambem funciona porque `LabLeafAPI.py`
aponta para o mesmo `app` de `main.py`. Ainda assim, o comando recomendado e `main:app`.

O arquivo `.pth` tem mais de 100 MB, entao normalmente voce nao deve subir ele direto no GitHub.
Suba o checkpoint em algum local com URL direta de download e configure no Render:

```text
MODEL_URL=https://sua-url-direta/best_resnet101.pt
```

A API baixa esse arquivo na inicializacao e carrega o modelo.

## Origem permitida

A API esta travada para aceitar requests de navegador somente do GitHub Pages:

```text
https://rafael-casarin.github.io
```

O site pode estar em `https://rafael-casarin.github.io/testeLab/`, mas o header `Origin` enviado pelo navegador contem apenas protocolo e dominio, sem o caminho `/testeLab/`. Por isso o CORS e o middleware validam `https://rafael-casarin.github.io`.

## Checkpoint recomendado

Use o melhor checkpoint produzido pelo treino completo:

```text
outputs_resnet101_full/best_resnet101.pt
```

Enquanto o treino completo ainda estiver rodando, o checkpoint parcial fica no mesmo caminho e vai sendo substituido quando a validacao melhora. Para deploy definitivo, use o arquivo final gerado apos o treino terminar.
