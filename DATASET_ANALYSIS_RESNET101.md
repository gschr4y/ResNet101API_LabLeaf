# Analise do dataset combinado para ResNet-101

## Fontes usadas

O pipeline agora usa a pasta `DataSets` como fonte principal. Foram combinadas duas estruturas:

| Fonte | Imagens validas brutas |
| --- | ---: |
| `DataSets/<classe>/` | 9648 |
| `DataSets/data_clean/train|val|test/<classe>/` | 524 |
| **Total bruto** | **10172** |

O pente fino removeu 65 variantes logicas, principalmente pares `.bmp`/`.jpg` da mesma folha no `data_clean`. Nao foram encontradas imagens invalidas, hashes duplicados exatos nem duplicatas exatas em classes diferentes.

## Dataset final

O novo `data_resnet101` ficou assim:

| Split | Imagens |
| --- | ---: |
| train | 7077 |
| val | 1515 |
| test | 1515 |
| **Total** | **10107** |

Classes finais:

| Classe | Total |
| --- | ---: |
| bacterial_blight | 534 |
| brown_spot | 54 |
| cercospora_leaf_blight | 1598 |
| crestamento | 5 |
| downey_mildew | 652 |
| ferrugem | 31 |
| frogeye | 1540 |
| healthy | 1632 |
| mosaic_virus | 12 |
| potassium_deficiency | 1034 |
| powdery_mildew | 30 |
| septoria | 21 |
| southern_blight | 36 |
| soybean_rust | 1627 |
| sudden_death_syndrome | 110 |
| target_spot | 1081 |
| yellow_mosaic | 110 |

## Ajustes feitos

- `prepare_resnet101_dataset.py` agora aceita fontes mistas: pastas diretas de classe e datasets ja divididos em `train/val/test`.
- Classes com espacos ou typos foram normalizadas para nomes estaveis, por exemplo `Mossaic Virus` -> `mosaic_virus` e `Sudden Death Syndrone` -> `sudden_death_syndrome`.
- O script remove duplicatas por hash exato e tambem por ID logico de imagem, evitando que `.bmp` e `.jpg` da mesma folha virem exemplos separados.
- O novo split e estratificado por classe.
- O relatorio completo fica em `data_resnet101/split_report.json`.

## Validacao

Checagens executadas no dataset final:

- imagens invalidas: 0;
- grupos com mesmo hash cruzando `train/val/test`: 0;
- total de imagens finais: 10107;
- smoke test de treino com 17 classes na RX 7600 via DirectML: OK;
- smoke test de predicao com o checkpoint combinado: OK.

## Comandos

Preparar novamente o dataset combinado:

```powershell
py .\prepare_resnet101_dataset.py --source-dir DataSets --output-dir data_resnet101 --overwrite
```

Treino recomendado na AMD Radeon RX 7600 usando DirectML:

```powershell
py .\train_resnet101.py --data-dir data_resnet101 --output-dir outputs_resnet101 --epochs 30 --batch-size 8 --device dml --optimizer sgd --learning-rate 0.01
```

Treino mais conservador se a VRAM ficar apertada:

```powershell
py .\train_resnet101.py --data-dir data_resnet101 --output-dir outputs_resnet101 --epochs 30 --batch-size 4 --device dml --optimizer sgd --learning-rate 0.01
```

Predicao:

```powershell
py .\predict_resnet101.py --checkpoint outputs_resnet101\best_resnet101.pt --image DataSets\healthy\healthy_1.jpg --device dml
```

## Observacao importante

Algumas classes novas sao muito pequenas, especialmente `crestamento` com 5 imagens, `mosaic_virus` com 12 e `septoria` com 21. O treino usa pesos por classe por padrao, mas essas classes ainda terao maior risco de instabilidade. O resultado final deve ser avaliado pelo `test`, nao apenas pela acuracia de validacao.
