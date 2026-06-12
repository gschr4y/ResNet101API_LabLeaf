# Auditoria do dataset ResNet18 soja

## Problema encontrado

Os resultados anteriores nao devem ser usados como metrica final. O dataset
original tinha vazamento entre treino e validacao:

- `data/train` continha imagens aumentadas offline com prefixo `aug_`.
- A classe `powdery_mildew` tinha versoes aumentadas no treino de imagens que
  tambem apareciam na validacao.
- Nao existia split `test/`, entao o script avaliava o melhor checkpoint no
  proprio `val` usado para escolher o modelo.
- O treino com `data_augmented_max` criou 50.000 imagens a partir de poucas
  imagens reais, o que infla a memorizacao e nao cria evidencia nova.

## Correcao aplicada

Foi criado o script `prepare_clean_dataset.py`, que reconstruiu um dataset novo
em `data_clean/` sem modificar a pasta `data/` original.

Regras usadas:

- ignorar imagens com prefixo `aug_`;
- remover nomes-base que aparecem em classes diferentes;
- manter o mesmo nome-base sempre no mesmo split;
- criar `train/`, `val/` e `test/`;
- salvar um relatorio em `data_clean/clean_dataset_report.json`.

## Novo split

O dataset limpo ficou com 524 imagens reais:

| Split | Total |
| --- | ---: |
| train | 367 |
| val | 77 |
| test | 80 |

Contagens por classe:

| Classe | Train | Val | Test |
| --- | ---: | ---: | ---: |
| Mossaic Virus | 8 | 2 | 2 |
| Southern blight | 25 | 5 | 6 |
| Sudden Death Syndrone | 77 | 16 | 17 |
| Yellow Mosaic | 77 | 16 | 17 |
| bacterial_blight | 62 | 13 | 13 |
| brown_spot | 57 | 12 | 12 |
| crestamento | 3 | 1 | 1 |
| ferrugen | 22 | 5 | 4 |
| powdery_mildew | 21 | 4 | 5 |
| septoria | 15 | 3 | 3 |

## Validacao da limpeza

Checagens feitas em `data_clean/`:

- arquivos com prefixo `aug_`: 0;
- mesmo nome-base cruzando `train/val/test`: 0;
- hash exato duplicado cruzando `train/val/test`: 0;
- nome-base aparecendo em classes diferentes: 0.

O treino agora usa `data_clean/` por padrao quando essa pasta existe. Se alguem
apontar o treino para `data/`, `data_augmented/` ou `data_augmented_max/`, o
script bloqueia a execucao quando encontra sinais de vazamento. Para auditoria
manual:

```powershell
py .\audit_dataset.py --data-dir data
py .\audit_dataset.py --data-dir data_clean --json-output data_clean\audit_report.json
```

## Comando recomendado

Para treinar de novo com a base limpa:

```powershell
py .\resnet18_soja.py --data-dir data_clean --output-dir resultados_clean --epochs 50 --batch-size 16 --device cpu --num-workers 0
```

Se quiser evitar qualquer dependencia de pesos pre-treinados:

```powershell
py .\resnet18_soja.py --data-dir data_clean --output-dir resultados_clean --epochs 50 --batch-size 16 --device cpu --num-workers 0 --no-pretrained
```

O resultado que importa agora e o `test`, nao apenas o `val`.
