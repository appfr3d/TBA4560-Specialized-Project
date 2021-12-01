# Prosjektoppgave

## Hvordan åpne prosjektet

Skriv `prosjektoppgave` i terminalen så flyttes man til riktig mappe og det virituelle python miljet starter.

Skriv `code .` for å åpne prosjektet i VS Code

## Hvordan trene og teste PointNet++ på ShapeNet

Trene: `python3 train_partseg.py --model pointnet2_part_seg_msg --normal --log_dir pointnet2_part_seg_msg --batch_size 12 --gpu "0,1"`

Teste: `python3 test_partseg.py --normal --log_dir pointnet2_part_seg_msg`

## Hvordan trene og teste instance segmentation PointNet++ på TR3DRoofs

Trene: `python3 train_instseg.py --model pointnet2_part_seg_msg --log_dir pointnet2_part_seg_msg --batch_size 12 --gpu "0,1"`

Teste: `python3 test_instseg.py --log_dir pointnet2_part_seg_msg`
