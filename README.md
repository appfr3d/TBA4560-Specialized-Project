# Prosjektoppgave

## Hvordan åpne prosjektet

Skriv `prosjektoppgave` i terminalen så flyttes man til riktig mappe og det virituelle python miljet starter.

Skriv `code .` for å åpne prosjektet i VS Code

## Hvordan trene og teste PointNet++ på ShapeNet

Trene: `python3 train_partseg.py --model pointnet2_part_seg_msg --normal --log_dir pointnet2_part_seg_msg --batch_size 12 --gpu "0,1"`

Teste: `python3 test_partseg.py --normal --log_dir pointnet2_part_seg_msg`

## Hvordan trene semantic og instance segmentation PointNet++ på TR3DRoofs

Semantic: `python3 train_partseg_tr3d.py --model pointnet2_part_seg_msg --log_dir pointnet2_part_seg_msg_tr3d --batch_size 8 --gpu "0,1"`

Instance: `python3 train_instseg_tr3d.py --model pointnet2_part_seg_msg --log_dir pointnet2_inst_seg_msg_tr3d --batch_size 8 --gpu "0,1"`

## Hvordan visualisere instance og semantic segmentation på TR3DRoofs

Semantic: `python3 test_partseg_tr3d.py --log_dir pointnet2_part_seg_msg_tr3d`

Instance: `python3 test_instseg_tr3d.py --log_dir pointnet2_inst_seg_msg_tr3d`
