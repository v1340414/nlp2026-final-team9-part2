# Checkpoint 위치

선택된 P3 prefix-tuning checkpoint를 아래 위치에 둔다.

```text
checkpoints/p3_selected_checkpoint.pt
```

사용한 원본 실험:

```text
P3_prefix20_hidden512_lr2e-4
```

checkpoint 파일은 약 300MB로 크기 때문에 repository에 직접 포함하지 않는다.
`download_checkpoint.py`를 사용해 Google Drive에서 다운로드한다.
