# nlp2026-final-team9-part2
[prefix Tuning 실행 예]
python sonnet_generation.py \
  --use_gpu \
  --epochs 10 \
  --batch_size 8 \
  --lr 2e-4 \
  --model_size gpt2 \
  --prefix_length 10 \
  --prefix_hidden_size 512 \
  --prefix_dropout 0.1 \
  --dev_sonnet_path data/TRUE_sonnets_held_out_dev.txt \
  --held_out_sonnet_path data/sonnets_held_out_dev.txt \
  --temperature 0.7 \
  --top_p 0.85 \
  --max_length 180 \
  --log_path logs/prefix_dev.log \
  --sonnet_out predictions/generated_sonnets_dev.txt
