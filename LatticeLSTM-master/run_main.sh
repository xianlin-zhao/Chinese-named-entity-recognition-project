#python main.py --status train \
#		--train ../data/onto4ner.cn/train.char.bmes \
#		--dev ../data/onto4ner.cn/dev.char.bmes \
#		--test ../data/onto4ner.cn/test.char.bmes \
#		--savemodel ../data/onto4ner.cn/saved_model \

#python main.py --status train \
#		--train ./data/train.char.bmes \
#		--dev ./data/dev.char.bmes \
#		--test ./data/test.char.bmes \
#		--savemodel ./data/my_model/saved_model \

python main.py --status train \
		--train ./data/train_data.txt \
		--dev ./data/val_data.txt \
		--test ./data/test_data.txt \
		--savemodel ./data/my_model/saved_model \


# python main.py --status decode \
# 		--raw ../data/onto4ner.cn/test.char.bmes \
# 		--savedset ../data/onto4ner.cn/saved_model \
# 		--loadmodel ../data/onto4ner.cn/saved_model.13.model \
# 		--output ../data/onto4ner.cn/raw.out \
