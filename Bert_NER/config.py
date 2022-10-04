import transformers

MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
EPOCHS = 5
MODEL_PATH = 'model.bin'
NER_LABELS = ['O', 'B-CN', 'I-CN', 'B-PN', 'I-PN', 'B-PV', 'I-PV']