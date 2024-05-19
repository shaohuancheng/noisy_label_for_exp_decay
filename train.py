from dataclasses import dataclass, field
from typing import Optional
import sys, os
from datasets_fig import *
from model import *
from trainer import SelfMixTrainer

from transformers import (
    AutoModel,
    HfArgumentParser,
    set_seed,
    AutoTokenizer,
)

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
logger = logging.getLogger(__name__)


def print_args(args):
    # print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    # print('--------args----------\n')

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune. 
    """
    
    # Huggingface's original arguments
    pretrained_model_name_or_path: Optional[str] = field(
        default='bert-base-uncased',
        metadata={
            "help": "The pretrained model checkpoint for weights initialization."
        },
    )
    dropout_rate: float = field(
        default=0.1,
        metadata={"help": "Dropout rate"}
    )
    temp: float = field(
        default=0.5,
        metadata={"help": "Temperature for sharpen function"}
    )

    p_threshold: float = field(
        default=0,
        metadata={"help": "Lower bound for exp decay"}
    )
    exp: int = field(
        default=3,
        metadata={"help": "Adjust the magnitude of exp decay."}
    )
    

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of dataset"}
    )
    train_file_path: Optional[str] = field(
        default=None,
        metadata={"help": "The train data file (.csv)"}
    )
    eval_file_path: Optional[str] = field(
        default=None,
        metadata={"help": "The eval data file (.csv)"}
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size"}
    )
    max_sentence_len: Optional[int] = field(
        default=256,
        metadata={
            "help": "The maximum total input sentence length after tokenization. Sequences longer."
        },
    )


@dataclass
class OurTrainingArguments:
    seed: Optional[int] = field(
        default=1,
        metadata={"help": "Seed"}
    )
    warmup_epochs: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of epochs to warmup the model"
            "only one of the warmup_epochs and warmup_samples should be specified"
        }
    )
    lambda_r: float = field(
        default=0.3,
        metadata={"help": "Weight for R-Drop loss"}
    )
    split_num: int = field(
        default=20,
        metadata={"help": "eval step split"}
    )
    ration: int = field(
        default=20,
        metadata={"help": "eval step split"}
    )
    train_epochs: int = field(
        default=4,
        metadata={"help": "Mix-up training epochs"}
    )
    lr: float = field(
        default=1e-5,
        metadata={"help": "Learning rate"}
    )
    model_save_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to save model"}
    )
    record_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to save records"}
    )

    
def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    logger.info("Model Parameters %s", model_args)
    logger.info("Data Parameters %s", data_args)
    logger.info("Training Parameters %s", training_args)
    # print_args(model_args)
    print_args(data_args)
    print_args(training_args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(training_args.seed)
    
    # load data
    train_datasets, train_num_classes = load_dataset(data_args.train_file_path, data_args.dataset_name)
    eval_datasets, eval_num_classes = load_dataset(data_args.eval_file_path, data_args.dataset_name)
    assert train_num_classes == eval_num_classes
    model_args.num_classes = train_num_classes
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_model_name_or_path)
    selfmix_train_data = SelfMixData(data_args, train_datasets, tokenizer)
    selfmix_eval_data = SelfMixData(data_args, eval_datasets, tokenizer)
    
    # load model
    model = Bert4Classify(model_args.pretrained_model_name_or_path, model_args.dropout_rate, model_args.num_classes)
    
    # build trainer
    trainer = SelfMixTrainer(
        model=model,
        train_data=selfmix_train_data,
        eval_data=selfmix_eval_data,
        model_args=model_args,
        training_args=training_args
    )
    
    # train and eval
    test_best_l, test_last_l = trainer.dynamic_train()
    print('test_best', test_best_l)
    print('test_last', test_last_l)
    print("Test best %f , last %f" % (test_best_l[-1], test_last_l[-1]))

    # records = {"Last": [test_last_l[-1]], "Best": [test_best_l[-1]]}
    # records = pd.DataFrame(records)
    # records.to_csv(training_args.record_path, index=False)


if __name__ == '__main__':
    main()
