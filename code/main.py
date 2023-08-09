from pipeline import Pipeline
# from trainer import (
#     FineTuner,
#     # PrefixTuner,
#     #AdapterTuner
# )
#from trainer import AdapterTuner
#import pandas as pd
import sys
import argparse
if __name__ == "__main__":

    parser = argparse.ArgumentParser("Evaluate model using zero/Few shot")
    parser.add_argument("--data_dir", default="v5")
    parser.add_argument("--result_dir", default="extractive/zero-shot/eos/")
    parser.add_argument("--model_name_or_path", default="google/flan-t5-small")
    parser.add_argument("--zero_shot", type=bool, default=False)
    parser.add_argument("--instruction_id", type=int, default=None)
    parser.add_argument("--template_name", type=str, default='t5')
    parser.add_argument("--template_id", type=int, default=-1)
    parser.add_argument("--ref_column", type=str, default='true_strong')
    parser.add_argument("--trial", type=str, default='0')
    parser.add_argument("--abstractive", type=bool, default=False)
   
    args, unparsed = parser.parse_known_args()
    #args = parser.parse_args()
    Pipeline(args=args).run()
    #APIPipeline('instructGPT').run()


    #FineTuner('google/flan-t5-large').train()
    #args = sys.argv
    #PrefixTuner().train()
    # AdapterTuner().train()
    #FineTuner().train()
    #google/flan-t5-large
    #FineTuner('google/flan-t5-xl').train()
    