import argparse
from asyncio.log import logger
from Layoutlmv3_inference.ocr import prepare_batch_for_inference
from Layoutlmv3_inference.inference_handler import handle
import logging
import os

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", type=str)
        parser.add_argument("--images_path", type=str)
        args, _ = parser.parse_known_args()
        images_path = args.images_path
        image_files = os.listdir(images_path)
        images_path = [images_path+f'/{image_file}' for image_file in image_files]
        
        # use current working directory as base dir
        base_dir = os.getcwd()
        
        inference_batch = prepare_batch_for_inference(images_path, base_dir)
        context = {"model_dir": args.model_path}
        handle(inference_batch,context,base_dir)
    except Exception as err:
        os.makedirs('log', exist_ok=True)
        logging.basicConfig(filename='log/error_output.log', level=logging.ERROR,
                            format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        logger.error(err)


