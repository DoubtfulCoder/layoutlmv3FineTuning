import os
from .utils import load_model,load_processor,normalize_box,compare_boxes,adjacent
from .annotate_image import get_flattened_output,annotate_image
from PIL import Image,ImageDraw, ImageFont
import logging
import torch
import json
import math


logger = logging.getLogger(__name__)

class ModelHandler(object):
    """
    A base Model handler implementation.
    """

    def __init__(self):
        self.model = None
        self.model_dir = None
        self.device = 'cpu'
        self.error = None
        # self._context = None
        # self._batch_size = 0
        self.initialized = False
        self._raw_input_data = None
        self._processed_data = None
        self._images_size = None

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        logger.info("Loading transformer model")

        self._context = context
        properties = self._context
        # self._batch_size = properties["batch_size"] or 1
        self.model_dir = properties.get("model_dir")
        self.model = self.load(self.model_dir)
        self.initialized = True

    def preprocess(self, batch, using_lilt=False):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and pre-process it make it inference ready
        # assert self._batch_size == len(batch), "Invalid input batch size: {}".format(len(batch))
        inference_dict = batch
        self._raw_input_data = inference_dict
        processor = load_processor()
        images = [Image.open(path).convert("RGB")
                  for path in inference_dict['image_path']]
        self._images_size = [img.size for img in images]
        words = inference_dict['words']
        boxes = [[normalize_box(box, images[i].size[0], images[i].size[1])
                  for box in doc] for i, doc in enumerate(inference_dict['bboxes'])]
        encoded_inputs = processor(
            images, words, boxes=boxes, return_tensors="pt", padding="max_length", truncation=True)
        if using_lilt:
            del encoded_inputs['pixel_values']
        self._processed_data = encoded_inputs
        return encoded_inputs

    def load(self, model_dir):
        """The load handler is responsible for loading the hunggingface transformer model.
        Returns:
            hf_pipeline (Pipeline): A Hugging Face Transformer pipeline.
        """
        # TODO model dir should be microsoft/layoutlmv2-base-uncased
        model = load_model(model_dir)
        return model

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # TODO load the model state_dict before running the inference
        # Do some inference call to engine here and return output
        with torch.no_grad():
            inference_outputs = self.model(**model_input)
            predictions = inference_outputs.logits.argmax(-1).tolist()
        results = []
        for i in range(len(predictions)):
            tmp = dict()
            tmp[f'output_{i}'] = predictions[i]
            results.append(tmp)

        return [results]

    def postprocess(self, inference_output):
        docs = []
        k = 0
        for page, doc_words in enumerate(self._raw_input_data['words']):
            doc_list = []
            width, height = self._images_size[page]
            for i, doc_word in enumerate(doc_words, start=0):
                word_tagging = None
                word_labels = []
                word = dict()
                word['id'] = k
                k += 1
                word['text'] = doc_word
                word['pageNum'] = page + 1
                word['box'] = self._raw_input_data['bboxes'][page][i]
                _normalized_box = normalize_box(
                    self._raw_input_data['bboxes'][page][i], width, height)
                for j, box in enumerate(self._processed_data['bbox'].tolist()[page]):
                    if compare_boxes(box, _normalized_box):
                        if self.model.config.id2label[inference_output[0][page][f'output_{page}'][j]] != 'O':
                            word_labels.append(
                                self.model.config.id2label[inference_output[0][page][f'output_{page}'][j]][2:])
                        else:
                            word_labels.append('other')
                if word_labels != []:
                    word_tagging = word_labels[0] if word_labels[0] != 'other' else word_labels[-1]
                else:
                    word_tagging = 'other'
                word['label'] = word_tagging
                word['pageSize'] = {'width': width, 'height': height}
                if word['label'] != 'other':
                    doc_list.append(word)
            spans = []
            def adjacents(entity): return [
                adj for adj in doc_list if adjacent(entity, adj)]
            output_test_tmp = doc_list[:]
            for entity in doc_list:
                if adjacents(entity) == []:
                    spans.append([entity])
                    output_test_tmp.remove(entity)

            while output_test_tmp != []:
                span = [output_test_tmp[0]]
                output_test_tmp = output_test_tmp[1:]
                while output_test_tmp != [] and adjacent(span[-1], output_test_tmp[0]):
                    span.append(output_test_tmp[0])
                    output_test_tmp.remove(output_test_tmp[0])
                spans.append(span)

            output_spans = []
            for span in spans:
                if len(span) == 1:
                    output_span = {"text": span[0]['text'],
                                   "label": span[0]['label'],
                                   "words": [{
                                       'id': span[0]['id'],
                                       'box': span[0]['box'],
                                       'text': span[0]['text']
                                   }],
                                   }
                else:
                    output_span = {"text": ' '.join([entity['text'] for entity in span]),
                                   "label": span[0]['label'],
                                   "words": [{
                                       'id': entity['id'],
                                       'box': entity['box'],
                                       'text': entity['text']
                                   } for entity in span]

                                   }
                output_spans.append(output_span)
            docs.append({f'output': output_spans})
        return [json.dumps(docs, ensure_ascii=False)]

    @staticmethod
    def combine_json_outputs(num_iters):
        json_outputs = []
        combined_json_output = [{'output': []}]

        # read all the json outputs and put them in a list
        for i in range(num_iters):
            with open(f'LayoutlMV3InferenceOutput_{i}.json', 'r') as inf_out:
                json_outputs.append(json.load(inf_out))
        
        i = 0
        for file in json_outputs:
            output = file[0]['output']
            for entity in output:
                bbox_1 = entity['words'][0]['box']
                # check if the entity is already in the combined json output
                already_exists = False
                for existing_entity in combined_json_output[0]['output']:
                    bbox_2 = existing_entity['words'][0]['box']
                    if bbox_1 == bbox_2:
                        already_exists = True
                        break

                if not already_exists:
                    combined_json_output[0]['output'].append(entity)
                
            
            i += 1

        return combined_json_output

    def handle(self, data, context, base_path, using_lilt=False):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """

        words = data['words']
        num_words = len(words[0])
        # number of iters = number of times we can slide windows of 300 words by 200 at a time 
        # (100 word overlap between windows to not miss context)
        # e.g. if we have 620 words, iters are [0, 300], [200, 500], [400, 620]
        num_iters = max(0, num_words - 300) 
        num_iters = math.ceil(num_iters / 200) + 1 # divide by 200 and add 1
        print('num iters = ', num_iters)
        print('base path', base_path)

        start_idx = 0
        for i in range(num_iters):
            end_idx = start_idx + 300 # we take 300 words at a time
            if end_idx > num_words:
                end_idx = num_words # if we are at the last iteration, we take the remaining words
             
            # extract subset of words for current window
            subset_words = [words[0][start_idx:end_idx]]
            subset_bboxes = [data['bboxes'][0][start_idx:end_idx]]
            image_path = data['image_path']
            new_data = {'words': subset_words, 'bboxes': subset_bboxes, 'image_path': image_path}
            
            # process and run inference
            model_input = self.preprocess(new_data, using_lilt)
            inference_output = self.inference(model_input)
            result = self.postprocess(inference_output)[0]
            
            # export inference output to json
            print(f'Exporting inference output to json file LayoutlMV3InferenceOutput_{i}.json')
            # print(result)
            with open(f'LayoutlMV3InferenceOutput_{i}.json', 'w') as inf_out:
                inf_out.write(result)

            # annotate image using json output
            inference_out_list = json.loads(result)
            flattened_output_list = get_flattened_output(inference_out_list)
            for j, flattened_output in enumerate(flattened_output_list):
              annotate_image(data['image_path'][j], flattened_output, base_path)

            # rename inference image name to add number of iteration
            image_name = os.path.basename(image_path[0])
            image_name = image_name[:image_name.find('.')]
            inference_img_path = f'{base_path}/{image_name}_inference.jpg'
            new_inference_img_path = f'{base_path}/{image_name}_inference_{i}.jpg'
            os.rename(inference_img_path, new_inference_img_path)
            
            # slide the window by 200 words (100 word overlap with previous window)
            start_idx += 200

        # combine json outputs into one json file
        combined_json = ModelHandler.combine_json_outputs(num_iters)
        print('combined json output: ', combined_json[:50])
        with open('LayoutlMV3InferenceOutput.json', 'w') as combined_json_out:
            combined_json_out.write(json.dumps(combined_json, ensure_ascii=False))

        # combined inference image
        # inference_out_list = json.loads(combined_json_output)
        flattened_output_list = get_flattened_output(combined_json)
        for j, flattened_output in enumerate(flattened_output_list):
          annotate_image(data['image_path'][j], flattened_output, base_path)



            


_service = ModelHandler()


def handle(data, context, base_path, using_lilt=False):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context, base_path, using_lilt)


