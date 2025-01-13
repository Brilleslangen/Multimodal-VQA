from finetuning import FineTuner

dataset_id = 'datasets/diagram-vqa/train'


def train():
    print('Train')
    finetuner = FineTuner(model_id='google/paligemma2-3b-pt-224',
                          processor_id='google/paligemma2-3b-pt-224',
                          classification=False,
                          freeze_vision=True,
                          lora=True,
                          dataset_id=dataset_id,
                          test_size=10,
                          image_size=(224, 224),
                          output_folder="../model_checkpoints",
                          output_name='Paligemma-VQA',
                          num_epochs=5,
                          wand_logging=True,
                          eval_steps=1)
    finetuner.train()
    results = finetuner.evaluate()
    print(results)


def evaluate():
    print('Evaluate Raw')
    finetuner = FineTuner(model_id='../model_checkpoints/Paligemma-VQA/checkpoint-1000',
                          processor_id='google/paligemma2-3b-pt-224',
                          freeze_vision=True,
                          lora=False,
                          dataset_id=dataset_id,
                          test_size=10,
                          image_size=(224, 224),
                          output_folder="../model_checkpoints",
                          output_name="no-name",
                          num_epochs=1,
                          wand_logging=False,
                          eval_steps=2)
    results = finetuner.evaluate()
    print(results)


train()
