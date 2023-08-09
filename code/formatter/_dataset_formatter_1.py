from prompts import PromptTemplate
from ._template_formatter import TemplateFormatter
class DatasetFormatter(TemplateFormatter):
    """
    A formatter that uses a template class to format an entire dataset through a DataLoader.
    Example:
    >>> template = PromptTemplate('John said:\n{story}\nlabel: {label}')
    >>> data_loader = StoryDataLoader()
    >>> formatter = DatasetFormatter(template, data_loader)
    >>> print(data_loader.load_dataset())
    [{'id: 1, 'story_column': 'This is a happy story.' , 'label_column': 'happy'}, {'id: 2, 'story_column': 'This is a sad story.', label_column: 'sad'}]
    >>> formatter.format({'story': 'story_column', 'label': 'label_column'})
    [{'id: 1, 'story_column': 'John said:\nThis is a happy story.\nlabel: happy'}, {'id: 2, 'story_column': 'John said:\nThis is a sad story.\nlabel: sad'}]
    """

    def __init__(self, template:PromptTemplate, data_loader):
        super().__init__(template)
        self.data_loader = data_loader
            

    """
    Format a dataset using a template by inserting the values in the data records of data loader into the template.
    :param data_dict: A dictionary of template variables and their values.
    :param data_loader: A DataLoader object that loads the dataset.
    :param data_loader_args: Arguments to pass to the data
    :return: A formatted dataset.
    """
    def format(self, instruction, domainQ = None, domainA = None, return_field_name= 'formatted_output', **data_loader_args):
        formatted_data_dict = {}
        dataset = self.data_loader.dataset
        
        for data in dataset.keys():
            formatted_data = []
            hide_label = True
            if data == 'train':
                hide_label = False
            for data_dict in dataset[data]:
                data_dict['instruction'] = instruction
                if domainQ and domainA:
                    data_dict['domainQ'] = domainQ
                    data_dict['domainA'] = domainA
                if return_field_name:
                    #print(super().format(data_dict, hide_label))
                    data_dict[return_field_name] = super().format(data_dict, hide_label)
                else:
                    formatted_data.append(super().format(data_dict, hide_label))
            if return_field_name is None:
                formatted_data_dict[data] = formatted_data
        if return_field_name:
            return dataset
        else:
            return formatted_data_dict
        
        
    def formatFewShotExample(self, instruction, fewShotEx=2, return_field_name= 'formatted_output', **data_loader_args):
        formatted_data_dict = {}
        dataset = self.data_loader.dataset
        
        for data in dataset.keys():
            formatted_data = []
            hide_label = True
            if data == 'train':
                hide_label = False
            output = ''
            if hide_label:
                output = '\n\n'.join([train_ex[return_field_name] for train_ex in dataset['train'][:fewShotEx]]) + '\n\n'
            for data_dict in dataset[data]:
                data_dict['instruction'] = instruction
                if return_field_name:
                    data_dict[return_field_name] = output + super().format(data_dict, hide_label)
                else:
                    formatted_data.append(output + super().format(data_dict, hide_label))
            if return_field_name is None:
                formatted_data_dict[data] = formatted_data
        if return_field_name:
            return dataset
        else:
            return formatted_data_dict




            

        