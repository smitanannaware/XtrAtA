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
    def format(self, instruction, ref_column = 'true_strong', domainQ = None, domainA = None, return_field_name= 'formatted_output', **data_loader_args):
        formatted_data_dict = {}
        dataset = self.data_loader.dataset
        
        for data in dataset.keys():
            formatted_data = []
            hide_label = True
            if data in ['train', 'dev']:
                hide_label = False
            for data_dict in dataset[data]:
                data_dict['instruction'] = instruction
                data_dict['label'] = data_dict[ref_column]
                if domainQ and domainA:
                    data_dict['domainQ'] = domainQ
                    data_dict['domainA'] = domainA
                if return_field_name:
                    data_dict[return_field_name] = super().format(data_dict, hide_label)
                else:
                    formatted_data.append(super().format(data_dict, hide_label))
            if return_field_name is None:
                formatted_data_dict[data] = formatted_data
        if return_field_name:
            return dataset
        else:
            return formatted_data_dict
        
        
    def formatFewShotExample(self, instruction, fewShotEx, ref_column = 'true_strong', domainQ = None, domainA = None, return_field_name= 'formatted_output', **data_loader_args):
        formatted_data_dict = {}
        dataset = self.data_loader.dataset
        
        for data in dataset.keys():
            formatted_data = []
            hide_label = True
            if data in ['train', 'dev']:
                hide_label = False
            output = ''
            if hide_label:
                output = '\n\n'.join([train_ex[return_field_name] for train_ex in fewShotEx]) + '\n\n'
            for data_dict in dataset[data]:
                if domainQ and domainA:
                    data_dict['domainQ'] = domainQ
                    data_dict['domainA'] = domainA
                data_dict['label'] = data_dict[ref_column]
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


    def format_labels(self, bnps):
        bnp_list = bnps.split(', ')
        if len(bnp_list) > 1:
            bnp_list[-1] = 'and ' + bnp_list[-1].strip()
        return ", ".join(bnp_list)


    def fine_tune_format(self, instruction, ref_column = 'true_strong', domainQ = None, domainA = None, abstractive=False, format_labels = False, return_field_name= 'formatted_output', **data_loader_args):
        formatted_data_dict = {}
        dataset = self.data_loader.dataset
        hide_label = True
        for data in dataset.keys():
            formatted_data = []
            for data_dict in dataset[data]:
                data_dict['instruction'] = instruction
                if domainQ and domainA:
                    data_dict['domainQ'] = domainQ
                    data_dict['domainA'] = domainA
                data_dict['label'] = data_dict[ref_column]
                if return_field_name:
                    data_dict[return_field_name] = super().format(data_dict, hide_label)
                else:
                    formatted_data.append(super().format(data_dict, hide_label))
            if return_field_name is None:
                formatted_data_dict[data] = formatted_data
        if return_field_name:
            return dataset
        else:
            return formatted_data_dict
            
    def fine_tune_reasoning_format(self, instruction, domainQ = None, domainA = None, type='NC', return_field_name= 'formatted_output', **data_loader_args):
        formatted_data_dict = {}
        dataset = self.data_loader.dataset
        hide_label = True
        for data in dataset.keys():
            formatted_data = []
            for data_dict in dataset[data]:
                data_dict['instruction'] = instruction
                if type == 'NC':
                    data_dict['label'] = f'Reasoning:\n' + data_dict['reasoning'] + '\nAtypical aspects:' + data_dict['true_nc']
                else:
                    data_dict['label'] = self.format_labels(data_dict['true_nc_oc'])
                if return_field_name:
                    data_dict[return_field_name] = super().format(data_dict, hide_label)
                else:
                    formatted_data.append(super().format(data_dict, hide_label))
            if return_field_name is None:
                formatted_data_dict[data] = formatted_data
        if return_field_name:
            return dataset
        else:
            return formatted_data_dict

        