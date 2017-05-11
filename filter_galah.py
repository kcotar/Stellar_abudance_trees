import numpy as np


class FILTER:
    def __init__(self, verbose=False):
        """

        :param verbose:
        """
        self.steps = list([])  # TODO string list of applied filter
        self.idx_ok = None
        self.n_ok = 0
        self.verbose = verbose
        self.logical = {'==': np.equal,
                        '!=': np.not_equal,
                        '<' : np.less,
                        '<=': np.less_equal,
                        '>' : np.greater,
                        '>=': np.greater_equal}

    def apply_filter(self, data):
        """

        :param data:
        :return:
        """
        if self.idx_ok is None:
            raise RuntimeError('Data filter not yet defined')
        if self.verbose:
            print 'Data lines that will be used: '+str(self.n_ok)+'.'
        return data[self.idx_ok]

    def _merge_ok(self, idx):
        """

        :param idx:
        :return:
        """
        if self.idx_ok is None:
            self.idx_ok = idx
        else:
            self.idx_ok = np.logical_and(self.idx_ok, idx)
        self.n_ok = np.sum(self.idx_ok)

    def _get_logical_operation(self, logical_str):
        """

        :param logical_str:
        :return:
        """
        if not self.logical.has_key(logical_str):
            raise AttributeError('Unrecognised comparator sign: '+logical_str+'.')
        else:
            return self.logical[logical_str]

    def filter_attribute(self, data, attribute=None, value=None, comparator='!='):
        """

        :param data:
        :param attribute:
        :param value: if two values are supplied, threaded as a range from min to max
        :param comparator:
        :return:
        """
        if attribute is None:
            raise AttributeError('Filter attribute not given.')
        else:
            if value is None:
                raise AttributeError('Filter value not given.')
            else:
                if isinstance(value, list):
                    idx_use = np.logical_and(data[attribute] >= value[0],
                                             data[attribute] <= value[1])
                else:
                    eval_func = self._get_logical_operation(comparator)
                    idx_use = eval_func(data[attribute], value)
        self._merge_ok(idx_use)

    def filter_objects(self, data, object_list, identifier='sobject_id'):
        """

        :param data:
        :param object_list:
        :param identifier:
        :return:
        """
        if identifier not in data.colnames or identifier not in object_list.colnames:
            raise SyntaxError('Identifier '+identifier+' not present in one of the data sets.')
        else:
            idx_use = np.in1d(data[identifier], object_list[identifier], assume_unique=True, invert=True)
        self._merge_ok(idx_use)

    def filter_valid_rows(self, data, cols=None):
        """

        :param data:
        :param cols:
        :return:
        """
        if cols is None:
            cols = data.colnames
        idx_use = np.isfinite(data[cols].to_pandas().values).all(axis=1)
        self._merge_ok(idx_use)
