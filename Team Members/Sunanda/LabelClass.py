class LabelCountEncoder(object):
    def __init__(self):
        self.count_dict = {}

    def fit(self, column):
        # This gives you a dictionary with level as the key and counts as the value
        count = column.value_counts()#.to_dict()
        # We want to rank the key by its value and use the rank as the new value
        self.count_dict = dict(zip(count.index,range(len(count),0,-1)))

    def transform(self, column):
        # If a category only appears in the test set, we will assign the value to zero.
        missing = 0
        # Your code here
        return column.map(lambda x: self.count_dict.get(x,missing)) # 2nd get input is if it doesnt exist

    def fit_transform(self, column):
        self.fit(column)
        return self.transform(column)
