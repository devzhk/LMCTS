
class Bandit(object):
    def __init__(self, num_arm,
                 dim_context,
                 datapath):
        '''
        Args: 
            - num_arm: number of arms
            - dim_context: dimensionality of context

        '''
        self.num_arm = num_arm
        self.dim_context = dim_context
